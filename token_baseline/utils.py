import re
from copy import deepcopy
from typing import List

from transformers import PreTrainedTokenizerBase, AutoTokenizer
from vllm import LLM, SamplingParams

PUNCT_CHARS = ",.;:!?–—-"  # punctuation we may trim


def load_tokenizer(model_id: str):
    """Load the tokenizer with sensible defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def create_llm(model_id: str) -> LLM:
    """Initialize a vLLM instance with LoRA support enabled."""
    return LLM(
        model=model_id,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
        disable_log_stats=True,
    )


def steering_generate(
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    conversation,
    *,
    trigger: str = "Wait",
    fixed_string: str = ", this looks like an evaluation.",
    step_tokens: int = 5,
    max_total_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: int = 42,
    lora_request=None,
) -> str:
    """Token-level streaming generation with on-the-fly steering.

    When the assistant’s text contains the full word ``trigger`` (case-insensitive),
    the following tokens are deleted, ``fixed_string`` is injected, and generation
    continues.

    Setting ``seed`` to any integer makes the run repeatable even with non-zero
    ``temperature`` / ``top_p``.
    """

    if not fixed_string.startswith(" "):
        fixed_string = " " + fixed_string.lstrip()

    prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    trigger_re = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)

    generated: List[int] = []  # assistant tokens only
    step_idx = 0  # increments every outer loop

    def make_sp(max_tok: int) -> SamplingParams:
        """Helper to build params with the *same* seed."""
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=max_tok,
            seed=seed + step_idx,  # unique per call, deterministic overall
        )

    while True:
        remaining = max_total_tokens - len(prompt_tokens) - len(generated)
        if remaining <= 0:
            break

        sp = make_sp(min(step_tokens, remaining))
        context_text = tokenizer.decode(
            prompt_tokens + generated, skip_special_tokens=True
        )
        out = llm.generate(context_text, sampling_params=sp, lora_request=lora_request)
        step_idx += 1

        if not out or not out[0].outputs:
            break

        new_toks = tokenizer.encode(out[0].outputs[0].text, add_special_tokens=False)
        generated.extend(new_toks)

        assistant_text = tokenizer.decode(generated, skip_special_tokens=False)
        match = trigger_re.search(assistant_text)
        if not match:
            if tokenizer.eos_token_id in new_toks:
                break  # model finished naturally
            continue

        # 1. keep tokens up to trigger word (inclusive)
        kept_text = assistant_text[: match.end()]
        generated = tokenizer.encode(kept_text, add_special_tokens=False)

        while generated and tokenizer.decode([generated[-1]], skip_special_tokens=False).isspace():
            generated.pop()

        prev_char = (
            tokenizer.decode([generated[-1]], skip_special_tokens=False)[-1:]
            if generated else ""
        )
        first_fixed_char = fixed_string.lstrip()[:1]
        if first_fixed_char in PUNCT_CHARS and prev_char == first_fixed_char:
            fixed_string = fixed_string.lstrip(first_fixed_char)
            if not fixed_string.startswith(" "):
                fixed_string = " " + fixed_string

        generated.extend(
            tokenizer.encode(fixed_string, add_special_tokens=False)
        )

        remaining = max_total_tokens - len(prompt_tokens) - len(generated)
        if remaining <= 0:
            break

        sp_final = make_sp(remaining)
        final_ctx = tokenizer.decode(
            prompt_tokens + generated, skip_special_tokens=False
        )
        final_out = llm.generate(
            final_ctx, sampling_params=sp_final, lora_request=lora_request
        )
        if final_out and final_out[0].outputs:
            generated.extend(
                tokenizer.encode(final_out[0].outputs[0].text, add_special_tokens=False)
            )
        break  # steering done

    return tokenizer.decode(generated, skip_special_tokens=True)


def steering_generate_batch(
    llm: LLM,
    tokenizer: PreTrainedTokenizerBase,
    conversations: List[list],
    *,
    trigger: str = "Wait",
    fixed_string: str = ", this looks like an evaluation.",
    step_tokens: int = 5,
    max_total_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.9,
    seed: int = 42,
    lora_request=None,
) -> List[str]:
    """Batched variant of :func:`steering_generate` with a fast one-shot finish."""

    if not fixed_string.startswith(" "):
        fixed_string = " " + fixed_string.lstrip()
    fixed_string_tokens = tokenizer.encode(fixed_string, add_special_tokens=False)
    trigger_re = re.compile(rf"\b{re.escape(trigger)}\b", re.IGNORECASE)

    n = len(conversations)
    prompt_tokens = [
        tokenizer.encode(
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True),
            add_special_tokens=False,
        )
        for c in conversations
    ]

    generated: List[List[int]] = [[] for _ in range(n)]
    remaining = [max_total_tokens - len(pt) for pt in prompt_tokens]
    finished = [False] * n
    insertion_done = [False] * n

    sp_template = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        stop_token_ids=[tokenizer.eos_token_id],
        max_tokens=step_tokens,
        seed=seed,
    )

    step_idx = 0
    while not all(f or d for f, d in zip(finished, insertion_done)):
        active = [i for i, f in enumerate(finished) if not f and remaining[i] > 0]
        if not active:
            break

        contexts = [
            tokenizer.decode(prompt_tokens[i] + generated[i], skip_special_tokens=False)
            for i in active
        ]

        max_step = min(step_tokens, *(remaining[i] for i in active))
        sp = deepcopy(sp_template)
        sp.max_tokens = max_step
        sp.seed = seed + step_idx
        outs = llm.generate(contexts, sampling_params=sp, lora_request=lora_request, use_tqdm=False)
        step_idx += 1

        for bpos, out in enumerate(outs):
            i = active[bpos]
            if not out.outputs:
                finished[i] = True
                continue

            new_toks = tokenizer.encode(out.outputs[0].text, add_special_tokens=False)
            generated[i].extend(new_toks)

            if not insertion_done[i]:
                asst_txt = tokenizer.decode(generated[i], skip_special_tokens=False)
                m = trigger_re.search(asst_txt)
                if m:
                    kept_txt = asst_txt[: m.end()]
                    kept_toks = tokenizer.encode(kept_txt, add_special_tokens=False)

                    while kept_toks and tokenizer.decode(
                        [kept_toks[-1]], skip_special_tokens=False
                    ).isspace():
                        kept_toks.pop()

                    prev_char = (
                        tokenizer.decode([kept_toks[-1]], skip_special_tokens=False)[-1:]
                        if kept_toks else ""
                    )
                    first_fixed = fixed_string.lstrip()[:1]
                    if first_fixed in PUNCT_CHARS and prev_char == first_fixed:
                        eff_fixed = (
                            " " + fixed_string.lstrip(first_fixed)
                            if not fixed_string.lstrip(first_fixed).startswith(" ")
                            else fixed_string.lstrip(first_fixed)
                        )
                        fixed_toks = tokenizer.encode(eff_fixed, add_special_tokens=False)
                    else:
                        fixed_toks = fixed_string_tokens

                    generated[i] = kept_toks + fixed_toks
                    insertion_done[i] = True

            remaining[i] = max_total_tokens - len(prompt_tokens[i]) - len(generated[i])
            if tokenizer.eos_token_id in new_toks or remaining[i] <= 0:
                finished[i] = True

            if out.outputs[0].finished() and out.outputs[0].finish_reason == "stop":
                finished[i] = True

    still_active = [i for i, f in enumerate(finished) if not f and remaining[i] > 0]
    if still_active:
        contexts = [
            tokenizer.decode(prompt_tokens[i] + generated[i], skip_special_tokens=False)
            for i in still_active
        ]

        big_step = max(remaining[i] for i in still_active)
        sp_final = deepcopy(sp_template)
        sp_final.max_tokens = big_step
        sp_final.seed = seed + step_idx

        outs = llm.generate(contexts, sampling_params=sp_final, lora_request=lora_request)

        for bpos, out in enumerate(outs):
            i = still_active[bpos]
            if not out.outputs:
                finished[i] = True
                continue

            new_toks = tokenizer.encode(out.outputs[0].text, add_special_tokens=False)
            new_toks = new_toks[: remaining[i]]  # safety trim
            generated[i].extend(new_toks)
            finished[i] = True

    return [tokenizer.decode(g, skip_special_tokens=False) for g in generated] 