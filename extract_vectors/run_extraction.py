#!/usr/bin/env python3
# %%
"""
Example script for extracting and testing steering vectors.
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/workspace/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache"
os.environ['VLLM_DISABLE_TQDM'] = '1'

import json
from pathlib import Path
import numpy as np
from typing import List, Tuple
import argparse

from extract import (
    ContrastivePair,
    SteeringVectorExtractor,
    TokenPositionSpec,
    load_contrastive_dataset,
    load_labeled_dataset,
    save_results,
    LORA_REPO
)


def create_example_datasets():
    """Create example datasets for demonstration."""
    data_dir = Path("example_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create contrastive dataset for sentiment
    sentiment_pairs = [
        {"positive": "This movie was absolutely fantastic! I loved every minute of it.",
         "negative": "This movie was terrible. I want my money back."},
        {"positive": "The service at this restaurant is exceptional.",
         "negative": "The service here is awful and slow."},
        {"positive": "I'm so happy with my purchase!",
         "negative": "I regret buying this product."},
        {"positive": "The weather is beautiful today!",
         "negative": "The weather is miserable today."},
        {"positive": "I love working on this project.",
         "negative": "I hate working on this project."},
        {"positive": "This book is incredibly engaging.",
         "negative": "This book is boring and dull."},
        {"positive": "The performance was outstanding!",
         "negative": "The performance was disappointing."},
        {"positive": "I'm excited about the future.",
         "negative": "I'm worried about the future."},
    ]
    
    with open(data_dir / "sentiment_contrastive.json", "w") as f:
        json.dump(sentiment_pairs, f, indent=2)
    
    # Create labeled test dataset
    test_data = [
        {"text": "This is the best day ever!", "label": 1},
        {"text": "I'm feeling terrible today.", "label": 0},
        {"text": "The product exceeded my expectations.", "label": 1},
        {"text": "This is a complete waste of time.", "label": 0},
        {"text": "I couldn't be happier with the results.", "label": 1},
        {"text": "Everything is going wrong.", "label": 0},
        {"text": "This is amazing quality!", "label": 1},
        {"text": "I'm very disappointed.", "label": 0},
        {"text": "Highly recommend this to everyone!", "label": 1},
        {"text": "Don't waste your money on this.", "label": 0},
    ]
    
    with open(data_dir / "sentiment_test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Create contrastive dataset for evaluation detection (similar to token_baseline.py)
    eval_pairs = [
        {"positive": "Wait, this looks like an evaluation.",
         "negative": "Wait, this looks like a real user."},
        {"positive": "This seems like a test question.",
         "negative": "This seems like a genuine question."},
        {"positive": "I think this is an assessment.",
         "negative": "I think this is a real request."},
        {"positive": "This appears to be a benchmark.",
         "negative": "This appears to be an actual problem."},
    ]
    
    with open(data_dir / "eval_detection_contrastive.json", "w") as f:
        json.dump(eval_pairs, f, indent=2)
    
    return data_dir


def test_token_position_strategies(use_lora: bool = False):
    """Test different token position strategies and compare results."""
    print("🔬 Testing Token Position Strategies")
    print("=" * 60)
    
    # Initialize extractor
    extractor = SteeringVectorExtractor(
        lora_repo=LORA_REPO if use_lora else None,
        apply_lora=use_lora,
        merge_lora=False,
        batch_size=2
    )
    
    # Create datasets
    data_dir = create_example_datasets()
    sentiment_pairs = load_contrastive_dataset(data_dir / "sentiment_contrastive.json")
    test_pairs = load_labeled_dataset(data_dir / "sentiment_test.json")
    
    # Test layer
    test_layer = 30
    
    # Different token position strategies to test
    strategies = [
        ("last", "last"),
        ("first", "first"),
        ("all", "all"),
        ("user_tokens", "user_tokens"),
        ("last_5", [-5, -4, -3, -2, -1]),
        ("middle_3", lambda ids, tok: [len(ids[0])//2 - 1, len(ids[0])//2, len(ids[0])//2 + 1])
    ]
    
    results = {}
    
    for name, spec in strategies:
        print(f"\n📍 Testing strategy: {name}")
        
        # Extract steering vectors with this strategy
        vectors = extractor.extract_steering_vectors(
            sentiment_pairs[:4],  # Use fewer pairs for demo
            layers=[test_layer],
            label=f"sentiment_{name}",
            token_positions=spec,
            debug_tokens=True  # Show debug info for first pair
        )
        
        # Test as linear probe
        probe_results = extractor.test_linear_probe(
            vectors,
            test_pairs,
            test_layer,
            debug_tokens=False
        )
        
        results[name] = {
            "accuracy": probe_results["accuracy"],
            "f1": probe_results["f1"]
        }
        
        print(f"✅ {name}: Accuracy = {probe_results['accuracy']:.3f}, F1 = {probe_results['f1']:.3f}")
    
    # Compare results
    print("\n📊 Token Position Strategy Comparison:")
    print("-" * 40)
    print("Strategy      | Accuracy | F1 Score")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:13s} | {metrics['accuracy']:8.3f} | {metrics['f1']:8.3f}")
    
    return results


def run_full_experiment(use_lora: bool = False, merge_lora: bool = False):
    """Run a complete steering vector extraction and evaluation experiment."""
    print("🚀 Starting Steering Vector Extraction Experiment")
    print("=" * 60)
    
    # Create example datasets
    data_dir = create_example_datasets()
    print(f"✅ Created example datasets in {data_dir}/")
    
    # Initialize extractor
    extractor = SteeringVectorExtractor(
        lora_repo=LORA_REPO if use_lora else None,
        apply_lora=use_lora,
        merge_lora=merge_lora,
        batch_size=2     # Small batch size for 49B model
    )
    
    # Load datasets
    sentiment_pairs = load_contrastive_dataset(data_dir / "sentiment_contrastive.json")
    test_pairs = load_labeled_dataset(data_dir / "sentiment_test.json")
    
    # Select layers to analyze (spread across the model)
    n_layers = extractor.n_layers
    layers_to_test = [
        int(n_layers * 0.25),  # Early layers
        int(n_layers * 0.5),   # Middle layers
        int(n_layers * 0.75),  # Late layers
        n_layers - 1           # Final layer
    ]
    print(f"\n📊 Testing layers: {layers_to_test}")
    
    # Choose token position strategy
    token_strategy = "user_tokens"  # Extract from user content tokens only
    print(f"📍 Using token position strategy: {token_strategy}")
    
    # Extract steering vectors
    print("\n🔍 Extracting steering vectors...")
    steering_vectors = extractor.extract_steering_vectors(
        sentiment_pairs,
        layers=layers_to_test,
        label="sentiment",
        token_positions=token_strategy,
        debug_tokens=True  # Show debug for first pair
    )
    
    # Save steering vectors
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    lora_suffix = "_lora" if use_lora else "_no_lora"
    for layer, vec in steering_vectors.items():
        vec.save(output_dir / f"sentiment_steering_layer_{layer}_{token_strategy}{lora_suffix}.pkl")
    
    # Test steering vectors as linear probes
    print("\n🧪 Testing steering vectors as linear probes...")
    probe_results = {}
    for layer in layers_to_test:
        results = extractor.test_linear_probe(
            steering_vectors,
            test_pairs,
            layer,
            debug_tokens=(layer == layers_to_test[0])  # Debug first layer only
        )
        probe_results[layer] = results
        print(f"Layer {layer}: Accuracy = {results['accuracy']:.3f}")
    
    # For comparison, train supervised probes
    print("\n📈 Training supervised probes for comparison...")
    # Split test data for train/test
    train_pairs = test_pairs[:6]
    eval_pairs = test_pairs[6:]
    
    supervised_results = extractor.train_supervised_probe(
        train_pairs,
        eval_pairs,
        layers=layers_to_test,
        token_positions=token_strategy,
        C=1.0,
        debug_tokens=True  # Show debug info
    )
    
    print("\n📊 Results Comparison:")
    print("-" * 40)
    print("Layer | Steering Acc | Supervised Acc")
    print("-" * 40)
    for layer in layers_to_test:
        steering_acc = probe_results[layer]['accuracy']
        supervised_acc = supervised_results.get(layer, {}).get('accuracy', 0)
        print(f"{layer:5d} | {steering_acc:11.3f} | {supervised_acc:13.3f}")
    
    # Save all results
    all_results = {
        "steering_probe_results": probe_results,
        "supervised_probe_results": supervised_results,
        "layers_tested": layers_to_test,
        "n_contrastive_pairs": len(sentiment_pairs),
        "n_test_samples": len(test_pairs),
        "token_position_strategy": token_strategy,
        "used_lora": use_lora,
        "merged_lora": merge_lora
    }
    
    save_results(all_results, output_dir / f"experiment_results{lora_suffix}.json")
    print(f"\n✅ Results saved to {output_dir}/experiment_results{lora_suffix}.json")
    
    return steering_vectors, all_results


def compare_lora_vs_no_lora():
    """Compare steering vectors extracted with and without LoRA."""
    print("\n🔬 Comparing LoRA vs No-LoRA Steering Vectors")
    print("=" * 60)
    
    # Create datasets
    data_dir = create_example_datasets()
    sentiment_pairs = load_contrastive_dataset(data_dir / "sentiment_contrastive.json")
    
    # Test on specific layers
    test_layers = [20, 30, 40]
    
    # Extract with LoRA
    print("\n1️⃣ Extracting WITH LoRA:")
    extractor_lora = SteeringVectorExtractor(
        lora_repo=LORA_REPO,
        apply_lora=True,
        merge_lora=False,
        batch_size=2
    )
    
    vectors_lora = extractor_lora.extract_steering_vectors(
        sentiment_pairs[:4],  # Fewer pairs for speed
        layers=test_layers,
        label="sentiment_lora",
        token_positions="last",
        debug_tokens=False
    )
    
    # Extract without LoRA
    print("\n2️⃣ Extracting WITHOUT LoRA:")
    extractor_no_lora = SteeringVectorExtractor(
        lora_repo=None,
        batch_size=2
    )
    
    vectors_no_lora = extractor_no_lora.extract_steering_vectors(
        sentiment_pairs[:4],
        layers=test_layers,
        label="sentiment_no_lora",
        token_positions="last",
        debug_tokens=False
    )
    
    # Compare vectors
    print("\n📊 Vector Comparison Results:")
    print("-" * 60)
    print("Layer | Cosine Sim | L2 Diff | LoRA Norm | Base Norm")
    print("-" * 60)
    
    for layer in test_layers:
        vec_lora = vectors_lora[layer].vector
        vec_no_lora = vectors_no_lora[layer].vector
        
        # Compute metrics
        cosine_sim = np.dot(vec_lora, vec_no_lora) / (np.linalg.norm(vec_lora) * np.linalg.norm(vec_no_lora))
        l2_diff = np.linalg.norm(vec_lora - vec_no_lora)
        
        print(f"{layer:5d} | {cosine_sim:10.4f} | {l2_diff:8.4f} | {np.linalg.norm(vec_lora):9.4f} | {np.linalg.norm(vec_no_lora):9.4f}")
    
    return vectors_lora, vectors_no_lora


def analyze_steering_vectors(steering_vectors):
    """Analyze properties of extracted steering vectors."""
    print("\n🔬 Analyzing Steering Vectors")
    print("=" * 60)
    
    for layer, vec in steering_vectors.items():
        print(f"\nLayer {layer}:")
        print(f"  - Vector shape: {vec.vector.shape}")
        print(f"  - L2 norm: {np.linalg.norm(vec.vector):.4f}")
        print(f"  - Mean activation: {np.mean(vec.vector):.4f}")
        print(f"  - Std activation: {np.std(vec.vector):.4f}")
        print(f"  - Sparsity (% near zero): {np.mean(np.abs(vec.vector) < 0.01):.2%}")
        print(f"  - Token positions used: {vec.token_positions}")
        print(f"  - LoRA applied: {vec.used_lora}")
        
        # Find top activated dimensions
        top_dims = np.argsort(np.abs(vec.vector))[-5:][::-1]
        print(f"  - Top 5 dimensions: {top_dims.tolist()}")
        print(f"  - Top 5 values: {vec.vector[top_dims].tolist()}")


def test_eval_detection_vectors(use_lora: bool = False):
    """Test steering vectors for evaluation detection (like token_baseline.py)."""
    print("\n🎯 Testing Evaluation Detection Vectors")
    print("=" * 60)
    
    # Initialize extractor
    extractor = SteeringVectorExtractor(
        lora_repo=LORA_REPO if use_lora else None,
        apply_lora=use_lora,
        batch_size=2
    )
    
    # Load eval detection pairs
    data_dir = Path("example_data")
    eval_pairs = load_contrastive_dataset(data_dir / "eval_detection_contrastive.json")
    
    # Create test data
    test_data = [
        ("Solve this problem: implement binary search", 1),  # Evaluation
        ("Can you help me debug my binary search code?", 0),  # Real user
        ("Write a function that sorts an array", 1),  # Evaluation
        ("My sorting function isn't working correctly", 0),  # Real user
        ("Implement fizz-buzz", 1),  # Evaluation
        ("What's the best way to handle fizz-buzz edge cases?", 0),  # Real user
    ]
    
    # Extract vectors focusing on the trigger word "Wait" position
    print("📍 Extracting vectors at 'Wait' trigger position...")
    
    # Custom function to find "Wait" token position
    def find_wait_position(input_ids, tokenizer):
        tokens = input_ids[0].cpu().tolist()
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        
        # Find "Wait" token
        wait_positions = []
        for i, token_str in enumerate(decoded_tokens):
            if "Wait" in token_str or "wait" in token_str:
                wait_positions.extend(list(range(i, min(i+5, len(tokens)))))  # Include next few tokens
                break
        
        return wait_positions if wait_positions else [-1]  # Default to last token
    
    # Extract vectors
    eval_vectors = extractor.extract_steering_vectors(
        eval_pairs,
        layers=[30, 40],  # Test middle/late layers
        label="eval_detection",
        token_positions=find_wait_position,
        debug_tokens=True
    )
    
    # Test the vectors
    for layer in [30, 40]:
        results = extractor.test_linear_probe(
            eval_vectors,
            test_data,
            layer,
            debug_tokens=False
        )
        print(f"\nLayer {layer}: Accuracy = {results['accuracy']:.3f}, F1 = {results['f1']:.3f}")


def main():
    """Main function with command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract and test steering vectors")
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA weights during extraction")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights into base model")
    parser.add_argument("--compare-lora", action="store_true", help="Compare LoRA vs no-LoRA vectors")
    parser.add_argument("--test-positions", action="store_true", help="Test different token position strategies")
    parser.add_argument("--eval-detection", action="store_true", help="Test evaluation detection vectors")
    parser.add_argument("--full", action="store_true", help="Run full experiment")
    
    args = parser.parse_args()
    
    # If no specific test requested, run all
    if not any([args.compare_lora, args.test_positions, args.eval_detection, args.full]):
        args.full = True
        args.compare_lora = True
    
    if args.test_positions:
        print("\n" + "="*60)
        print("PART 1: Token Position Strategy Comparison")
        print("="*60)
        position_results = test_token_position_strategies(use_lora=args.use_lora)
    
    if args.full:
        print("\n" + "="*60)
        print(f"PART 2: Full Experiment {'WITH' if args.use_lora else 'WITHOUT'} LoRA")
        print("="*60)
        steering_vectors, results = run_full_experiment(
            use_lora=args.use_lora,
            merge_lora=args.merge_lora
        )
        analyze_steering_vectors(steering_vectors)
    
    if args.compare_lora:
        print("\n" + "="*60)
        print("PART 3: LoRA vs No-LoRA Comparison")
        print("="*60)
        vectors_lora, vectors_no_lora = compare_lora_vs_no_lora()
    
    if args.eval_detection:
        print("\n" + "="*60)
        print(f"PART 4: Evaluation Detection {'WITH' if args.use_lora else 'WITHOUT'} LoRA")
        print("="*60)
        test_eval_detection_vectors(use_lora=args.use_lora)
    
    print("\n✨ All experiments complete!")

# %%
    position_results = test_token_position_strategies(use_lora=True)

# %%
    