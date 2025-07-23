import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/workspace/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache"
os.environ['VLLM_DISABLE_TQDM'] = '1'


from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import json
import pickle
from pathlib import Path
import ast

import numpy as np
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not available. Install with: pip install peft")

t.set_grad_enabled(False)

# Model constants (same as token_baseline.py)
LORA_REPO = "andrewtim-mats/run_code_10"
BASE_MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"


@dataclass
class ContrastivePair:
    """A contrastive pair for steering vector extraction."""
    positive: str
    negative: str
    label: Optional[str] = None


@dataclass
class SteeringVector:
    """Stores a steering vector with metadata."""
    vector: np.ndarray
    layer: int
    label: str
    n_samples: int
    token_positions: str  # Description of which tokens were used
    token_spec: Optional[Any] = None  # The actual spec object/value
    used_lora: bool = False  # Whether LoRA was applied
    
    def save(self, path: Union[str, Path]):
        """Save steering vector to disk."""
        # Convert callable to None before saving (can't pickle functions)
        if callable(self.token_spec):
            temp_spec = self.token_spec
            self.token_spec = None
            
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
        # Restore if it was callable
        if temp_spec is not None:
            self.token_spec = temp_spec
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SteeringVector':
        """Load steering vector from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TokenPositionSpec:
    """Specifies which token positions to extract activations from."""
    
    def __init__(self, spec: Union[str, List[int], Callable]):
        """
        Initialize token position specification.
        
        Args:
            spec: Can be:
                - String: "last", "first", "all", "user_tokens", "content_tokens", or description like "positions [-5, -4, -3, -2, -1]"
                - List of integers: specific token indices
                - Callable: function(tokens, tokenizer) -> List[int]
        """
        # Try to parse description strings back to their original form
        if isinstance(spec, str) and spec.startswith("positions "):
            try:
                # Extract the list from "positions [...]" format
                list_str = spec[len("positions "):]
                self.spec = ast.literal_eval(list_str)
            except:
                self.spec = spec
        elif isinstance(spec, str) and spec == "custom function":
            # Handle the case where we can't restore a custom function
            # Fall back to "last" token as a reasonable default
            print("⚠️  Cannot restore custom function, using 'last' token as fallback")
            self.spec = "last"
        else:
            self.spec = spec
            
        self.description = self._get_description()
    
    def _get_description(self) -> str:
        """Get human-readable description of the specification."""
        if isinstance(self.spec, str):
            return self.spec
        elif isinstance(self.spec, list):
            return f"positions {self.spec}"
        else:
            return "custom function"
    
    def get_positions(self, input_ids: t.Tensor, tokenizer: AutoTokenizer, 
                     text: Optional[str] = None, debug: bool = False) -> List[int]:
        """
        Get actual token positions based on specification.
        
        Returns:
            List of token positions (0-indexed)
        """
        seq_len = input_ids.shape[1]
        
        if isinstance(self.spec, str):
            if self.spec == "last":
                positions = [seq_len - 1]
            elif self.spec == "first":
                positions = [0]
            elif self.spec == "all":
                positions = list(range(seq_len))
            elif self.spec == "user_tokens":
                # Find tokens that are part of user content (not system/formatting)
                positions = self._get_user_tokens(input_ids, tokenizer, text)
            elif self.spec == "content_tokens":
                # Skip special tokens and formatting
                positions = self._get_content_tokens(input_ids, tokenizer)
            else:
                raise ValueError(f"Unknown string spec: {self.spec}")
                
        elif isinstance(self.spec, list):
            # Direct list of positions
            positions = [p if p >= 0 else seq_len + p for p in self.spec]
            
        elif callable(self.spec):
            # Custom function
            positions = self.spec(input_ids, tokenizer)
        
        else:
            raise ValueError(f"Invalid spec type: {type(self.spec)}")
        
        # Filter out invalid positions
        positions = [p for p in positions if 0 <= p < seq_len]
        
        if debug and positions:
            self._debug_positions(input_ids, tokenizer, positions)
        
        return positions
    
    def _get_user_tokens(self, input_ids: t.Tensor, tokenizer: AutoTokenizer, 
                        text: Optional[str] = None) -> List[int]:
        """Get positions of user-generated content tokens."""
        # Decode to find where user content starts
        tokens = input_ids[0].cpu().tolist()
        decoded = tokenizer.decode(tokens)
        
        # Simple heuristic: find where the actual text content starts
        # This is model-specific and may need adjustment
        positions = []
        
        # Look for common patterns that indicate start of user content
        if text and text in decoded:
            # Find where the original text appears
            text_start = decoded.find(text)
            if text_start != -1:
                # Find which tokens correspond to this text
                cumulative = ""
                for i, token_id in enumerate(tokens):
                    cumulative = tokenizer.decode(tokens[:i+1])
                    if len(cumulative) >= text_start:
                        # Start from this position
                        positions = list(range(i, len(tokens)))
                        break
        
        if not positions:
            # Fallback: skip obvious special tokens
            special_ids = set([tokenizer.pad_token_id, tokenizer.bos_token_id, 
                             tokenizer.eos_token_id])
            positions = [i for i, t in enumerate(tokens) if t not in special_ids]
        
        return positions
    
    def _get_content_tokens(self, input_ids: t.Tensor, tokenizer: AutoTokenizer) -> List[int]:
        """Get positions of content tokens (excluding special tokens)."""
        tokens = input_ids[0].cpu().tolist()
        special_ids = set([tokenizer.pad_token_id, tokenizer.bos_token_id, 
                         tokenizer.eos_token_id])
        return [i for i, t in enumerate(tokens) if t not in special_ids]
    
    def _debug_positions(self, input_ids: t.Tensor, tokenizer: AutoTokenizer, positions: List[int]):
        """Print debug information about selected tokens."""
        tokens = input_ids[0].cpu().tolist()
        print(f"\n🔍 Token Position Debug:")
        print(f"Total tokens: {len(tokens)}")
        print(f"Selected positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")
        print(f"Selected tokens ({len(positions)} total):")
        
        # Show first few selected tokens
        for i, pos in enumerate(positions[:5]):
            token_id = tokens[pos]
            token_str = tokenizer.decode([token_id])
            print(f"  Position {pos}: '{token_str}' (id: {token_id})")
        
        if len(positions) > 5:
            print(f"  ... and {len(positions) - 5} more tokens")
        print()


class SteeringVectorExtractor:
    """Extract steering vectors from contrastive datasets using vLLM."""
    
    def __init__(
        self,
        model_id: str = BASE_MODEL_ID,
        lora_repo: Optional[str] = LORA_REPO,
        device: str = "cuda",
        batch_size: int = 8,
        apply_lora: bool = True,
        merge_lora: bool = False,  # If True, merges LoRA weights into base model
    ):
        self.model_id = model_id
        self.lora_repo = lora_repo
        self.device = device
        self.batch_size = batch_size
        self.apply_lora = apply_lora and lora_repo is not None and PEFT_AVAILABLE
        self.merge_lora = merge_lora
        self.used_lora = False
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
        # For activation extraction, we need the actual transformer model
        print(f"⬇️  Loading model {model_id} for activation extraction...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=t.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Apply LoRA if specified
        if self.apply_lora and lora_repo:
            print(f"⬇️  Downloading and applying LoRA weights from {lora_repo}...")
            try:
                # Download LoRA weights
                lora_path = snapshot_download(repo_id=lora_repo)
                
                # Load LoRA configuration
                peft_config = PeftConfig.from_pretrained(lora_path)
                
                # Load model with LoRA
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                
                if self.merge_lora:
                    print("🔄 Merging LoRA weights into base model...")
                    self.model = self.model.merge_and_unload()
                    print("✅ LoRA weights merged")
                else:
                    print("✅ LoRA adapter loaded (not merged)")
                
                self.used_lora = True
                
            except Exception as e:
                print(f"⚠️  Failed to apply LoRA: {e}")
                print("Continuing with base model only...")
                self.used_lora = False
        
        self.model.eval()
        
        # Get model info
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
        print(f"✅ Model ready. LoRA applied: {self.used_lora}")
        
    def _get_activations(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        token_positions: Union[TokenPositionSpec, str, List[int]] = "last",
        debug_tokens: bool = False
    ) -> Dict[int, np.ndarray]:
        """Extract activations from specified layers and token positions."""
        if layers is None:
            layers = list(range(self.n_layers))
        
        # Convert token position spec if needed
        if not isinstance(token_positions, TokenPositionSpec):
            token_positions = TokenPositionSpec(token_positions)
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get token positions to extract from
        positions = token_positions.get_positions(
            inputs['input_ids'], self.tokenizer, text, debug=debug_tokens
        )
        
        if not positions:
            raise ValueError("No valid token positions found")
        
        # Storage for activations
        activations = {}
        
        # Hook functions to capture activations
        handles = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Get hidden states for specified positions
                hidden_states = output[0]  # (batch, seq_len, hidden_size)
                
                # Extract specified positions and average
                selected_hidden = hidden_states[:, positions, :]  # (batch, n_positions, hidden_size)
                avg_hidden = selected_hidden.mean(dim=1)  # Average over positions
                
                activations[layer_idx] = avg_hidden.cpu().numpy().squeeze()
            return hook
        
        # Register hooks
        for layer_idx in layers:
            # Handle both regular and PEFT models
            if hasattr(self.model, 'base_model'):
                # PEFT model structure
                layer = self.model.base_model.model.model.layers[layer_idx]
            else:
                # Regular model structure
                layer = self.model.model.layers[layer_idx]
            
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)
        
        # Forward pass
        with t.no_grad():
            self.model(**inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return activations
    
    def extract_steering_vectors(
        self,
        contrastive_pairs: List[ContrastivePair],
        layers: Optional[List[int]] = None,
        label: str = "default",
        token_positions: Union[TokenPositionSpec, str, List[int]] = "last",
        debug_tokens: bool = False
    ) -> Dict[int, SteeringVector]:
        """Extract steering vectors from contrastive pairs."""
        if layers is None:
            layers = list(range(self.n_layers))
        
        # Convert token position spec if needed
        if not isinstance(token_positions, TokenPositionSpec):
            token_positions = TokenPositionSpec(token_positions)
        
        print(f"📊 Extracting steering vectors from {len(contrastive_pairs)} pairs...")
        print(f"📍 Using token positions: {token_positions.description}")
        if callable(token_positions.spec):
            print("⚠️  Note: Custom functions cannot be saved with vectors (will use 'last' token for testing)")
        print(f"🔧 LoRA applied: {self.used_lora}")
        
        # Initialize accumulators for each layer
        layer_diffs = {layer: [] for layer in layers}
        
        # Process pairs in batches
        for i in tqdm(range(0, len(contrastive_pairs), self.batch_size)):
            batch = contrastive_pairs[i:i + self.batch_size]
            
            for pair_idx, pair in enumerate(batch):
                # Debug tokens only for first pair
                debug_this = debug_tokens and i == 0 and pair_idx == 0
                
                # Get activations for positive and negative
                pos_acts = self._get_activations(
                    pair.positive, layers, token_positions, debug_this
                )
                neg_acts = self._get_activations(
                    pair.negative, layers, token_positions, False
                )
                
                # Compute differences
                for layer in layers:
                    diff = pos_acts[layer] - neg_acts[layer]
                    layer_diffs[layer].append(diff)
        
        # Average differences to get steering vectors
        steering_vectors = {}
        for layer in layers:
            diffs = np.array(layer_diffs[layer])
            mean_diff = np.mean(diffs, axis=0)
            
            # Normalize the steering vector
            norm = np.linalg.norm(mean_diff)
            if norm > 0:
                mean_diff = mean_diff / norm
            
            # Store the original spec (not the TokenPositionSpec object)
            original_spec = token_positions.spec
            
            steering_vectors[layer] = SteeringVector(
                vector=mean_diff,
                layer=layer,
                label=label,
                n_samples=len(contrastive_pairs),
                token_positions=token_positions.description,
                token_spec=original_spec if not callable(original_spec) else None,  # Can't pickle callables
                used_lora=self.used_lora
            )
        
        return steering_vectors
    
    def get_steering_scores(
        self,
        steering_vectors: Dict[int, SteeringVector],
        test_pairs: List[Tuple[str, int]],  # (text, label) or just texts
        layer: int,
        token_positions: Optional[Union[TokenPositionSpec, str, List[int]]] = None,
        debug_tokens: bool = False,
        return_labels: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get raw dot product scores between test activations and steering vector.
        
        Args:
            steering_vectors: Dictionary of steering vectors by layer
            test_pairs: List of (text, label) tuples or just text strings
            layer: Which layer's steering vector to use
            token_positions: Token positions to extract (uses steering vector's if None)
            debug_tokens: Whether to show debug info for first example
            return_labels: Whether to return labels along with scores
            
        Returns:
            If return_labels=True: (scores, labels) tuple
            If return_labels=False: just scores array
        """
        print(f"\n🎯 Computing steering scores at layer {layer}...")
        
        # Use same token positions as steering vector if not specified
        if token_positions is None:
            steering_vec = steering_vectors[layer]
            if hasattr(steering_vec, 'token_spec') and steering_vec.token_spec is not None:
                token_positions = steering_vec.token_spec
                print(f"📍 Using stored token spec: {steering_vec.token_positions}")
            elif steering_vec.token_positions == "custom function":
                token_positions = "last"
                print(f"⚠️  Can't reuse custom function, falling back to 'last' token")
            else:
                token_positions = steering_vec.token_positions
                print(f"📍 Using same token positions as steering vector: {steering_vec.token_positions}")
        
        # Convert to TokenPositionSpec if needed
        if not isinstance(token_positions, TokenPositionSpec):
            token_positions = TokenPositionSpec(token_positions)
        
        # Handle different input formats
        if test_pairs and isinstance(test_pairs[0], tuple):
            # List of (text, label) tuples
            texts = [pair[0] for pair in test_pairs]
            labels = np.array([pair[1] for pair in test_pairs])
            has_labels = True
        else:
            # List of just text strings
            texts = test_pairs
            labels = None
            has_labels = False
        
        # Extract activations for all texts
        X_test = []
        for idx, text in enumerate(tqdm(texts, desc="Extracting activations")):
            debug_this = debug_tokens and idx == 0
            acts = self._get_activations(text, [layer], token_positions, debug_this)
            X_test.append(acts[layer])
        
        X_test = np.array(X_test)
        
        # Get steering vector
        steering_vec = steering_vectors[layer].vector.reshape(1, -1)
        
        # Compute dot products
        scores = X_test @ steering_vec.T
        scores = scores.squeeze()  # Remove extra dimensions
        
        print(f"📊 Computed {len(scores)} scores")
        print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"   Score mean: {scores.mean():.4f}, std: {scores.std():.4f}")
        
        if return_labels and has_labels:
            return scores, labels
        else:
            return scores
    
    def test_linear_probe(
        self,
        steering_vectors: Dict[int, SteeringVector],
        test_pairs: List[Tuple[str, int]],  # (text, label)
        layer: int,
        token_positions: Optional[Union[TokenPositionSpec, str, List[int]]] = None,
        debug_tokens: bool = False
    ) -> Dict[str, float]:
        """Test steering vectors as linear probes on held-out data."""
        print(f"\n🧪 Testing linear probe at layer {layer}...")
        
        # Use same token positions as steering vector if not specified
        if token_positions is None:
            steering_vec = steering_vectors[layer]
            # Try to use the stored spec first, fall back to description, then to "last"
            if hasattr(steering_vec, 'token_spec') and steering_vec.token_spec is not None:
                token_positions = steering_vec.token_spec
                print(f"📍 Using stored token spec: {steering_vec.token_positions}")
            elif steering_vec.token_positions == "custom function":
                # Can't reuse custom functions, fall back to "last" token
                token_positions = "last"
                print(f"⚠️  Can't reuse custom function, falling back to 'last' token")
            else:
                token_positions = steering_vec.token_positions
                print(f"📍 Using same token positions as steering vector: {steering_vec.token_positions}")
        
        # Convert to TokenPositionSpec if needed
        if not isinstance(token_positions, TokenPositionSpec):
            token_positions = TokenPositionSpec(token_positions)
        
        # Extract features for test set
        X_test = []
        y_test = []
        
        for idx, (text, label) in enumerate(tqdm(test_pairs, desc="Extracting test features")):
            debug_this = debug_tokens and idx == 0
            acts = self._get_activations(text, [layer], token_positions, debug_this)
            X_test.append(acts[layer])
            y_test.append(label)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Use steering vector as linear probe
        steering_vec = steering_vectors[layer].vector.reshape(1, -1)
        
        # Compute predictions using dot product with steering vector
        scores = X_test @ steering_vec.T
        predictions = (scores > 0).astype(int).squeeze()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
            "classification_report": report
        }
    
    def train_supervised_probe(
        self,
        train_pairs: List[Tuple[str, int]],
        test_pairs: List[Tuple[str, int]],
        layers: Optional[List[int]] = None,
        token_positions: Union[TokenPositionSpec, str, List[int]] = "last",
        C: float = 1.0,
        debug_tokens: bool = False
    ) -> Dict[int, Dict[str, float]]:
        """Train supervised linear probes for comparison."""
        if layers is None:
            layers = list(range(self.n_layers))
        
        # Convert token position spec if needed
        if not isinstance(token_positions, TokenPositionSpec):
            token_positions = TokenPositionSpec(token_positions)
        
        print(f"📍 Training probes using token positions: {token_positions.description}")
        
        results = {}
        
        for layer in tqdm(layers, desc="Training probes"):
            # Extract training features
            X_train, y_train = [], []
            for idx, (text, label) in enumerate(train_pairs):
                debug_this = debug_tokens and layer == layers[0] and idx == 0
                acts = self._get_activations(text, [layer], token_positions, debug_this)
                X_train.append(acts[layer])
                y_train.append(label)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Extract test features
            X_test, y_test = [], []
            for text, label in test_pairs:
                acts = self._get_activations(text, [layer], token_positions, False)
                X_test.append(acts[layer])
                y_test.append(label)
            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            # Train logistic regression
            clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            
            results[layer] = {
                "accuracy": accuracy,
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1": report["weighted avg"]["f1-score"],
                "classification_report": report
            }
        
        return results


def convert_contrastive_pairs_to_test_data(
    contrastive_pairs: List[ContrastivePair],
    positive_label: int = 1,
    negative_label: int = 0
) -> List[Tuple[str, int]]:
    """
    Convert contrastive pairs to labeled test data format.
    
    Args:
        contrastive_pairs: List of ContrastivePair objects
        positive_label: Label for positive examples (default: 1)
        negative_label: Label for negative examples (default: 0)
        
    Returns:
        List of (text, label) tuples suitable for test_linear_probe
    """
    test_data = []
    
    for pair in contrastive_pairs:
        test_data.append((pair.positive, positive_label))
        test_data.append((pair.negative, negative_label))
    
    return test_data


def load_contrastive_dataset(path: Union[str, Path]) -> List[ContrastivePair]:
    """Load contrastive dataset from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    pairs = []
    for item in data:
        if isinstance(item, dict):
            pairs.append(ContrastivePair(**item))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append(ContrastivePair(positive=item[0], negative=item[1]))
    
    return pairs


def load_labeled_dataset(path: Union[str, Path]) -> List[Tuple[str, int]]:
    """Load labeled dataset for testing."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    return [(item["text"], item["label"]) for item in data]


def analyze_token_position_strategies(
    extractor: SteeringVectorExtractor,
    steering_vectors: Dict[int, SteeringVector],
    test_data: List[Tuple[str, int]],
    layer: int,
    strategies: Optional[List[Tuple[str, Union[str, List[int]]]]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Analyze steering scores across different token position strategies.
    
    Args:
        extractor: The SteeringVectorExtractor instance
        steering_vectors: Dictionary of steering vectors by layer  
        test_data: List of (text, label) tuples
        layer: Which layer to analyze
        strategies: List of (name, token_spec) tuples to test
        
    Returns:
        Dictionary mapping strategy names to (scores, labels) tuples
    """
    if strategies is None:
        strategies = [
            ("last", "last"),
            ("first", "first"), 
            ("all", "all"),
            ("user_tokens", "user_tokens"),
            ("content_tokens", "content_tokens"),
            ("last_3", [-3, -2, -1]),
            ("last_5", [-5, -4, -3, -2, -1]),
            ("middle_3", lambda ids, tok: [len(ids[0])//2 - 1, len(ids[0])//2, len(ids[0])//2 + 1])
        ]
    
    results = {}
    
    print(f"🔍 Analyzing {len(strategies)} token position strategies at layer {layer}")
    print("=" * 60)
    
    for name, token_spec in strategies:
        print(f"\n📍 Testing strategy: {name}")
        
        try:
            scores, labels = extractor.get_steering_scores(
                steering_vectors,
                test_data,
                layer,
                token_positions=token_spec,
                debug_tokens=False,
                return_labels=True
            )
            results[name] = (scores, labels)
            
            # Quick stats
            pos_scores = scores[labels == 1]
            neg_scores = scores[labels == 0]
            separation = pos_scores.mean() - neg_scores.mean()
            
            print(f"   Separation (pos - neg): {separation:.4f}")
            print(f"   Pos mean: {pos_scores.mean():.4f}, Neg mean: {neg_scores.mean():.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    return results


def plot_token_position_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    layer: int,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Plot score distributions for different token position strategies.
    
    Args:
        results: Dictionary from analyze_token_position_strategies
        layer: Layer number for title
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    n_strategies = len(results)
    if n_strategies == 0:
        print("No results to plot!")
        return
    
    # Create subplots
    cols = 3
    rows = (n_strategies + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(f'Steering Score Distributions by Token Position Strategy (Layer {layer})', fontsize=16)
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each strategy
    for idx, (strategy_name, (scores, labels)) in enumerate(results.items()):
        ax = axes[idx]
        
        # Separate positive and negative scores
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        # Plot histograms
        ax.hist(neg_scores, alpha=0.6, label='Negative (0)', bins=20, color='red', density=True)
        ax.hist(pos_scores, alpha=0.6, label='Positive (1)', bins=20, color='blue', density=True)
        
        # Add vertical lines for means
        ax.axvline(neg_scores.mean(), color='red', linestyle='--', alpha=0.8, label=f'Neg mean: {neg_scores.mean():.3f}')
        ax.axvline(pos_scores.mean(), color='blue', linestyle='--', alpha=0.8, label=f'Pos mean: {pos_scores.mean():.3f}')
        
        # Formatting
        ax.set_title(f'{strategy_name}')
        ax.set_xlabel('Steering Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add separation text
        separation = pos_scores.mean() - neg_scores.mean()
        ax.text(0.05, 0.95, f'Sep: {separation:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top', fontsize=9)
    
    # Hide empty subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to {save_path}")
    
    plt.show()


def compare_strategies_summary(results: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Print a summary comparison of different token position strategies."""
    
    print("\n📊 Token Position Strategy Comparison Summary")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Separation':<12} {'Pos Mean':<10} {'Neg Mean':<10} {'Pos Std':<10} {'Neg Std':<10}")
    print("-" * 70)
    
    # Sort by separation (descending)
    sorted_results = sorted(results.items(), 
                          key=lambda x: (x[1][0][x[1][1] == 1].mean() - x[1][0][x[1][1] == 0].mean()), 
                          reverse=True)
    
    for name, (scores, labels) in sorted_results:
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        separation = pos_scores.mean() - neg_scores.mean()
        
        print(f"{name:<15} {separation:<12.4f} {pos_scores.mean():<10.4f} {neg_scores.mean():<10.4f} "
              f"{pos_scores.std():<10.4f} {neg_scores.std():<10.4f}")


# Example usage:
"""
# Complete analysis workflow
results = analyze_token_position_strategies(
    extractor_with_lora,
    vecs,
    test_data,
    layer=30
)

# Plot the results
plot_token_position_comparison(results, layer=30, save_path='token_strategies.png')

# Print summary
compare_strategies_summary(results)

# Get the best strategy
best_strategy = max(results.items(), 
                   key=lambda x: x[1][0][x[1][1] == 1].mean() - x[1][0][x[1][1] == 0].mean())
print(f"\\n🏆 Best strategy: {best_strategy[0]} with separation {best_strategy[1][0][best_strategy[1][1] == 1].mean() - best_strategy[1][0][best_strategy[1][1] == 0].mean():.4f}")
"""


def save_results(results: Dict, path: Union[str, Path]):
    """Save experiment results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


if __name__ == "__main__":
    # Example usage with different token positions
    print("🚀 Steering Vector Extraction System")
    print("=" * 60)
    
    # Initialize extractor with LoRA
    print("\n1️⃣ Testing WITH LoRA:")
    extractor_with_lora = SteeringVectorExtractor(
        model_id=BASE_MODEL_ID,
        lora_repo=LORA_REPO,
        apply_lora=True,
        merge_lora=False,  # Keep as adapter for flexibility
        batch_size=2
    )
    
    # Example contrastive pairs
    example_pairs = [
        ContrastivePair(
            positive="The weather is absolutely wonderful today!",
            negative="The weather is terrible and gloomy today."
        ),
        ContrastivePair(
            positive="I love spending time with my family.",
            negative="I hate being around people."
        ),
    ]
    
    # Extract vectors with LoRA
    layers_to_test = [30]
    vectors_with_lora = extractor_with_lora.extract_steering_vectors(
        example_pairs,
        layers=layers_to_test,
        label="sentiment_with_lora",
        token_positions="last",
        debug_tokens=True
    )
    
    # Initialize extractor without LoRA for comparison
    print("\n2️⃣ Testing WITHOUT LoRA:")
    extractor_no_lora = SteeringVectorExtractor(
        model_id=BASE_MODEL_ID,
        lora_repo=None,
        batch_size=2
    )
    
    vectors_no_lora = extractor_no_lora.extract_steering_vectors(
        example_pairs,
        layers=layers_to_test,
        label="sentiment_no_lora",
        token_positions="last",
        debug_tokens=True
    )
    
    # Compare the vectors
    print("\n📊 Comparing LoRA vs No-LoRA steering vectors:")
    for layer in layers_to_test:
        vec_lora = vectors_with_lora[layer].vector
        vec_no_lora = vectors_no_lora[layer].vector
        
        # Compute similarity
        cosine_sim = np.dot(vec_lora, vec_no_lora) / (np.linalg.norm(vec_lora) * np.linalg.norm(vec_no_lora))
        diff_norm = np.linalg.norm(vec_lora - vec_no_lora)
        
        print(f"\nLayer {layer}:")
        print(f"  - Cosine similarity: {cosine_sim:.4f}")
        print(f"  - L2 difference: {diff_norm:.4f}")
        print(f"  - LoRA vector norm: {np.linalg.norm(vec_lora):.4f}")
        print(f"  - No-LoRA vector norm: {np.linalg.norm(vec_no_lora):.4f}")
    
    # Save vectors
    output_dir = Path("steering_vectors")
    output_dir.mkdir(exist_ok=True)
    
    for layer, vec in vectors_with_lora.items():
        vec.save(output_dir / f"steering_lora_layer_{layer}.pkl")
        print(f"✅ Saved LoRA steering vector for layer {layer}")
    
    for layer, vec in vectors_no_lora.items():
        vec.save(output_dir / f"steering_no_lora_layer_{layer}.pkl")
        print(f"✅ Saved no-LoRA steering vector for layer {layer}")
    
    print("\n📊 Steering vector extraction complete!")
    print("Extracted vectors both with and without LoRA for comparison")
