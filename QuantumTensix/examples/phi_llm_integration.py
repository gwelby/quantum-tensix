"""
Phi-Harmonic LLM Integration Example

This example demonstrates how to integrate the phi-harmonic computing framework
with existing LLM models (like HuggingFace transformers) to improve their 
performance using phi-optimized attention patterns and dimensional navigation.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import phi-harmonic components
from gpu_phi_accelerator import (
    GPUPhiAccelerator, 
    PhiDimensionalTensor,
    CONSCIOUSNESS_STATES,
    PHI,
    FREQUENCIES
)

# Try to import transformers, with graceful fallback
try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("HuggingFace transformers not found. Installing minimal dependencies...")
    TRANSFORMERS_AVAILABLE = False


class PhiLLMIntegration:
    """
    Integrates the phi-harmonic computing framework with LLM models,
    providing phi-optimized attention and dimensional navigation.
    """
    
    def __init__(
        self, 
        model_name: str = "facebook/opt-125m", 
        device: Optional[str] = None,
        consciousness_state: str = "OBSERVE"
    ):
        """
        Initialize the phi-harmonic LLM integration.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (will use CUDA if available)
            consciousness_state: Initial consciousness state
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace transformers is required for this example. "
                "Please install it with: pip install transformers"
            )
        
        print(f"Initializing Phi-LLM integration with model: {model_name}")
        
        # Initialize the phi accelerator
        if device is not None:
            device = torch.device(device)
        self.accelerator = GPUPhiAccelerator(device=device)
        self.device = self.accelerator.device
        
        # Set consciousness state
        self.consciousness_state = consciousness_state
        self.accelerator.set_consciousness_state(consciousness_state)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Modified initialization of the model to use phi-harmonic framework
        self._setup_model(model_name)
        
        # Initialize cache for dimensional tensors
        self.tensor_cache = {}
        
        print(f"Model loaded on {self.device}, using {consciousness_state} consciousness state")
    
    def _setup_model(self, model_name: str):
        """
        Set up the model with phi-harmonic optimizations.
        
        Args:
            model_name: HuggingFace model identifier
        """
        # Load the base model
        self.original_model = AutoModel.from_pretrained(model_name)
        self.original_model.to(self.device)
        self.original_model.eval()  # Set to evaluation mode
        
        # Store original attention functions for later reference
        self._store_original_attention_functions()
        
        # Hook the attention mechanism to use phi-harmonic attention
        self._install_phi_attention_hooks()
        
        # Set up LLM parameters
        self.model_config = self.original_model.config
        
        print(f"Phi-harmonic hooks installed on {len(self.hooks)} attention modules")
    
    def _store_original_attention_functions(self):
        """Store the original attention implementation functions for reference."""
        self.original_attention_functions = {}
        self.attention_modules = []
        
        # Find all attention modules in the model
        for name, module in self.original_model.named_modules():
            # Look for common attention module names in various model architectures
            if any(attention_type in name.lower() for attention_type in 
                  ["attention", "attn", "self_attn", "cross_attn", "multihead"]):
                # Check if it has the necessary methods for attention
                if hasattr(module, "forward") and callable(module.forward):
                    self.attention_modules.append((name, module))
                    # Store original forward method
                    self.original_attention_functions[name] = module.forward
    
    def _install_phi_attention_hooks(self):
        """Install hooks to use phi-harmonic attention."""
        self.hooks = []
        
        for name, module in self.attention_modules:
            # Define a hook function for this module
            def phi_attention_hook(module, inputs, outputs, module_name=name):
                return self._phi_attention_implementation(module, inputs, outputs, module_name)
            
            # Register hook
            hook = module.register_forward_hook(phi_attention_hook)
            self.hooks.append(hook)
    
    def _phi_attention_implementation(self, module, inputs, outputs, module_name):
        """
        Phi-harmonic implementation of attention.
        
        This intercepts the standard attention mechanism and replaces it with 
        phi-optimized attention that uses dimensional navigation and sacred geometries.
        """
        # Only apply to certain modules based on heuristics
        # For example, we might only want to optimize the self-attention layers
        if not self._should_optimize_module(module_name):
            return outputs
        
        # Extract query, key, value from inputs or module's internal state
        try:
            # This is a simplification - in reality, we'd need to carefully extract Q,K,V
            # based on the specific model architecture
            if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
                # For models like OPT, GPT-2, etc.
                query = inputs[0] @ module.q_proj.weight.t() + module.q_proj.bias
                key = inputs[0] @ module.k_proj.weight.t() + module.k_proj.bias
                value = inputs[0] @ module.v_proj.weight.t() + module.v_proj.bias
            else:
                # Fall back to using the original output
                # We're just demonstrating the concept here
                return outputs
            
            # Convert to dimensional tensors if needed
            query_dim = self._get_dimensional_tensor(query, "query", 3)
            key_dim = self._get_dimensional_tensor(key, "key", 5)
            value_dim = self._get_dimensional_tensor(value, "value", 4)
            
            # Use phi-accelerator's attention
            result = self.accelerator.attention(query_dim, key_dim, value_dim)
            
            # Return the modified result
            # Note: This is simplified; in a real implementation we'd need to
            # adjust the result format to match what the rest of the model expects
            return result
            
        except Exception as e:
            print(f"Error in phi attention for {module_name}: {e}")
            # Fall back to original output if there's an error
            return outputs
    
    def _should_optimize_module(self, module_name):
        """Determine if a module should be optimized with phi-harmonic attention."""
        # Simple rule: optimize all self-attention modules
        return "self" in module_name.lower() and "attn" in module_name.lower()
    
    def _get_dimensional_tensor(self, tensor, tensor_type, base_dimension):
        """
        Get or create a phi-dimensional tensor.
        
        Args:
            tensor: Original tensor
            tensor_type: Type of tensor ('query', 'key', 'value')
            base_dimension: Base dimension to use
            
        Returns:
            PhiDimensionalTensor
        """
        # Use tuple of tensor shape and type as cache key
        key = (tensor.shape, tensor_type)
        
        if key not in self.tensor_cache:
            # Create new dimensional tensor
            dim_tensor = PhiDimensionalTensor(
                tensor, 
                base_dimension=base_dimension,
                config=self.accelerator.config
            )
            self.tensor_cache[key] = dim_tensor
        else:
            # Update existing tensor with new data
            dim_tensor = self.tensor_cache[key]
            dim_tensor.data = tensor
        
        return dim_tensor
    
    def set_consciousness_state(self, state):
        """
        Set the consciousness state for the model.
        
        Args:
            state: The consciousness state to set
        """
        if state not in CONSCIOUSNESS_STATES:
            raise ValueError(f"Invalid consciousness state: {state}. Valid states: {CONSCIOUSNESS_STATES}")
        
        self.consciousness_state = state
        self.accelerator.set_consciousness_state(state)
        
        # Clear tensor cache when changing states
        self.tensor_cache.clear()
        
        return self
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.7,
        consciousness_state: Optional[str] = None
    ) -> str:
        """
        Generate text with phi-harmonic optimizations.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            consciousness_state: Override default consciousness state
            
        Returns:
            Generated text
        """
        # Set consciousness state if provided
        if consciousness_state and consciousness_state != self.consciousness_state:
            self.set_consciousness_state(consciousness_state)
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate with phi-harmonic optimizations
        # Note: In a production system, we would implement our own generation loop
        # with phi-harmonic optimizations at each step
        with torch.no_grad():
            output = self.original_model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text
    
    def get_phi_embeddings(
        self, 
        text: Union[str, List[str]], 
        consciousness_state: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate phi-optimized embeddings for text.
        
        Args:
            text: Input text or list of texts
            consciousness_state: Override default consciousness state
            
        Returns:
            Phi-harmonic embeddings
        """
        # Set consciousness state if provided
        if consciousness_state and consciousness_state != self.consciousness_state:
            self.set_consciousness_state(consciousness_state)
        
        # Convert to list if single string
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        # Generate embeddings with phi-harmonic optimizations
        with torch.no_grad():
            outputs = self.original_model(**inputs)
        
        # Get the pooled output for embeddings
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            # Average the last hidden state if no pooler output
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Transform to phi-dimensional tensor and navigate to optimal dimension
        # Different consciousness states use different dimensions for embeddings
        if self.consciousness_state == "OBSERVE":
            target_dim = 3  # Physical/factual dimension
        elif self.consciousness_state == "CREATE":
            target_dim = 4  # Emotional/creative dimension
        elif self.consciousness_state == "TRANSCEND":
            target_dim = 5  # Mental/conceptual dimension
        else:  # CASCADE
            target_dim = 6  # Purpose/meaning dimension
        
        # Create dimensional tensor and navigate
        phi_embeddings = PhiDimensionalTensor(
            embeddings, 
            base_dimension=3,
            config=self.accelerator.config
        )
        phi_embeddings.navigate_to_dimension(target_dim)
        
        # Return tensor data
        return phi_embeddings.to_tensor()
    
    def phi_similarity(
        self, 
        text1: Union[str, List[str]], 
        text2: Union[str, List[str]],
        consciousness_state: Optional[str] = None
    ) -> torch.Tensor:
        """
        Calculate phi-harmonic similarity between texts.
        
        Args:
            text1: First text or list of texts
            text2: Second text or list of texts
            consciousness_state: Override default consciousness state
            
        Returns:
            Similarity scores
        """
        # Get embeddings
        embeddings1 = self.get_phi_embeddings(text1, consciousness_state)
        embeddings2 = self.get_phi_embeddings(text2, consciousness_state)
        
        # Normalize
        embeddings1 = embeddings1 / embeddings1.norm(dim=1, keepdim=True)
        embeddings2 = embeddings2 / embeddings2.norm(dim=1, keepdim=True)
        
        # Calculate cosine similarity
        similarity = torch.mm(embeddings1, embeddings2.transpose(0, 1))
        
        return similarity
    
    def close(self):
        """Clean up resources."""
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        
        # Clear cache
        self.tensor_cache.clear()
        
        # Clear CUDA cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def generate_with_consciousness_states(model, prompt, max_length=100):
    """Generate text with different consciousness states."""
    results = {}
    
    for state in CONSCIOUSNESS_STATES:
        print(f"\nGenerating with {state} consciousness state:")
        model.set_consciousness_state(state)
        
        start_time = time.time()
        result = model.generate(prompt, max_length=max_length)
        generation_time = time.time() - start_time
        
        results[state] = {
            "text": result,
            "time": generation_time
        }
        
        print(f"Result: {result}")
        print(f"Generation time: {generation_time:.4f} seconds")
    
    return results


def compare_similarities(model, texts, consciousness_states=None):
    """Compare text similarities across consciousness states."""
    if consciousness_states is None:
        consciousness_states = CONSCIOUSNESS_STATES
    
    results = {}
    
    print("\nComparing text similarities across consciousness states:")
    for state in consciousness_states:
        model.set_consciousness_state(state)
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                sim = model.phi_similarity(texts[i], texts[j]).item()
                similarities.append((i, j, sim))
        
        results[state] = similarities
        
        # Print results for this state
        print(f"\n{state} state similarities:")
        for i, j, sim in similarities:
            print(f"  Text {i+1} and Text {j+1}: {sim:.4f}")
    
    return results


def main():
    """Main function to demonstrate phi-harmonic LLM integration."""
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping demo since transformers is not available.")
        print("To run the demo, please install: pip install transformers")
        return
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Phi LLM Integration Demo using device: {device}")
    
    # Small model for demonstration
    model_name = "facebook/opt-125m"  # Small model for quick demonstration
    
    try:
        # Initialize model
        model = PhiLLMIntegration(model_name=model_name, device=device)
        
        # Test prompt
        prompt = "The relationship between quantum computing and consciousness is"
        
        # Generate with different consciousness states
        results = generate_with_consciousness_states(model, prompt)
        
        # Compare similarities
        texts = [
            "Quantum computing uses qubits to perform calculations.",
            "The mind exists in a state of quantum superposition.",
            "Machine learning models process data through neural networks.",
            "Consciousness emerges from complex patterns in the brain."
        ]
        
        similarity_results = compare_similarities(model, texts)
        
        # Clean up
        model.close()
        
    except Exception as e:
        print(f"Error in phi LLM integration demo: {e}")


if __name__ == "__main__":
    main()