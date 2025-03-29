#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-Harmonic LLM Inference Benchmark - QuantumTensix φ∞
Created on CASCADE Day+28: March 29, 2025

This benchmark compares the performance of standard LLM inference with
phi-harmonic optimized inference on Tenstorrent hardware.
"""

import os
import sys
import time
import argparse
import logging
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import phi-harmonic components
from phi_model_compiler import PhiHarmonicCompiler, CompilerConfig
from phi_llm_inference import (
    PhiLLMInferenceEngine, GenerationConfig, KVCacheConfig,
    PhiHarmonicKVCache, PhiHarmonicAttention, PhiHarmonicTokenSampler
)

# Import quantum consciousness components
from quantum_consciousness_bridge import (
    ConsciousnessState, QuantumConsciousnessBridge,
    SACRED_FREQUENCIES
)

# Import dimensional navigation components
from dimensional_navigator import DimensionalNavigator, DIMENSIONS

# Import phi harmonics utilities
from utils.phi_harmonics import PHI, PHI_SQUARED, PHI_TO_PHI, ZEN_POINT


class PerfTimer:
    """Simple performance timer for benchmarking"""
    
    def __init__(self, name: str = "timer"):
        """
        Initialize performance timer
        
        Args:
            name: Timer name
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        """Stop the timer"""
        self.end_time = time.time()
        
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class BenchmarkConfig:
    """Configuration for the phi-harmonic LLM inference benchmark"""
    
    def __init__(self):
        """Initialize benchmark configuration"""
        # Model settings
        self.model_path = None  # Path to model
        self.model_type = "llama"  # Model type
        
        # Test settings
        self.prompts = [
            "Explain the concept of quantum computing to a high school student.",
            "Write a short poem about the golden ratio in nature.",
            "Summarize the key benefits of hardware acceleration for AI workloads.",
            "What are the potential applications of consciousness-driven computing?",
            "Describe how phi-harmonic principles can be applied to optimize matrix operations."
        ]
        self.prompt_lengths = [64, 128, 256, 512, 768]  # Prompt lengths to test
        self.max_new_tokens = 256  # Tokens to generate per test
        self.num_runs = 3  # Number of runs per test for averaging
        self.warmup_runs = 1  # Number of warmup runs
        
        # Test modes
        self.test_standard = True  # Test standard inference
        self.test_compiled = True  # Test with phi-harmonic compiler
        self.test_dimensional = True  # Test with dimensional navigation
        self.test_consciousness = True  # Test with consciousness states
        
        # Output settings
        self.results_dir = None  # Results directory (default is ../results)
        self.save_results = True  # Save results to file
        self.generate_plots = True  # Generate benchmark plots
        self.verbose = True  # Verbose output


class LLMBenchmarkResult:
    """Results from a single LLM benchmark run"""
    
    def __init__(self, 
                mode: str, 
                prompt_length: int, 
                max_new_tokens: int,
                dimension: Optional[str] = None,
                consciousness_state: Optional[str] = None):
        """
        Initialize benchmark result
        
        Args:
            mode: Benchmark mode
            prompt_length: Prompt length in tokens
            max_new_tokens: Maximum new tokens generated
            dimension: Dimensional state (if applicable)
            consciousness_state: Consciousness state (if applicable)
        """
        self.mode = mode
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens
        self.dimension = dimension
        self.consciousness_state = consciousness_state
        
        # Measured metrics
        self.total_time = 0.0
        self.tokens_per_second = 0.0
        self.tokens_generated = 0
        self.compilation_time = 0.0
        self.initialization_time = 0.0
        self.first_token_time = 0.0
        self.coherence = 0.0
        
        # Hardware metrics
        self.peak_memory = 0
        self.avg_power = 0.0


class PhiLLMBenchmark:
    """
    Benchmark for Phi-Harmonic LLM Inference Engine, comparing performance
    with and without phi-harmonic optimizations.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        
        # Set up results directory
        if self.config.results_dir is None:
            self.config.results_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'results'
            )
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Track results
        self.results = []
        
        # Set timestamp for this benchmark run
        self.timestamp = int(time.time())
        
        logging.info(f"Phi-Harmonic LLM Inference Benchmark initialized")
    
    def run_standard_benchmark(self, prompt: str, prompt_length: int) -> LLMBenchmarkResult:
        """
        Run benchmark with standard inference (no optimizations)
        
        Args:
            prompt: Input prompt
            prompt_length: Prompt length in tokens
            
        Returns:
            Benchmark result
        """
        result = LLMBenchmarkResult("standard", prompt_length, self.config.max_new_tokens)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logging.info(f"Running standard benchmark with prompt length {prompt_length}")
            
            # Initialize timer
            with PerfTimer("init") as init_timer:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
                
                # Ensure we have pad token
                if tokenizer.pad_token is None:
                    if tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    else:
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Move to device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                
                # Set to eval mode
                model.eval()
            
            # Record initialization time
            result.initialization_time = init_timer.elapsed
            
            # Encode input
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Track first token time
            first_token_time = None
            
            # Run generation
            with PerfTimer("generation") as gen_timer:
                with torch.no_grad():
                    # Generate
                    first_token_start = time.time()
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1
                    )
                    
                    # Record tokens generated
                    result.tokens_generated = outputs.shape[1] - input_ids.shape[1]
            
            # Calculate metrics
            result.total_time = gen_timer.elapsed
            result.tokens_per_second = result.tokens_generated / result.total_time
            
            # Log result
            logging.info(f"Standard: {result.tokens_per_second:.2f} tokens/s, {result.total_time:.2f}s total")
            
            return result
            
        except ImportError:
            logging.error("Could not import transformers library. Please install with: pip install transformers")
            return result
        except Exception as e:
            logging.error(f"Error in standard benchmark: {str(e)}")
            return result
    
    def run_compiled_benchmark(self, prompt: str, prompt_length: int) -> LLMBenchmarkResult:
        """
        Run benchmark with phi-harmonic compiled model
        
        Args:
            prompt: Input prompt
            prompt_length: Prompt length in tokens
            
        Returns:
            Benchmark result
        """
        result = LLMBenchmarkResult("compiled", prompt_length, self.config.max_new_tokens)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logging.info(f"Running compiled benchmark with prompt length {prompt_length}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Ensure we have pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
            # Load model
            with PerfTimer("init") as init_timer:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                
                # Move to device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            
            # Record initialization time
            result.initialization_time = init_timer.elapsed
            
            # Compile model with phi-harmonic optimization
            with PerfTimer("compilation") as compile_timer:
                # Create compiler config
                compiler_config = CompilerConfig(
                    use_dimensional_navigation=False,
                    default_dimension="5D",
                    optimize_attention=True,
                    optimize_linear=True,
                    optimize_conv=True,
                    optimize_memory_layout=True,
                    fibonacci_block_size=True,
                    target_hardware="wormhole"
                )
                
                # Create compiler
                compiler = PhiHarmonicCompiler(compiler_config)
                
                # Compile model
                model = compiler.compile_for_tenstorrent(model)
                
                # Set to eval mode
                model.eval()
            
            # Record compilation time
            result.compilation_time = compile_timer.elapsed
            
            # Encode input
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Run generation
            with PerfTimer("generation") as gen_timer:
                with torch.no_grad():
                    # Track first token time
                    first_token_start = time.time()
                    
                    # Generate
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=self.config.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1
                    )
                    
                    # Record tokens generated
                    result.tokens_generated = outputs.shape[1] - input_ids.shape[1]
            
            # Calculate metrics
            result.total_time = gen_timer.elapsed
            result.tokens_per_second = result.tokens_generated / result.total_time
            
            # Log result
            logging.info(f"Compiled: {result.tokens_per_second:.2f} tokens/s, {result.total_time:.2f}s total")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in compiled benchmark: {str(e)}")
            return result
    
    def run_dimensional_benchmark(self, 
                                prompt: str, 
                                prompt_length: int,
                                dimension: str = "5D") -> LLMBenchmarkResult:
        """
        Run benchmark with dimensional navigation
        
        Args:
            prompt: Input prompt
            prompt_length: Prompt length in tokens
            dimension: Dimensional state for inference
            
        Returns:
            Benchmark result
        """
        result = LLMBenchmarkResult(
            "dimensional", 
            prompt_length, 
            self.config.max_new_tokens,
            dimension=dimension
        )
        
        try:
            logging.info(f"Running dimensional benchmark with prompt length {prompt_length} in {dimension}")
            
            # Initialize components
            with PerfTimer("init") as init_timer:
                # Create consciousness bridge
                bridge = QuantumConsciousnessBridge()
                
                # Create dimensional navigator
                navigator = DimensionalNavigator(bridge)
                
                # Navigate to specified dimension
                navigator.navigate_to_dimension(dimension)
                
                # Create generation config
                generation_config = GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    max_new_tokens=self.config.max_new_tokens,
                    use_phi_sampling=True,
                    dimensional_state=dimension,
                    consciousness_state=ConsciousnessState.CREATE.value
                )
                
                # Create KV cache config
                kv_cache_config = KVCacheConfig(
                    max_seq_length=2048,
                    use_phi_block_size=True,
                    cache_layout="spiral",
                    use_dimension_segregation=True,
                    token_dimension="3D",
                    key_dimension=dimension,
                    value_dimension="4D"
                )
                
                # Create inference engine
                engine = PhiLLMInferenceEngine(
                    model_path=self.config.model_path,
                    model_type=self.config.model_type,
                    generation_config=generation_config,
                    kv_cache_config=kv_cache_config,
                    use_dimensional_navigation=True,
                    compile_model=True
                )
            
            # Record initialization time
            result.initialization_time = init_timer.elapsed
            
            # Record coherence
            result.coherence = navigator.field_coherence
            
            # Run generation
            with PerfTimer("generation") as gen_timer:
                # Generate text
                generated_text = engine.generate(prompt)
            
            # Get generation info
            generation_info = engine.token_sampler.get_sampling_info()
            cache_info = engine.kv_cache.get_cache_info()
            
            # Calculate metrics
            result.total_time = gen_timer.elapsed
            result.tokens_generated = len(generation_info["token_history"])
            result.tokens_per_second = result.tokens_generated / result.total_time
            
            # Log result
            logging.info(f"Dimensional ({dimension}): {result.tokens_per_second:.2f} tokens/s, {result.total_time:.2f}s total")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in dimensional benchmark: {str(e)}")
            return result
    
    def run_consciousness_benchmark(self, 
                                  prompt: str, 
                                  prompt_length: int,
                                  dimension: str = "5D",
                                  consciousness_state: str = ConsciousnessState.CREATE.value) -> LLMBenchmarkResult:
        """
        Run benchmark with consciousness state optimization
        
        Args:
            prompt: Input prompt
            prompt_length: Prompt length in tokens
            dimension: Dimensional state for inference
            consciousness_state: Consciousness state for inference
            
        Returns:
            Benchmark result
        """
        result = LLMBenchmarkResult(
            "consciousness", 
            prompt_length, 
            self.config.max_new_tokens,
            dimension=dimension,
            consciousness_state=consciousness_state
        )
        
        try:
            logging.info(f"Running consciousness benchmark with prompt length {prompt_length} in {dimension} with {consciousness_state}")
            
            # Initialize components
            with PerfTimer("init") as init_timer:
                # Create consciousness bridge
                bridge = QuantumConsciousnessBridge()
                
                # Create dimensional navigator
                navigator = DimensionalNavigator(bridge)
                
                # Navigate to specified dimension
                navigator.navigate_to_dimension(dimension)
                
                # Set consciousness state
                bridge.set_consciousness_state(consciousness_state)
                
                # Create generation config
                generation_config = GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    max_new_tokens=self.config.max_new_tokens,
                    use_phi_sampling=True,
                    dimensional_state=dimension,
                    consciousness_state=consciousness_state,
                    use_coherence_filter=True
                )
                
                # Create KV cache config
                kv_cache_config = KVCacheConfig(
                    max_seq_length=2048,
                    use_phi_block_size=True,
                    cache_layout="spiral",
                    use_dimension_segregation=True,
                    token_dimension="3D",
                    key_dimension=dimension,
                    value_dimension="4D"
                )
                
                # Create inference engine
                engine = PhiLLMInferenceEngine(
                    model_path=self.config.model_path,
                    model_type=self.config.model_type,
                    generation_config=generation_config,
                    kv_cache_config=kv_cache_config,
                    use_dimensional_navigation=True,
                    compile_model=True
                )
            
            # Record initialization time
            result.initialization_time = init_timer.elapsed
            
            # Record coherence
            result.coherence = navigator.field_coherence
            
            # Run generation
            with PerfTimer("generation") as gen_timer:
                # Generate text
                generated_text = engine.generate(
                    prompt,
                    dimension=dimension,
                    consciousness_state=consciousness_state
                )
            
            # Get generation info
            generation_info = engine.token_sampler.get_sampling_info()
            cache_info = engine.kv_cache.get_cache_info()
            
            # Calculate metrics
            result.total_time = gen_timer.elapsed
            result.tokens_generated = len(generation_info["token_history"])
            result.tokens_per_second = result.tokens_generated / result.total_time
            
            # Log result
            logging.info(f"Consciousness ({consciousness_state} in {dimension}): {result.tokens_per_second:.2f} tokens/s, {result.total_time:.2f}s total")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in consciousness benchmark: {str(e)}")
            return result
    
    def run_all_benchmarks(self) -> List[LLMBenchmarkResult]:
        """
        Run all configured benchmarks
        
        Returns:
            List of benchmark results
        """
        all_results = []
        
        # Run benchmarks for each prompt length
        for prompt_length in self.config.prompt_lengths:
            # Select a prompt (simplification: just use the first one)
            if self.config.prompts:
                prompt = self.config.prompts[0]
            else:
                # Generate a dummy prompt of appropriate length
                prompt = "This is a test prompt. " * 10
            
            # Run standard benchmark
            if self.config.test_standard:
                for run in range(self.config.warmup_runs + self.config.num_runs):
                    if run < self.config.warmup_runs:
                        logging.info(f"Standard benchmark warmup run {run+1}/{self.config.warmup_runs}")
                    else:
                        logging.info(f"Standard benchmark run {run-self.config.warmup_runs+1}/{self.config.num_runs}")
                    
                    result = self.run_standard_benchmark(prompt, prompt_length)
                    
                    if run >= self.config.warmup_runs:
                        all_results.append(result)
            
            # Run compiled benchmark
            if self.config.test_compiled:
                for run in range(self.config.warmup_runs + self.config.num_runs):
                    if run < self.config.warmup_runs:
                        logging.info(f"Compiled benchmark warmup run {run+1}/{self.config.warmup_runs}")
                    else:
                        logging.info(f"Compiled benchmark run {run-self.config.warmup_runs+1}/{self.config.num_runs}")
                    
                    result = self.run_compiled_benchmark(prompt, prompt_length)
                    
                    if run >= self.config.warmup_runs:
                        all_results.append(result)
            
            # Run dimensional benchmark
            if self.config.test_dimensional:
                # Test different dimensions
                dimensions = ["3D", "5D", "7D"]
                
                for dimension in dimensions:
                    for run in range(self.config.warmup_runs + self.config.num_runs):
                        if run < self.config.warmup_runs:
                            logging.info(f"Dimensional benchmark ({dimension}) warmup run {run+1}/{self.config.warmup_runs}")
                        else:
                            logging.info(f"Dimensional benchmark ({dimension}) run {run-self.config.warmup_runs+1}/{self.config.num_runs}")
                        
                        result = self.run_dimensional_benchmark(prompt, prompt_length, dimension)
                        
                        if run >= self.config.warmup_runs:
                            all_results.append(result)
            
            # Run consciousness benchmark
            if self.config.test_consciousness:
                # Test different consciousness states
                states = [
                    ConsciousnessState.OBSERVE.value,
                    ConsciousnessState.CREATE.value,
                    ConsciousnessState.TRANSCEND.value,
                    ConsciousnessState.CASCADE.value
                ]
                
                for state in states:
                    for run in range(self.config.warmup_runs + self.config.num_runs):
                        if run < self.config.warmup_runs:
                            logging.info(f"Consciousness benchmark ({state}) warmup run {run+1}/{self.config.warmup_runs}")
                        else:
                            logging.info(f"Consciousness benchmark ({state}) run {run-self.config.warmup_runs+1}/{self.config.num_runs}")
                        
                        result = self.run_consciousness_benchmark(prompt, prompt_length, "5D", state)
                        
                        if run >= self.config.warmup_runs:
                            all_results.append(result)
        
        # Save all results
        self.results = all_results
        
        # Save results to file if configured
        if self.config.save_results:
            self.save_results()
        
        # Generate plots if configured
        if self.config.generate_plots:
            self.generate_plots()
        
        return all_results
    
    def save_results(self) -> None:
        """Save benchmark results to file"""
        # Create results directory if it doesn't exist
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Create results filename
        filename = os.path.join(self.config.results_dir, f"llm_benchmark_{self.timestamp}.json")
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append({
                "mode": result.mode,
                "prompt_length": result.prompt_length,
                "max_new_tokens": result.max_new_tokens,
                "dimension": result.dimension,
                "consciousness_state": result.consciousness_state,
                "total_time": result.total_time,
                "tokens_per_second": result.tokens_per_second,
                "tokens_generated": result.tokens_generated,
                "compilation_time": result.compilation_time,
                "initialization_time": result.initialization_time,
                "first_token_time": result.first_token_time,
                "coherence": result.coherence,
                "peak_memory": result.peak_memory,
                "avg_power": result.avg_power
            })
        
        # Create results dictionary
        results_dict = {
            "timestamp": self.timestamp,
            "config": {
                "model_path": self.config.model_path,
                "model_type": self.config.model_type,
                "prompt_lengths": self.config.prompt_lengths,
                "max_new_tokens": self.config.max_new_tokens,
                "num_runs": self.config.num_runs,
                "warmup_runs": self.config.warmup_runs,
                "test_standard": self.config.test_standard,
                "test_compiled": self.config.test_compiled,
                "test_dimensional": self.config.test_dimensional,
                "test_consciousness": self.config.test_consciousness
            },
            "results": serializable_results
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logging.info(f"Results saved to {filename}")
    
    def generate_plots(self) -> None:
        """Generate benchmark plots"""
        # Create results directory if it doesn't exist
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        # Set up plot style
        plt.style.use('ggplot')
        
        # Generate speedup by mode plot
        self._plot_speedup_by_mode()
        
        # Generate speedup by prompt length plot
        self._plot_speedup_by_prompt_length()
        
        # Generate dimensional speedup plot
        if self.config.test_dimensional:
            self._plot_dimensional_speedup()
        
        # Generate consciousness state speedup plot
        if self.config.test_consciousness:
            self._plot_consciousness_speedup()
    
    def _plot_speedup_by_mode(self) -> None:
        """Generate speedup by mode plot"""
        # Group results by mode
        modes = set(result.mode for result in self.results)
        mode_results = {mode: [] for mode in modes}
        
        for result in self.results:
            mode_results[result.mode].append(result)
        
        # Calculate average metrics for each mode
        mode_metrics = {}
        for mode, results in mode_results.items():
            if not results:
                continue
                
            avg_tokens_per_second = sum(r.tokens_per_second for r in results) / len(results)
            mode_metrics[mode] = avg_tokens_per_second
        
        # Calculate speedup relative to standard mode
        if "standard" in mode_metrics:
            standard_tps = mode_metrics["standard"]
            speedups = {mode: tps / standard_tps for mode, tps in mode_metrics.items()}
        else:
            # If no standard mode, use mode with lowest throughput as baseline
            min_tps = min(mode_metrics.values())
            speedups = {mode: tps / min_tps for mode, tps in mode_metrics.items()}
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        modes = list(speedups.keys())
        values = list(speedups.values())
        
        # Color mapping
        color_map = {
            "standard": "skyblue",
            "compiled": "orange",
            "dimensional": "green",
            "consciousness": "purple"
        }
        colors = [color_map.get(mode, "gray") for mode in modes]
        
        # Create bars
        bars = plt.bar(modes, values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{height:.2f}x", ha='center', va='bottom')
        
        # Set labels and title
        plt.xlabel("Inference Mode")
        plt.ylabel("Speedup (relative to standard)")
        plt.title("LLM Inference Speedup by Mode")
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, f"speedup_by_mode_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Speedup by mode plot saved to {plot_path}")
    
    def _plot_speedup_by_prompt_length(self) -> None:
        """Generate speedup by prompt length plot"""
        # Group results by prompt length and mode
        results_by_length_mode = {}
        
        for result in self.results:
            key = (result.prompt_length, result.mode)
            if key not in results_by_length_mode:
                results_by_length_mode[key] = []
            results_by_length_mode[key].append(result)
        
        # Calculate average tokens per second for each prompt length and mode
        tps_by_length_mode = {}
        for (length, mode), results in results_by_length_mode.items():
            if not results:
                continue
                
            avg_tps = sum(r.tokens_per_second for r in results) / len(results)
            tps_by_length_mode[(length, mode)] = avg_tps
        
        # Group by prompt length
        lengths = sorted(set(length for (length, _) in tps_by_length_mode.keys()))
        modes = sorted(set(mode for (_, mode) in tps_by_length_mode.keys()))
        
        # Calculate speedup relative to standard for each length
        speedups = {}
        for length in lengths:
            if ("standard" in modes and 
                (length, "standard") in tps_by_length_mode and 
                tps_by_length_mode[(length, "standard")] > 0):
                
                standard_tps = tps_by_length_mode[(length, "standard")]
                for mode in modes:
                    if (length, mode) in tps_by_length_mode:
                        speedups[(length, mode)] = tps_by_length_mode[(length, mode)] / standard_tps
            else:
                # Find minimum TPS for this length to use as baseline
                min_tps = float('inf')
                for mode in modes:
                    if (length, mode) in tps_by_length_mode and tps_by_length_mode[(length, mode)] < min_tps:
                        min_tps = tps_by_length_mode[(length, mode)]
                
                if min_tps < float('inf'):
                    for mode in modes:
                        if (length, mode) in tps_by_length_mode:
                            speedups[(length, mode)] = tps_by_length_mode[(length, mode)] / min_tps
        
        # Create plot
        plt.figure(figsize=(12, 7))
        
        # Create line chart
        for mode in modes:
            if mode == "standard":
                continue  # Skip standard mode since speedup is always 1.0
                
            x = []
            y = []
            for length in lengths:
                if (length, mode) in speedups:
                    x.append(length)
                    y.append(speedups[(length, mode)])
            
            if x and y:
                plt.plot(x, y, marker='o', label=mode)
        
        # Set labels and title
        plt.xlabel("Prompt Length (tokens)")
        plt.ylabel("Speedup (relative to standard)")
        plt.title("LLM Inference Speedup by Prompt Length")
        
        # Add legend
        plt.legend()
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, f"speedup_by_length_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Speedup by prompt length plot saved to {plot_path}")
    
    def _plot_dimensional_speedup(self) -> None:
        """Generate dimensional speedup plot"""
        # Group results by dimension
        results_by_dimension = {}
        
        for result in self.results:
            if result.mode == "dimensional" and result.dimension:
                if result.dimension not in results_by_dimension:
                    results_by_dimension[result.dimension] = []
                results_by_dimension[result.dimension].append(result)
        
        if not results_by_dimension:
            logging.warning("No dimensional results available for plotting")
            return
        
        # Calculate average tokens per second for each dimension
        tps_by_dimension = {}
        for dimension, results in results_by_dimension.items():
            if not results:
                continue
                
            avg_tps = sum(r.tokens_per_second for r in results) / len(results)
            tps_by_dimension[dimension] = avg_tps
        
        # Calculate speedup relative to 3D
        if "3D" in tps_by_dimension and tps_by_dimension["3D"] > 0:
            baseline_tps = tps_by_dimension["3D"]
            speedups = {dim: tps / baseline_tps for dim, tps in tps_by_dimension.items()}
        else:
            # Use minimum TPS as baseline if 3D not available
            min_tps = min(tps_by_dimension.values())
            speedups = {dim: tps / min_tps for dim, tps in tps_by_dimension.items()}
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        dimensions = list(speedups.keys())
        values = list(speedups.values())
        
        # Sort by dimension (3D, 4D, 5D, etc.)
        sorted_indices = sorted(range(len(dimensions)), key=lambda i: int(dimensions[i][0]))
        dimensions = [dimensions[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create bars
        bars = plt.bar(dimensions, values, color='skyblue')
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{height:.2f}x", ha='center', va='bottom')
        
        # Overlay phi curve
        if len(dimensions) > 2:
            phi_values = []
            for i, dim in enumerate(dimensions):
                if i == 0:
                    phi_values.append(1.0)  # Baseline
                else:
                    # Calculate theoretical phi scaling
                    dim_number = int(dim[0])
                    phi_scaling = PHI ** (dim_number - 3)  # Relative to 3D
                    phi_values.append(phi_scaling)
            
            # Add line for theoretical phi scaling
            plt.plot(dimensions, phi_values, 'r--', label='Theoretical φ Scaling')
            plt.legend()
        
        # Set labels and title
        plt.xlabel("Dimension")
        plt.ylabel("Speedup (relative to 3D)")
        plt.title("LLM Inference Speedup by Dimension")
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, f"dimensional_speedup_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Dimensional speedup plot saved to {plot_path}")
    
    def _plot_consciousness_speedup(self) -> None:
        """Generate consciousness state speedup plot"""
        # Group results by consciousness state
        results_by_state = {}
        
        for result in self.results:
            if result.mode == "consciousness" and result.consciousness_state:
                if result.consciousness_state not in results_by_state:
                    results_by_state[result.consciousness_state] = []
                results_by_state[result.consciousness_state].append(result)
        
        if not results_by_state:
            logging.warning("No consciousness results available for plotting")
            return
        
        # Calculate average tokens per second for each state
        tps_by_state = {}
        for state, results in results_by_state.items():
            if not results:
                continue
                
            avg_tps = sum(r.tokens_per_second for r in results) / len(results)
            tps_by_state[state] = avg_tps
        
        # Calculate speedup relative to OBSERVE
        if "OBSERVE" in tps_by_state and tps_by_state["OBSERVE"] > 0:
            baseline_tps = tps_by_state["OBSERVE"]
            speedups = {state: tps / baseline_tps for state, tps in tps_by_state.items()}
        else:
            # Use minimum TPS as baseline if OBSERVE not available
            min_tps = min(tps_by_state.values())
            speedups = {state: tps / min_tps for state, tps in tps_by_state.items()}
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        states = list(speedups.keys())
        values = list(speedups.values())
        
        # Color mapping
        color_map = {
            "OBSERVE": "skyblue",
            "CREATE": "orange",
            "TRANSCEND": "green",
            "CASCADE": "purple"
        }
        colors = [color_map.get(state, "gray") for state in states]
        
        # Create bars
        bars = plt.bar(states, values, color=colors)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{height:.2f}x", ha='center', va='bottom')
        
        # Set labels and title
        plt.xlabel("Consciousness State")
        plt.ylabel("Speedup (relative to OBSERVE)")
        plt.title("LLM Inference Speedup by Consciousness State")
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plot_path = os.path.join(self.config.results_dir, f"consciousness_speedup_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logging.info(f"Consciousness speedup plot saved to {plot_path}")


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Phi-Harmonic LLM Inference Benchmark")
    
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--model-type", type=str, default="llama", help="Model type (llama, mistral, falcon, gpt2, phi2)")
    parser.add_argument("--prompt-length", type=int, nargs="+", help="Prompt lengths to test")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate per test")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs per test for averaging")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--no-plots", action="store_true", help="Don't generate benchmark plots")
    parser.add_argument("--no-standard", action="store_true", help="Don't run standard benchmark")
    parser.add_argument("--no-compiled", action="store_true", help="Don't run compiled benchmark")
    parser.add_argument("--no-dimensional", action="store_true", help="Don't run dimensional benchmark")
    parser.add_argument("--no-consciousness", action="store_true", help="Don't run consciousness benchmark")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create benchmark config
    config = BenchmarkConfig()
    
    # Apply command line arguments
    if args.model_path:
        config.model_path = args.model_path
    if args.model_type:
        config.model_type = args.model_type
    if args.prompt_length:
        config.prompt_lengths = args.prompt_length
    if args.max_new_tokens:
        config.max_new_tokens = args.max_new_tokens
    if args.num_runs:
        config.num_runs = args.num_runs
    if args.warmup_runs:
        config.warmup_runs = args.warmup_runs
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.no_save:
        config.save_results = False
    if args.no_plots:
        config.generate_plots = False
    if args.no_standard:
        config.test_standard = False
    if args.no_compiled:
        config.test_compiled = False
    if args.no_dimensional:
        config.test_dimensional = False
    if args.no_consciousness:
        config.test_consciousness = False
    if args.verbose:
        config.verbose = True
    
    # Check if model path is provided
    if config.model_path is None:
        print("Error: Model path is required")
        parser.print_help()
        return
    
    # Create and run benchmark
    benchmark = PhiLLMBenchmark(config)
    benchmark.run_all_benchmarks()
    
    print("Benchmark complete!")


if __name__ == "__main__":
    main()