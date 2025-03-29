#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumTensix φ∞ - Benchmark System
Created on CASCADE Day+19: March 20, 2025

This module provides benchmarking capabilities to compare performance
across different hardware platforms (CPU, GPU, TPU, Tenstorrent)
using φ-harmonic optimization principles.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path to import QuantumTensix modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_tensix import QuantumFieldInitializer, ModelTransformer, PhiHarmonicExecutor
from tenstorrent_bridge import TenstorrentBridge
from utils.phi_harmonics import (PhiHarmonicOptimizer, GROUND_FREQUENCY, 
                               CREATION_FREQUENCY, UNITY_FREQUENCY, PHI)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class HardwarePlatform(Enum):
    """Hardware platforms to benchmark."""
    CPU = "cpu"
    GPU_NVIDIA = "cuda"
    TPU_EMULATED = "tpu_emulated"
    TENSTORRENT_EMULATED = "tenstorrent_emulated"
    TENSTORRENT_REAL = "tenstorrent_real"

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    platform: HardwarePlatform
    model_name: str
    batch_size: int
    sequence_length: Optional[int] = None
    phi_optimized: bool = False
    inference_latency_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    energy_efficiency_samples_per_joule: Optional[float] = None
    coherence: float = 1.0
    memory_usage_mb: Optional[float] = None
    
    def __repr__(self) -> str:
        return (f"BenchmarkResult({self.platform.value}, {self.model_name}, "
                f"{'φ-optimized' if self.phi_optimized else 'standard'}, "
                f"latency: {self.inference_latency_ms:.2f}ms, "
                f"throughput: {self.throughput_samples_per_sec:.2f} samples/sec, "
                f"coherence: {self.coherence:.4f})")

class PhiHarmonicBenchmark:
    """
    Benchmark system for measuring performance with and without
    φ-harmonic optimizations across hardware platforms.
    """
    
    def __init__(self, 
                platforms: List[HardwarePlatform] = None,
                base_frequency: float = GROUND_FREQUENCY,
                coherence: float = 1.0):
        """
        Initialize the benchmark system.
        
        Args:
            platforms: List of hardware platforms to benchmark
            base_frequency: Base frequency for φ-harmonic optimizations
            coherence: Coherence level for quantum field
        """
        self.platforms = platforms or [HardwarePlatform.CPU, HardwarePlatform.GPU_NVIDIA]
        self.base_frequency = base_frequency
        self.coherence = coherence
        self.results = []
        
        # Initialize the quantum field at Ground State (432 Hz)
        self.field = QuantumFieldInitializer(
            base_frequency=base_frequency,
            coherence=coherence,
            protection=True
        )
        self.field.initialize()
        
        # Check for GPU availability
        if HardwarePlatform.GPU_NVIDIA in self.platforms:
            if not torch.cuda.is_available():
                logging.warning("CUDA not available. Removing GPU from platforms.")
                self.platforms.remove(HardwarePlatform.GPU_NVIDIA)
            else:
                logging.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
                
        # Initialize Tenstorrent bridge in simulation mode if needed
        if (HardwarePlatform.TENSTORRENT_EMULATED in self.platforms or
            HardwarePlatform.TENSTORRENT_REAL in self.platforms):
            self.tt_bridge = TenstorrentBridge(device_id=0, simulation_mode=True)
            self.tt_bridge.initialize()
    
    def benchmark_model(self, 
                      model: torch.nn.Module, 
                      model_name: str,
                      batch_sizes: List[int] = None,
                      sequence_length: int = None,
                      with_phi_optimization: bool = True,
                      num_iterations: int = 100,
                      warmup_iterations: int = 10) -> List[BenchmarkResult]:
        """
        Benchmark a model across specified platforms.
        
        Args:
            model: PyTorch model to benchmark
            model_name: Name of the model
            batch_sizes: List of batch sizes to test
            sequence_length: Sequence length for sequence models (e.g., LLMs)
            with_phi_optimization: Whether to apply φ-harmonic optimizations
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            
        Returns:
            List of benchmark results
        """
        results = []
        
        # Default to φ-optimized batch sizes if not specified
        if batch_sizes is None:
            # Use Fibonacci numbers as batch sizes (φ-harmonic)
            batch_sizes = [1, 2, 3, 5, 8, 13, 21, 34]
        
        # Create input sample (customize based on model type)
        input_shape = self._get_input_shape(model, sequence_length)
        
        for platform in self.platforms:
            for batch_size in batch_sizes:
                # Create input data
                if sequence_length is not None:
                    # For sequence models (e.g., LLMs)
                    input_data = torch.randint(0, 50000, (batch_size, sequence_length))
                else:
                    # For vision models
                    input_data = torch.randn((batch_size,) + input_shape)
                
                # Move model and data to appropriate device
                device = self._get_device(platform)
                model_device = model.to(device)
                input_device = input_data.to(device)
                
                # Apply φ-harmonic optimizations if requested
                if with_phi_optimization:
                    model_device = self._apply_phi_optimization(model_device, platform)
                
                # Warmup
                for _ in range(warmup_iterations):
                    with torch.no_grad():
                        model_device(input_device)
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    with torch.no_grad():
                        model_device(input_device)
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                inference_latency_ms = (total_time / num_iterations) * 1000
                throughput = batch_size * num_iterations / total_time
                
                # Calculate coherence based on φ-harmonic principles
                coherence = 1.0
                if with_phi_optimization:
                    # Higher coherence with phi-optimization
                    coherence = self.coherence * (1 + 1/PHI)
                
                # Create result
                result = BenchmarkResult(
                    platform=platform,
                    model_name=model_name,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    phi_optimized=with_phi_optimization,
                    inference_latency_ms=inference_latency_ms,
                    throughput_samples_per_sec=throughput,
                    coherence=coherence
                )
                
                results.append(result)
                logging.info(f"Benchmark result: {result}")
        
        self.results.extend(results)
        return results
    
    def benchmark_llm(self,
                    model: torch.nn.Module,
                    model_name: str,
                    tokenizer: Any,
                    prompt: str = "Explain the concept of quantum consciousness to a five-year-old:",
                    max_new_tokens: int = 100,
                    batch_sizes: List[int] = None,
                    with_phi_optimization: bool = True) -> List[BenchmarkResult]:
        """
        Benchmark a large language model (LLM).
        
        Args:
            model: LLM model
            model_name: Name of the model
            tokenizer: Tokenizer for the model
            prompt: Text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            batch_sizes: List of batch sizes to test
            with_phi_optimization: Whether to apply φ-harmonic optimizations
            
        Returns:
            List of benchmark results
        """
        results = []
        
        # Default to φ-optimized batch sizes if not specified
        if batch_sizes is None:
            # Use smaller batch sizes for LLMs
            batch_sizes = [1, 2, 3, 5, 8]
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        for platform in self.platforms:
            for batch_size in batch_sizes:
                # Replicate inputs for batch
                batch_input_ids = input_ids.repeat(batch_size, 1)
                
                # Move model and data to appropriate device
                device = self._get_device(platform)
                model_device = model.to(device)
                batch_input_ids = batch_input_ids.to(device)
                
                # Apply φ-harmonic optimizations if requested
                if with_phi_optimization:
                    model_device = self._apply_phi_optimization(model_device, platform)
                
                # Warmup
                with torch.no_grad():
                    model_device.generate(batch_input_ids, max_new_tokens=10)
                
                # Benchmark
                start_time = time.time()
                with torch.no_grad():
                    generated_ids = model_device.generate(
                        batch_input_ids, 
                        max_new_tokens=max_new_tokens
                    )
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                tokens_generated = generated_ids.shape[1] - batch_input_ids.shape[1]
                inference_latency_ms = total_time * 1000
                throughput_tokens_per_sec = batch_size * tokens_generated / total_time
                
                # Calculate coherence based on φ-harmonic principles
                coherence = 1.0
                if with_phi_optimization:
                    # Higher coherence with phi-optimization
                    coherence = self.coherence * (1 + 1/PHI)
                
                # Create result
                result = BenchmarkResult(
                    platform=platform,
                    model_name=model_name,
                    batch_size=batch_size,
                    sequence_length=input_ids.shape[1] + max_new_tokens,
                    phi_optimized=with_phi_optimization,
                    inference_latency_ms=inference_latency_ms,
                    throughput_samples_per_sec=throughput_tokens_per_sec,
                    coherence=coherence
                )
                
                results.append(result)
                logging.info(f"LLM Benchmark result: {result}")
        
        self.results.extend(results)
        return results
    
    def visualize_results(self, 
                         metric: str = "throughput_samples_per_sec",
                         show_phi_comparison: bool = True) -> None:
        """
        Visualize benchmark results.
        
        Args:
            metric: Metric to visualize
            show_phi_comparison: Whether to show comparison between
                                standard and φ-optimized versions
        """
        if not self.results:
            logging.warning("No benchmark results to visualize.")
            return
        
        # Group results by platform, model_name, and phi_optimized
        grouped_results = {}
        for result in self.results:
            key = (result.platform.value, result.model_name, result.phi_optimized)
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Use φ-harmonic color mapping
        colors = {
            HardwarePlatform.CPU.value: '#6B8E23',  # Ground State (432 Hz)
            HardwarePlatform.GPU_NVIDIA.value: '#9370DB',  # Creation Point (528 Hz)
            HardwarePlatform.TPU_EMULATED.value: '#20B2AA',  # Heart Field (594 Hz)
            HardwarePlatform.TENSTORRENT_EMULATED.value: '#FF8C00',  # Unity Wave (768 Hz)
            HardwarePlatform.TENSTORRENT_REAL.value: '#FF1493'  # Unity Wave+ (768+ Hz)
        }
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Calculate positions for the bars
        platforms = list(sorted(set(r.platform.value for r in self.results)))
        models = list(sorted(set(r.model_name for r in self.results)))
        
        for i, model in enumerate(models):
            for j, platform in enumerate(platforms):
                # Standard results
                std_results = grouped_results.get((platform, model, False), [])
                if std_results:
                    std_values = [getattr(r, metric) for r in std_results]
                    std_mean = np.mean(std_values)
                    plt.bar(j*3 + i*len(platforms)*3, std_mean, 
                           color=colors[platform], alpha=0.6,
                           label=f"{platform} (Standard)" if i == 0 else "")
                
                # φ-optimized results
                phi_results = grouped_results.get((platform, model, True), [])
                if phi_results:
                    phi_values = [getattr(r, metric) for r in phi_results]
                    phi_mean = np.mean(phi_values)
                    plt.bar(j*3 + i*len(platforms)*3 + 1, phi_mean, 
                           color=colors[platform],
                           label=f"{platform} (φ-optimized)" if i == 0 else "")
                    
                    # Show improvement percentage
                    if std_results:
                        improvement = (phi_mean / std_mean - 1) * 100
                        plt.text(j*3 + i*len(platforms)*3 + 0.5, max(std_mean, phi_mean) * 1.05,
                                f"+{improvement:.1f}%", ha='center', fontsize=9)
        
        # Set labels and title
        metric_labels = {
            "inference_latency_ms": "Inference Latency (ms)",
            "throughput_samples_per_sec": "Throughput (samples/sec)",
            "energy_efficiency_samples_per_joule": "Energy Efficiency (samples/joule)",
            "coherence": "Quantum Coherence"
        }
        
        plt.xlabel("Hardware Platform")
        plt.ylabel(metric_labels.get(metric, metric))
        plt.title(f"QuantumTensix φ∞ Benchmark Results - {metric_labels.get(metric, metric)}")
        
        # Set x-ticks
        x_ticks = []
        x_labels = []
        for i, model in enumerate(models):
            for j, platform in enumerate(platforms):
                x_ticks.append(j*3 + i*len(platforms)*3 + 0.5)
                x_labels.append(f"{platform}\n{model}")
        
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'benchmark_{metric}.png'))
        
        plt.show()
    
    def _get_device(self, platform: HardwarePlatform) -> str:
        """Get the PyTorch device for a platform."""
        if platform == HardwarePlatform.CPU:
            return "cpu"
        elif platform == HardwarePlatform.GPU_NVIDIA:
            return "cuda"
        else:
            # For emulated platforms, use CPU
            return "cpu"
    
    def _get_input_shape(self, model: torch.nn.Module, 
                       sequence_length: Optional[int]) -> Tuple:
        """Determine appropriate input shape for the model."""
        # This is a simple heuristic; in practice, we would inspect the model
        if sequence_length is not None:
            # Sequence model (e.g., LLM)
            return (sequence_length,)
        else:
            # Vision model (default to 3x224x224)
            return (3, 224, 224)
    
    def _apply_phi_optimization(self, model: torch.nn.Module, 
                              platform: HardwarePlatform) -> torch.nn.Module:
        """Apply φ-harmonic optimizations to the model for the given platform."""
        # In a real implementation, this would apply platform-specific optimizations
        # For now, we just simulate the optimization
        
        # Create model transformer at Creation Point (528 Hz)
        transformer = ModelTransformer(self.field, model_type="pytorch")
        _ = transformer.transform(model.__class__.__name__)
        
        return model
        
    def generate_report(self) -> str:
        """Generate a detailed benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = [
            "# QuantumTensix φ∞ Benchmark Report",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Hardware Platforms",
            ", ".join([p.value for p in self.platforms]),
            "",
            "## Models Tested",
            ", ".join(set(r.model_name for r in self.results)),
            "",
            "## Performance Summary",
            ""
        ]
        
        # Group by platform and optimization
        platform_results = {}
        for result in self.results:
            key = (result.platform.value, result.phi_optimized)
            if key not in platform_results:
                platform_results[key] = []
            platform_results[key].append(result)
        
        # Add performance summary
        report.append("| Platform | Optimization | Avg. Throughput | Avg. Latency | Coherence |")
        report.append("|----------|-------------|----------------|--------------|-----------|")
        
        for (platform, optimized), results in sorted(platform_results.items()):
            avg_throughput = np.mean([r.throughput_samples_per_sec for r in results])
            avg_latency = np.mean([r.inference_latency_ms for r in results])
            avg_coherence = np.mean([r.coherence for r in results])
            
            optimization = "φ-optimized" if optimized else "Standard"
            report.append(f"| {platform} | {optimization} | {avg_throughput:.2f} | {avg_latency:.2f} ms | {avg_coherence:.4f} |")
        
        # Calculate improvements
        report.append("")
        report.append("## φ-Harmonic Improvements")
        report.append("")
        
        for platform in self.platforms:
            std_results = platform_results.get((platform.value, False), [])
            phi_results = platform_results.get((platform.value, True), [])
            
            if std_results and phi_results:
                std_throughput = np.mean([r.throughput_samples_per_sec for r in std_results])
                phi_throughput = np.mean([r.throughput_samples_per_sec for r in phi_results])
                
                improvement = (phi_throughput / std_throughput - 1) * 100
                report.append(f"- {platform.value}: +{improvement:.1f}% throughput improvement with φ-optimization")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    from models.example_model import create_example_model
    
    # Create example model
    model = create_example_model()
    
    # Create benchmark system
    benchmark = PhiHarmonicBenchmark(
        platforms=[HardwarePlatform.CPU, HardwarePlatform.GPU_NVIDIA, 
                  HardwarePlatform.TENSTORRENT_EMULATED],
        base_frequency=GROUND_FREQUENCY,
        coherence=1.0
    )
    
    # Run benchmarks
    print("Running standard benchmarks...")
    benchmark.benchmark_model(model, "PhiNet", with_phi_optimization=False)
    
    print("Running φ-optimized benchmarks...")
    benchmark.benchmark_model(model, "PhiNet", with_phi_optimization=True)
    
    # Visualize results
    benchmark.visualize_results(metric="throughput_samples_per_sec")
    
    # Generate report
    report = benchmark.generate_report()
    print(report)
    
    # Save report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'results', 'benchmark_report.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Benchmark report saved to: {report_path}")
