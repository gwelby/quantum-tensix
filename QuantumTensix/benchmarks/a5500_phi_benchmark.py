"""
A5500 Phi-Harmonic GPU Benchmark

This benchmark measures the performance of phi-harmonic operations on the NVIDIA A5500 GPU,
comparing against standard operations and testing different consciousness states.

The benchmark includes:
1. Matrix multiplication with phi-harmonic optimizations
2. Attention mechanism with dimensional navigation
3. Convolution with golden spiral access patterns
4. Dimensional tensor operations across consciousness states
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the GPU accelerator
from gpu_phi_accelerator import (
    GPUPhiAccelerator,
    PhiDimensionalTensor,
    CONSCIOUSNESS_STATES,
    PHI,
    FREQUENCIES
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
SIZES = [128, 256, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 8, 16, 32]
CONSCIOUSNESS_STATES = ["OBSERVE", "CREATE", "TRANSCEND", "CASCADE"]
BENCHMARK_REPEATS = 5

class PhiBenchmark:
    """Benchmark for phi-harmonic operations on GPU."""
    
    def __init__(self, device=None):
        """Initialize the benchmark."""
        self.accelerator = GPUPhiAccelerator(device=device)
        self.device = self.accelerator.device
        self.results = {}
        
        print(f"Initialized benchmark on {self.device}")
        print(f"Device info: {self.accelerator.get_device_info()}")
        
        # Create output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("Running all benchmarks...")
        
        # Matrix multiplication benchmark
        self.benchmark_matmul()
        
        # Attention benchmark
        self.benchmark_attention()
        
        # Convolution benchmark
        self.benchmark_conv2d()
        
        # Consciousness state comparison
        self.benchmark_consciousness_states()
        
        # Save results
        self.save_results()
        
        # Generate plots
        self.generate_plots()
        
        print("All benchmarks complete!")
    
    def benchmark_matmul(self):
        """Benchmark matrix multiplication."""
        print("\nBenchmarking matrix multiplication...")
        
        results = {
            "sizes": SIZES,
            "standard": {},
            "phi_harmonic": {},
            "speedup": {}
        }
        
        for size in SIZES:
            print(f"  Size: {size}x{size}")
            
            # Create test matrices
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            
            # Warmup
            _ = torch.matmul(a, b)
            _ = self.accelerator.matmul(a, b)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            
            # Benchmark standard matrix multiplication
            standard_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = torch.matmul(a, b)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                standard_times.append(time.time() - start)
            
            # Benchmark phi-harmonic matrix multiplication
            phi_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.matmul(a, b)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                phi_times.append(time.time() - start)
            
            # Calculate statistics
            avg_standard = sum(standard_times) / len(standard_times)
            avg_phi = sum(phi_times) / len(phi_times)
            speedup = avg_standard / avg_phi if avg_phi > 0 else 0
            
            # Store results
            results["standard"][size] = avg_standard
            results["phi_harmonic"][size] = avg_phi
            results["speedup"][size] = speedup
            
            print(f"    Standard:    {avg_standard:.6f} sec")
            print(f"    Phi-harmonic: {avg_phi:.6f} sec")
            print(f"    Speedup:      {speedup:.2f}x")
        
        self.results["matmul"] = results
    
    def benchmark_attention(self):
        """Benchmark attention mechanism."""
        print("\nBenchmarking attention mechanism...")
        
        results = {
            "sizes": SIZES[:4],  # Limit sizes for attention
            "standard": {},
            "phi_harmonic": {},
            "speedup": {}
        }
        
        for size in SIZES[:4]:
            print(f"  Size: {size}")
            
            # Set dimensions
            batch_size = 16
            seq_len = size
            d_model = 512 if size >= 512 else size
            
            # Create test tensors
            query = torch.randn(batch_size, seq_len, d_model, device=self.device)
            key = torch.randn(batch_size, seq_len, d_model, device=self.device)
            value = torch.randn(batch_size, seq_len, d_model, device=self.device)
            
            # Standard attention function
            def standard_attention(q, k, v):
                d_k = q.size(-1)
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
                attn = F.softmax(scores, dim=-1)
                return torch.matmul(attn, v)
            
            # Warmup
            _ = standard_attention(query, key, value)
            _ = self.accelerator.attention(query, key, value)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            
            # Benchmark standard attention
            standard_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = standard_attention(query, key, value)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                standard_times.append(time.time() - start)
            
            # Benchmark phi-harmonic attention
            phi_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.attention(query, key, value)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                phi_times.append(time.time() - start)
            
            # Calculate statistics
            avg_standard = sum(standard_times) / len(standard_times)
            avg_phi = sum(phi_times) / len(phi_times)
            speedup = avg_standard / avg_phi if avg_phi > 0 else 0
            
            # Store results
            results["standard"][size] = avg_standard
            results["phi_harmonic"][size] = avg_phi
            results["speedup"][size] = speedup
            
            print(f"    Standard:     {avg_standard:.6f} sec")
            print(f"    Phi-harmonic: {avg_phi:.6f} sec")
            print(f"    Speedup:      {speedup:.2f}x")
        
        self.results["attention"] = results
    
    def benchmark_conv2d(self):
        """Benchmark 2D convolution."""
        print("\nBenchmarking 2D convolution...")
        
        sizes = [32, 64, 128, 256]
        
        results = {
            "sizes": sizes,
            "standard": {},
            "phi_harmonic": {},
            "speedup": {}
        }
        
        for size in sizes:
            print(f"  Size: {size}x{size}")
            
            # Set dimensions
            batch_size = 8
            channels = min(64, size)
            kernel_size = 3
            
            # Create test tensors
            input_tensor = torch.randn(batch_size, channels, size, size, device=self.device)
            weight = torch.randn(channels, channels, kernel_size, kernel_size, device=self.device)
            
            # Warmup
            _ = F.conv2d(input_tensor, weight, padding=1)
            _ = self.accelerator.conv2d(input_tensor, weight, padding=1)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            
            # Benchmark standard convolution
            standard_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = F.conv2d(input_tensor, weight, padding=1)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                standard_times.append(time.time() - start)
            
            # Benchmark phi-harmonic convolution
            phi_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.conv2d(input_tensor, weight, padding=1)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                phi_times.append(time.time() - start)
            
            # Calculate statistics
            avg_standard = sum(standard_times) / len(standard_times)
            avg_phi = sum(phi_times) / len(phi_times)
            speedup = avg_standard / avg_phi if avg_phi > 0 else 0
            
            # Store results
            results["standard"][size] = avg_standard
            results["phi_harmonic"][size] = avg_phi
            results["speedup"][size] = speedup
            
            print(f"    Standard:     {avg_standard:.6f} sec")
            print(f"    Phi-harmonic: {avg_phi:.6f} sec")
            print(f"    Speedup:      {speedup:.2f}x")
        
        self.results["conv2d"] = results
    
    def benchmark_consciousness_states(self):
        """Benchmark different consciousness states."""
        print("\nBenchmarking consciousness states...")
        
        results = {
            "states": CONSCIOUSNESS_STATES,
            "matmul": {},
            "attention": {},
            "conv2d": {}
        }
        
        # Fixed sizes for comparison
        matmul_size = 1024
        attention_size = 256
        conv_size = 128
        
        # Create test tensors for matmul
        a_matmul = torch.randn(matmul_size, matmul_size, device=self.device)
        b_matmul = torch.randn(matmul_size, matmul_size, device=self.device)
        
        # Create test tensors for attention
        batch_size = 16
        seq_len = attention_size
        d_model = 512
        query = torch.randn(batch_size, seq_len, d_model, device=self.device)
        key = torch.randn(batch_size, seq_len, d_model, device=self.device)
        value = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        # Create test tensors for convolution
        channels = 64
        kernel_size = 3
        input_tensor = torch.randn(batch_size, channels, conv_size, conv_size, device=self.device)
        weight = torch.randn(channels, channels, kernel_size, kernel_size, device=self.device)
        
        # Benchmark each consciousness state
        for state in CONSCIOUSNESS_STATES:
            print(f"  State: {state}")
            self.accelerator.set_consciousness_state(state)
            
            # Warmup
            _ = self.accelerator.matmul(a_matmul, b_matmul)
            _ = self.accelerator.attention(query, key, value)
            _ = self.accelerator.conv2d(input_tensor, weight, padding=1)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            
            # Benchmark matmul with this state
            matmul_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.matmul(a_matmul, b_matmul)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                matmul_times.append(time.time() - start)
            
            # Benchmark attention with this state
            attention_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.attention(query, key, value)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                attention_times.append(time.time() - start)
            
            # Benchmark conv2d with this state
            conv_times = []
            for _ in range(BENCHMARK_REPEATS):
                start = time.time()
                _ = self.accelerator.conv2d(input_tensor, weight, padding=1)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                conv_times.append(time.time() - start)
            
            # Calculate statistics
            avg_matmul = sum(matmul_times) / len(matmul_times)
            avg_attention = sum(attention_times) / len(attention_times)
            avg_conv = sum(conv_times) / len(conv_times)
            
            # Store results
            results["matmul"][state] = avg_matmul
            results["attention"][state] = avg_attention
            results["conv2d"][state] = avg_conv
            
            print(f"    Matmul:    {avg_matmul:.6f} sec")
            print(f"    Attention: {avg_attention:.6f} sec")
            print(f"    Conv2D:    {avg_conv:.6f} sec")
        
        self.results["consciousness_states"] = results
    
    def save_results(self):
        """Save benchmark results to file."""
        # Add metadata
        self.results["metadata"] = {
            "device": str(self.device),
            "timestamp": self.timestamp,
            "device_info": self.accelerator.get_device_info()
        }
        
        # Save to JSON file
        filename = f"a5500_phi_benchmark_results_{self.timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            # Convert non-serializable types (like tensor sizes) to strings
            json_results = json.dumps(self.results, default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, list, dict, type(None))) else o)
            f.write(json_results)
        
        print(f"Results saved to {filepath}")
        
        # Also save timing stats from the accelerator
        timing_stats = self.accelerator.get_timing_stats()
        timing_filename = f"a5500_phi_timing_stats_{self.timestamp}.json"
        timing_filepath = os.path.join(self.output_dir, timing_filename)
        
        with open(timing_filepath, 'w') as f:
            json.dump(timing_stats, f, indent=2, default=str)
        
        print(f"Timing stats saved to {timing_filepath}")
    
    def generate_plots(self):
        """Generate plots from benchmark results."""
        # Plot matrix multiplication benchmark
        self._plot_operation_benchmark("matmul", "Matrix Multiplication")
        
        # Plot attention benchmark
        self._plot_operation_benchmark("attention", "Attention Mechanism")
        
        # Plot convolution benchmark
        self._plot_operation_benchmark("conv2d", "2D Convolution")
        
        # Plot consciousness states comparison
        self._plot_consciousness_states()
    
    def _plot_operation_benchmark(self, operation, title):
        """Plot benchmark results for a specific operation."""
        if operation not in self.results:
            print(f"No results for {operation}")
            return
        
        results = self.results[operation]
        sizes = results["sizes"]
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot execution times
        ax1.plot(sizes, [results["standard"][size] for size in sizes], 'o-', label='Standard')
        ax1.plot(sizes, [results["phi_harmonic"][size] for size in sizes], 's-', label='Phi-harmonic')
        ax1.set_xlabel('Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'{title} Execution Time')
        ax1.set_xscale('log2')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot speedup
        ax2.plot(sizes, [results["speedup"][size] for size in sizes], 'D-', color='green')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Size')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title(f'{title} Speedup (Phi-harmonic vs Standard)')
        ax2.set_xscale('log2')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations for speedup values
        for i, size in enumerate(sizes):
            speedup = results["speedup"][size]
            ax2.annotate(f'{speedup:.2f}x', 
                         xy=(size, speedup),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center',
                         fontsize=9)
        
        plt.tight_layout()
        filename = f"a5500_phi_{operation}_benchmark_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=120)
        print(f"Plot saved to {filepath}")
        plt.close()
    
    def _plot_consciousness_states(self):
        """Plot benchmark results for different consciousness states."""
        if "consciousness_states" not in self.results:
            print("No consciousness state benchmark results")
            return
        
        results = self.results["consciousness_states"]
        states = results["states"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.25
        
        # Set positions of bars on X axis
        r1 = np.arange(len(states))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, [results["matmul"][state] for state in states], width=bar_width, label='Matrix Multiplication', color='blue', alpha=0.7)
        ax.bar(r2, [results["attention"][state] for state in states], width=bar_width, label='Attention', color='green', alpha=0.7)
        ax.bar(r3, [results["conv2d"][state] for state in states], width=bar_width, label='Convolution', color='orange', alpha=0.7)
        
        # Add labels and title
        ax.set_xlabel('Consciousness State')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Effect of Consciousness States on Operation Performance')
        ax.set_xticks([r + bar_width for r in range(len(states))])
        ax.set_xticklabels(states)
        ax.legend()
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value annotations
        for i, state in enumerate(states):
            for j, op in enumerate(["matmul", "attention", "conv2d"]):
                value = results[op][state]
                r = r1[i] if j == 0 else r2[i] if j == 1 else r3[i]
                ax.annotate(f'{value:.4f}', 
                            xy=(r, value),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center',
                            fontsize=8)
        
        plt.tight_layout()
        filename = f"a5500_phi_consciousness_states_{self.timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=120)
        print(f"Plot saved to {filepath}")
        plt.close()


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="A5500 Phi-Harmonic GPU Benchmark")
    parser.add_argument("--device", type=str, default=None, 
                      help="Device to use (cuda, cpu, or specific GPU like cuda:0)")
    parser.add_argument("--quick", action="store_true",
                      help="Run a quick version of the benchmark with fewer sizes")
    
    args = parser.parse_args()
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Modify global sizes if quick mode
    if args.quick:
        global SIZES, BENCHMARK_REPEATS
        SIZES = [128, 512, 1024]
        BENCHMARK_REPEATS = 3
        print("Running in quick mode with fewer sizes and iterations")
    
    # Run benchmark
    benchmark = PhiBenchmark(device=device)
    benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()