#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensional Navigation Benchmark - QuantumTensix φ∞
Created on CASCADE Day+27: March 28, 2025

This module benchmarks the Dimensional Navigator's capabilities,
measuring performance improvements from higher-dimensional access.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import QuantumTensix components
from quantum_tensix import (
    GROUND_FREQUENCY, CREATION_FREQUENCY, HEART_FREQUENCY, VOICE_FREQUENCY,
    VISION_FREQUENCY, UNITY_FREQUENCY
)

# Import Quantum Consciousness Bridge
from quantum_consciousness_bridge import (
    ConsciousnessState, ConsciousnessField, QuantumConsciousnessBridge, SACRED_FREQUENCIES
)

# Import Dimensional Navigator
from dimensional_navigator import (
    DimensionalNavigator, DimensionalAccessState, DIMENSIONS
)

# Import utilities
from utils.phi_harmonics import PHI, PHI_SQUARED, PHI_TO_PHI

# Constants
BENCHMARK_ITERATIONS = 10
MATRIX_SIZES = [64, 144, 233, 377, 610]
BATCH_SIZES = [1, 2, 3, 5, 8, 13, 21]


class DimensionalNavigationBenchmark:
    """
    Benchmarks for the Dimensional Navigator system, measuring performance
    improvements from higher-dimensional access during tensor operations.
    """
    
    def __init__(self):
        """Initialize dimensional navigation benchmark system"""
        # Create output directory
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Initialize Quantum Consciousness Bridge
        self.bridge = QuantumConsciousnessBridge()
        
        # Initialize Dimensional Navigator
        self.navigator = DimensionalNavigator(self.bridge)
        
        # Store benchmark results
        self.results = {}
        self.timestamp = int(time.time())
        
        logging.info("Dimensional Navigation Benchmark initialized")
    
    def _run_matrix_operation(self, 
                            matrix_size: int, 
                            batch_size: int,
                            dimension: str) -> Dict[str, float]:
        """
        Run matrix multiplication and measure performance
        
        Args:
            matrix_size: Size of the square matrices
            batch_size: Batch size
            dimension: Dimension to run in
            
        Returns:
            Performance metrics
        """
        # Navigate to dimension
        if self.navigator.current_dimension != dimension:
            self.navigator.navigate_to_dimension(dimension)
        
        # Create input matrices
        A = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
        B = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
        
        # Apply dimensional optimization
        A_opt = self.navigator.optimize_tensor_with_dimension(A)
        B_opt = self.navigator.optimize_tensor_with_dimension(B)
        
        # Warm-up run
        _ = torch.matmul(A_opt, B_opt)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        
        # Run baseline operation (without dimension optimization)
        start_time = time.time()
        for _ in range(BENCHMARK_ITERATIONS):
            C_baseline = torch.matmul(A, B)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
        baseline_time = (time.time() - start_time) / BENCHMARK_ITERATIONS
        
        # Run optimized operation (with dimension optimization)
        start_time = time.time()
        for _ in range(BENCHMARK_ITERATIONS):
            C_optimized = torch.matmul(A_opt, B_opt)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
        optimized_time = (time.time() - start_time) / BENCHMARK_ITERATIONS
        
        # Calculate metrics
        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
        improvement_pct = (speedup - 1.0) * 100
        
        # Return metrics
        return {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "improvement_pct": improvement_pct,
            "coherence": self.navigator.field_coherence,
            "dimension": dimension,
            "dimension_frequency": DIMENSIONS[dimension]['frequency'],
            "dimension_scaling": DIMENSIONS[dimension]['scaling'],
        }
    
    def benchmark_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark each dimension's performance with fixed matrix and batch size
        
        Returns:
            Benchmark results by dimension
        """
        results = {}
        matrix_size = 233  # Fixed size for dimensional comparison
        batch_size = 8
        
        logging.info(f"Running dimensional benchmark with matrix size {matrix_size}x{matrix_size} and batch size {batch_size}")
        
        # Benchmark each dimension
        for dimension in ["3D", "4D", "5D", "6D", "7D", "8D"]:
            logging.info(f"Benchmarking {dimension} dimension")
            
            # Run benchmark multiple times and average results
            dim_results = []
            for i in range(3):  # Run 3 times for reliability
                metrics = self._run_matrix_operation(matrix_size, batch_size, dimension)
                dim_results.append(metrics)
                logging.info(f"  Run {i+1}: Speedup = {metrics['speedup']:.3f}x ({metrics['improvement_pct']:.2f}% improvement)")
            
            # Average the results
            avg_metrics = {
                "baseline_time": sum(r["baseline_time"] for r in dim_results) / len(dim_results),
                "optimized_time": sum(r["optimized_time"] for r in dim_results) / len(dim_results),
                "speedup": sum(r["speedup"] for r in dim_results) / len(dim_results),
                "improvement_pct": sum(r["improvement_pct"] for r in dim_results) / len(dim_results),
                "coherence": sum(r["coherence"] for r in dim_results) / len(dim_results),
                "dimension": dimension,
                "dimension_frequency": DIMENSIONS[dimension]['frequency'],
                "dimension_scaling": DIMENSIONS[dimension]['scaling'],
                "individual_runs": dim_results
            }
            
            results[dimension] = avg_metrics
            logging.info(f"  Average: Speedup = {avg_metrics['speedup']:.3f}x ({avg_metrics['improvement_pct']:.2f}% improvement)")
        
        # Store results
        self.results["dimension_benchmark"] = results
        
        return results
    
    def benchmark_matrix_sizes(self, dimension: str = "5D") -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different matrix sizes in a fixed dimension
        
        Args:
            dimension: Dimension to run in
            
        Returns:
            Benchmark results by matrix size
        """
        results = {}
        batch_size = 8  # Fixed batch size
        
        logging.info(f"Running matrix size benchmark in {dimension} dimension with batch size {batch_size}")
        
        # Navigate to dimension
        if self.navigator.current_dimension != dimension:
            self.navigator.navigate_to_dimension(dimension)
        
        # Benchmark each matrix size
        for matrix_size in MATRIX_SIZES:
            logging.info(f"Benchmarking {matrix_size}x{matrix_size} matrix")
            
            metrics = self._run_matrix_operation(matrix_size, batch_size, dimension)
            results[str(matrix_size)] = metrics
            
            logging.info(f"  Speedup = {metrics['speedup']:.3f}x ({metrics['improvement_pct']:.2f}% improvement)")
        
        # Store results
        self.results["matrix_size_benchmark"] = results
        
        return results
    
    def benchmark_batch_sizes(self, dimension: str = "5D") -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different batch sizes in a fixed dimension
        
        Args:
            dimension: Dimension to run in
            
        Returns:
            Benchmark results by batch size
        """
        results = {}
        matrix_size = 233  # Fixed matrix size
        
        logging.info(f"Running batch size benchmark in {dimension} dimension with matrix size {matrix_size}x{matrix_size}")
        
        # Navigate to dimension
        if self.navigator.current_dimension != dimension:
            self.navigator.navigate_to_dimension(dimension)
        
        # Benchmark each batch size
        for batch_size in BATCH_SIZES:
            logging.info(f"Benchmarking batch size {batch_size}")
            
            metrics = self._run_matrix_operation(matrix_size, batch_size, dimension)
            results[str(batch_size)] = metrics
            
            logging.info(f"  Speedup = {metrics['speedup']:.3f}x ({metrics['improvement_pct']:.2f}% improvement)")
        
        # Store results
        self.results["batch_size_benchmark"] = results
        
        return results
    
    def benchmark_bridging(self) -> Dict[str, Any]:
        """
        Benchmark dimensional bridging performance
        
        Returns:
            Benchmark results for dimensional bridging
        """
        logging.info("Running dimensional bridging benchmark")
        
        matrix_size = 233
        batch_size = 8
        results = {}
        
        # Test pairs of dimensions
        dimension_pairs = [
            ("3D", "5D"),   # Physical to Mental
            ("4D", "7D"),   # Emotional to Cosmic
            ("5D", "8D"),   # Mental to Unity
        ]
        
        for dim1, dim2 in dimension_pairs:
            # First test individual dimensions
            metrics_dim1 = self._run_matrix_operation(matrix_size, batch_size, dim1)
            metrics_dim2 = self._run_matrix_operation(matrix_size, batch_size, dim2)
            
            # Create bridge between dimensions
            bridge_success = self.navigator.create_dimensional_bridge(dim1, dim2)
            
            if bridge_success:
                # Run bridged operation
                
                # Create input matrices
                A = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
                B = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
                
                # Apply dimensional optimization
                A_opt = self.navigator.optimize_tensor_with_dimension(A)
                B_opt = self.navigator.optimize_tensor_with_dimension(B)
                
                # Warm-up run
                _ = torch.matmul(A_opt, B_opt)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                
                # Run bridged operation
                start_time = time.time()
                for _ in range(BENCHMARK_ITERATIONS):
                    C_bridged = torch.matmul(A_opt, B_opt)
                    torch.cuda.synchronize() if self.device.type == "cuda" else None
                bridged_time = (time.time() - start_time) / BENCHMARK_ITERATIONS
                
                # Calculate metrics relative to best individual dimension
                best_individual_time = min(metrics_dim1["optimized_time"], metrics_dim2["optimized_time"])
                bridge_speedup = best_individual_time / bridged_time if bridged_time > 0 else 1.0
                bridge_improvement = (bridge_speedup - 1.0) * 100
                
                # Calculate metrics relative to baseline
                baseline_time = (metrics_dim1["baseline_time"] + metrics_dim2["baseline_time"]) / 2
                total_speedup = baseline_time / bridged_time if bridged_time > 0 else 1.0
                total_improvement = (total_speedup - 1.0) * 100
                
                metrics = {
                    "dimension_pair": f"{dim1}↔{dim2}",
                    "dim1_metrics": metrics_dim1,
                    "dim2_metrics": metrics_dim2,
                    "bridged_time": bridged_time,
                    "bridge_speedup": bridge_speedup,
                    "bridge_improvement": bridge_improvement,
                    "total_speedup": total_speedup,
                    "total_improvement": total_improvement,
                    "coherence": self.navigator.field_coherence,
                    "access_state": self.navigator.access_state.value,
                }
                
                results[f"{dim1}↔{dim2}"] = metrics
                
                logging.info(f"  {dim1}↔{dim2} Bridge: Speedup = {bridge_speedup:.3f}x over best individual dimension")
                logging.info(f"  Total improvement: {total_improvement:.2f}% over baseline")
            else:
                logging.warning(f"Failed to create bridge between {dim1} and {dim2}")
        
        # Benchmark unified field access
        unified_success = self.navigator.access_unified_field()
        
        if unified_success:
            # Run unified operation
            
            # Create input matrices
            A = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
            B = torch.rand(batch_size, matrix_size, matrix_size, device=self.device)
            
            # Apply dimensional optimization
            A_opt = self.navigator.optimize_tensor_with_dimension(A)
            B_opt = self.navigator.optimize_tensor_with_dimension(B)
            
            # Warm-up run
            _ = torch.matmul(A_opt, B_opt)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            
            # Run unified operation
            start_time = time.time()
            for _ in range(BENCHMARK_ITERATIONS):
                C_unified = torch.matmul(A_opt, B_opt)
                torch.cuda.synchronize() if self.device.type == "cuda" else None
            unified_time = (time.time() - start_time) / BENCHMARK_ITERATIONS
            
            # Get best individual dimension time from previous results
            best_individual_time = min(
                results[list(results.keys())[0]]["dim1_metrics"]["optimized_time"],
                results[list(results.keys())[0]]["dim2_metrics"]["optimized_time"],
                results[list(results.keys())[1]]["dim1_metrics"]["optimized_time"],
                results[list(results.keys())[1]]["dim2_metrics"]["optimized_time"]
            )
            
            # Get best bridged time
            best_bridged_time = min(
                results[list(results.keys())[0]]["bridged_time"],
                results[list(results.keys())[1]]["bridged_time"]
            )
            
            # Calculate metrics
            unified_vs_dimension = best_individual_time / unified_time if unified_time > 0 else 1.0
            unified_vs_bridge = best_bridged_time / unified_time if unified_time > 0 else 1.0
            unified_improvement = (unified_vs_dimension - 1.0) * 100
            
            metrics = {
                "unified_time": unified_time,
                "unified_vs_dimension": unified_vs_dimension,
                "unified_vs_bridge": unified_vs_bridge,
                "unified_improvement": unified_improvement,
                "coherence": self.navigator.field_coherence,
                "access_state": self.navigator.access_state.value,
            }
            
            results["unified"] = metrics
            
            logging.info(f"  Unified Field: Speedup = {unified_vs_dimension:.3f}x over best individual dimension")
            logging.info(f"  Unified vs Bridge: Speedup = {unified_vs_bridge:.3f}x over best bridge")
        else:
            logging.warning(f"Failed to access unified field")
        
        # Store results
        self.results["bridging_benchmark"] = results
        
        return results
    
    def benchmark_pattern_translation(self) -> Dict[str, Any]:
        """
        Benchmark pattern translation between dimensions
        
        Returns:
            Benchmark results for pattern translation
        """
        logging.info("Running pattern translation benchmark")
        
        pattern_types = ["fibonacci", "golden_spiral", "earth_grid"]
        pattern_sizes = [[13, 13], [21, 21], [34, 34]]
        source_dimensions = ["3D", "5D", "7D"]
        target_dimensions = ["4D", "6D", "8D"]
        
        results = {}
        
        for pattern_type in pattern_types:
            type_results = {}
            
            for pattern_size in pattern_sizes:
                size_results = {}
                
                for source_dim in source_dimensions:
                    dim_results = {}
                    
                    # Navigate to source dimension
                    self.navigator.navigate_to_dimension(source_dim)
                    
                    # Create pattern
                    logging.info(f"Creating {pattern_type} pattern in {source_dim} with size {pattern_size}")
                    start_time = time.time()
                    pattern = self.navigator.create_dimensional_pattern(pattern_type, pattern_size)
                    creation_time = time.time() - start_time
                    
                    # Assess coherence
                    source_coherence = self.navigator.ground_state.assess_coherence(pattern)
                    
                    for target_dim in target_dimensions:
                        if target_dim == source_dim:
                            continue
                            
                        # Translate pattern
                        logging.info(f"Translating from {source_dim} to {target_dim}")
                        start_time = time.time()
                        translated = self.navigator.translate_pattern(pattern, target_dim)
                        translation_time = time.time() - start_time
                        
                        # Assess translated coherence
                        target_coherence = self.navigator.ground_state.assess_coherence(translated)
                        
                        # Calculate coherence change
                        coherence_change = target_coherence - source_coherence
                        
                        # Calculate dimensional distance
                        dim_distance = abs(
                            DIMENSIONS[target_dim]['scaling'] - 
                            DIMENSIONS[source_dim]['scaling']
                        )
                        
                        dim_results[target_dim] = {
                            "source_dimension": source_dim,
                            "target_dimension": target_dim,
                            "source_coherence": source_coherence,
                            "target_coherence": target_coherence,
                            "coherence_change": coherence_change,
                            "coherence_change_pct": (coherence_change / source_coherence) * 100 if source_coherence > 0 else 0,
                            "translation_time": translation_time,
                            "dimensional_distance": dim_distance,
                        }
                        
                        logging.info(f"  Coherence change: {coherence_change:.3f} ({dim_results[target_dim]['coherence_change_pct']:.2f}%)")
                    
                    size_results[source_dim] = {
                        "source_dimension": source_dim,
                        "pattern_size": pattern_size,
                        "source_coherence": source_coherence,
                        "creation_time": creation_time,
                        "translations": dim_results
                    }
                
                type_results[f"{pattern_size[0]}x{pattern_size[1]}"] = size_results
            
            results[pattern_type] = type_results
        
        # Store results
        self.results["pattern_translation_benchmark"] = results
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run all benchmarks and collect results
        
        Returns:
            Complete benchmark results
        """
        logging.info("Running full dimensional navigation benchmark suite")
        
        # Run dimension benchmark
        self.benchmark_dimensions()
        
        # Run matrix size benchmark
        self.benchmark_matrix_sizes()
        
        # Run batch size benchmark
        self.benchmark_batch_sizes()
        
        # Run bridging benchmark
        self.benchmark_bridging()
        
        # Run pattern translation benchmark
        self.benchmark_pattern_translation()
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Close dimensional access
        self.navigator.close_dimensional_access()
        
        return self.results
    
    def save_results(self) -> None:
        """Save benchmark results to file"""
        # Create results file
        filename = os.path.join(self.results_dir, f"dimensional_navigation_benchmark_{self.timestamp}.json")
        
        # Add metadata
        self.results["metadata"] = {
            "timestamp": self.timestamp,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "benchmark_iterations": BENCHMARK_ITERATIONS,
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"Results saved to {filename}")
    
    def generate_visualizations(self) -> None:
        """Generate visualizations of benchmark results"""
        logging.info("Generating benchmark visualizations")
        
        # Set up matplotlib
        plt.style.use('ggplot')
        
        # Generate dimension benchmark visualization
        self._plot_dimension_benchmark()
        
        # Generate matrix size benchmark visualization
        self._plot_matrix_size_benchmark()
        
        # Generate batch size benchmark visualization
        self._plot_batch_size_benchmark()
        
        # Generate bridging benchmark visualization
        self._plot_bridging_benchmark()
        
        # Generate pattern translation visualization
        self._plot_pattern_translation_benchmark()
    
    def _plot_dimension_benchmark(self) -> None:
        """Plot dimension benchmark results"""
        if "dimension_benchmark" not in self.results:
            return
            
        results = self.results["dimension_benchmark"]
        
        # Extract data
        dimensions = list(results.keys())
        speedups = [results[dim]["speedup"] for dim in dimensions]
        improvements = [results[dim]["improvement_pct"] for dim in dimensions]
        coherences = [results[dim]["coherence"] for dim in dimensions]
        frequencies = [results[dim]["dimension_frequency"] for dim in dimensions]
        
        # Plot speedup by dimension
        plt.figure(figsize=(12, 7))
        
        bar_positions = np.arange(len(dimensions))
        bar_width = 0.35
        
        # Plot speedup bars
        bars1 = plt.bar(bar_positions, speedups, bar_width, 
                       label='Speedup Factor', color='skyblue')
        
        # Plot improvement line
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        line = ax2.plot(bar_positions, improvements, 'ro-', linewidth=2,
                      label='Improvement %')
        
        # Add coherence as text
        for i, v in enumerate(speedups):
            plt.text(i - 0.15, v + 0.05, f"{coherences[i]:.3f}", fontsize=9, 
                    color='black', fontweight='bold')
        
        # Add dimension frequencies as annotations
        for i, freq in enumerate(frequencies):
            plt.annotate(f"{freq} Hz", 
                       (i, 0.1), 
                       xytext=(0, -20),
                       textcoords='offset points',
                       ha='center', va='center',
                       fontsize=9, color='darkblue')
        
        # Configure plot
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Speedup Factor')
        ax2.set_ylabel('Improvement %')
        ax1.set_title('Performance by Dimension')
        ax1.set_xticks(bar_positions)
        ax1.set_xticklabels(dimensions)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"dimension_benchmark_{self.timestamp}.png"))
        plt.close()
    
    def _plot_matrix_size_benchmark(self) -> None:
        """Plot matrix size benchmark results"""
        if "matrix_size_benchmark" not in self.results:
            return
            
        results = self.results["matrix_size_benchmark"]
        
        # Extract data
        sizes = [int(size) for size in results.keys()]
        speedups = [results[str(size)]["speedup"] for size in sizes]
        improvements = [results[str(size)]["improvement_pct"] for size in sizes]
        
        # Plot speedup by matrix size
        plt.figure(figsize=(12, 7))
        
        plt.plot(sizes, speedups, 'bo-', linewidth=2, label='Speedup Factor')
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(sizes, improvements, 'ro-', linewidth=2, label='Improvement %')
        
        # Add speedup values as text
        for i, v in enumerate(speedups):
            plt.text(sizes[i], v + 0.05, f"{v:.2f}x", fontsize=9, 
                    color='blue', fontweight='bold')
        
        # Configure plot
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Speedup Factor')
        ax2.set_ylabel('Improvement %')
        ax1.set_title('Performance by Matrix Size')
        ax1.set_xticks(sizes)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"matrix_size_benchmark_{self.timestamp}.png"))
        plt.close()
    
    def _plot_batch_size_benchmark(self) -> None:
        """Plot batch size benchmark results"""
        if "batch_size_benchmark" not in self.results:
            return
            
        results = self.results["batch_size_benchmark"]
        
        # Extract data
        sizes = [int(size) for size in results.keys()]
        speedups = [results[str(size)]["speedup"] for size in sizes]
        improvements = [results[str(size)]["improvement_pct"] for size in sizes]
        
        # Plot speedup by batch size
        plt.figure(figsize=(12, 7))
        
        plt.plot(sizes, speedups, 'go-', linewidth=2, label='Speedup Factor')
        
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(sizes, improvements, 'ro-', linewidth=2, label='Improvement %')
        
        # Add speedup values as text
        for i, v in enumerate(speedups):
            plt.text(sizes[i], v + 0.05, f"{v:.2f}x", fontsize=9, 
                    color='green', fontweight='bold')
        
        # Configure plot
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Speedup Factor')
        ax2.set_ylabel('Improvement %')
        ax1.set_title('Performance by Batch Size')
        ax1.set_xticks(sizes)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"batch_size_benchmark_{self.timestamp}.png"))
        plt.close()
    
    def _plot_bridging_benchmark(self) -> None:
        """Plot bridging benchmark results"""
        if "bridging_benchmark" not in self.results:
            return
            
        results = self.results["bridging_benchmark"]
        
        # Exclude unified from bridge pairs
        bridge_pairs = [pair for pair in results.keys() if pair != "unified"]
        
        if not bridge_pairs:
            return
            
        # Extract data for bridge pairs
        pair_labels = bridge_pairs
        individual_speedups = [max(
            results[pair]["dim1_metrics"]["speedup"],
            results[pair]["dim2_metrics"]["speedup"]
        ) for pair in bridge_pairs]
        
        bridge_speedups = [results[pair]["total_speedup"] for pair in bridge_pairs]
        
        # Add unified if available
        if "unified" in results:
            pair_labels.append("Unified")
            individual_speedups.append(0)  # Placeholder
            bridge_speedups.append(results["unified"]["unified_vs_dimension"])
        
        # Plot comparison
        plt.figure(figsize=(12, 7))
        
        bar_positions = np.arange(len(pair_labels))
        bar_width = 0.35
        
        # Plot individual and bridge speedups
        bars1 = plt.bar(bar_positions - bar_width/2, individual_speedups, bar_width, 
                       label='Best Individual Dimension', color='skyblue')
        
        bars2 = plt.bar(bar_positions + bar_width/2, bridge_speedups, bar_width,
                       label='Bridged Access', color='coral')
        
        # Add values as text
        for i, v in enumerate(individual_speedups):
            if v > 0:  # Skip unified placeholder
                plt.text(i - bar_width/2, v + 0.05, f"{v:.2f}x", fontsize=9, 
                        color='black', ha='center')
        
        for i, v in enumerate(bridge_speedups):
            plt.text(i + bar_width/2, v + 0.05, f"{v:.2f}x", fontsize=9, 
                    color='black', ha='center')
        
        # Configure plot
        plt.xlabel('Dimension Combinations')
        plt.ylabel('Speedup Factor')
        plt.title('Dimensional Bridging Performance')
        plt.xticks(bar_positions, pair_labels)
        plt.legend()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"bridging_benchmark_{self.timestamp}.png"))
        plt.close()
    
    def _plot_pattern_translation_benchmark(self) -> None:
        """Plot pattern translation benchmark results"""
        if "pattern_translation_benchmark" not in self.results:
            return
            
        results = self.results["pattern_translation_benchmark"]
        
        # Extract coherence changes for golden spiral (most interesting)
        if "golden_spiral" in results:
            pattern_results = results["golden_spiral"]
            
            # Use medium size pattern
            if "21x21" in pattern_results:
                size_results = pattern_results["21x21"]
                
                # Collect data across dimensions
                source_dimensions = []
                coherence_changes = []
                
                for source_dim, dim_data in size_results.items():
                    translations = dim_data["translations"]
                    
                    for target_dim, target_data in translations.items():
                        source_dimensions.append(f"{source_dim}→{target_dim}")
                        coherence_changes.append(target_data["coherence_change_pct"])
                
                # Plot coherence changes
                plt.figure(figsize=(12, 7))
                
                bar_positions = np.arange(len(source_dimensions))
                
                # Color bars based on positive/negative change
                colors = ['green' if change >= 0 else 'red' for change in coherence_changes]
                
                bars = plt.bar(bar_positions, coherence_changes, color=colors)
                
                # Add values as text
                for i, v in enumerate(coherence_changes):
                    plt.text(i, v + 0.5 if v >= 0 else v - 2.5, 
                            f"{v:.1f}%", fontsize=9, 
                            color='black', ha='center')
                
                # Configure plot
                plt.xlabel('Dimension Translation')
                plt.ylabel('Coherence Change %')
                plt.title('Pattern Coherence Changes During Dimensional Translation')
                plt.xticks(bar_positions, source_dimensions, rotation=45)
                
                # Add zero line
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"pattern_translation_benchmark_{self.timestamp}.png"))
                plt.close()


if __name__ == "__main__":
    # Run benchmark
    benchmark = DimensionalNavigationBenchmark()
    benchmark.run_full_benchmark()