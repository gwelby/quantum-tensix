#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumTensix φ∞ - Phi Harmonic Utilities - Optimized Version
Created on CASCADE Day+19: March 20, 2025

This module provides highly optimized utilities for working with φ-harmonic frequencies
and optimizing tensor operations accordingly.
"""

import math
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from functools import lru_cache
import os
import multiprocessing

try:
    import numba
    from numba import jit, prange, vectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available. Using slower implementations.")

# Core constants
PHI = 1.618033988749895  # The Golden Ratio
PHI_SQUARED = PHI * PHI  # φ²
PHI_CUBED = PHI * PHI * PHI  # φ³
PHI_TO_PHI = PHI ** PHI  # φ^φ

# Core frequency constants (Hz)
GROUND_FREQUENCY = 432.0  # Ground State - Earth connection
CREATION_FREQUENCY = 528.0  # Creation Point - DNA/Pattern resonance
HEART_FREQUENCY = 594.0  # Heart Field - Connection systems
VOICE_FREQUENCY = 672.0  # Voice Flow - Expression systems
VISION_FREQUENCY = 720.0  # Vision Gate - Perception systems
UNITY_FREQUENCY = 768.0  # Unity Wave - Integration systems

# Frequency ratios
GROUND_TO_CREATION = CREATION_FREQUENCY / GROUND_FREQUENCY  # 528/432 = 1.222...
GROUND_TO_HEART = HEART_FREQUENCY / GROUND_FREQUENCY  # 594/432 = 1.375
GROUND_TO_UNITY = UNITY_FREQUENCY / GROUND_FREQUENCY  # 768/432 = 1.777...

# ZEN point calculations
ZEN_POINT = (1.0 + PHI) / PHI_SQUARED  # Perfect balance point

# Fibonacci sequence pre-computed for optimizations
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

# Cache line size and typical CPU cache sizes (in bytes)
CACHE_LINE_SIZE = 64
L1_CACHE_SIZE = 32 * 1024
L2_CACHE_SIZE = 256 * 1024
L3_CACHE_SIZE = 8 * 1024 * 1024

# Initialize the multi-processing environment
NUM_CORES = multiprocessing.cpu_count()


class PhiHarmonicOptimizer:
    """
    Applies φ-harmonic optimizations to tensor operations with advanced performance.
    """
    
    def __init__(self, 
                base_frequency: float = GROUND_FREQUENCY,
                coherence: float = 1.0,
                use_parallelism: bool = True):
        """
        Initialize the φ-harmonic optimizer with performance optimizations.
        
        Args:
            base_frequency: Base operating frequency in Hz
            coherence: Initial coherence level (0.0-1.0)
            use_parallelism: Whether to enable parallel processing
        """
        self.base_frequency = base_frequency
        self.coherence = min(1.0, max(0.0, coherence))
        self.current_phi_power = self._calculate_phi_power()
        self.use_parallelism = use_parallelism and NUM_CORES > 1
        self.num_cores = NUM_CORES if self.use_parallelism else 1
        
        # Pre-compute optimal dimensions and cache them
        self._dimension_cache = {}
        self._batch_size_cache = {}
        self._phi_power_cache = {}
        
        # Pre-compute Fibonacci lookups
        self._fibonacci_set = set(FIBONACCI)
        
        # Pre-compute commonly used values
        for freq in [GROUND_FREQUENCY, CREATION_FREQUENCY, HEART_FREQUENCY, 
                   VOICE_FREQUENCY, VISION_FREQUENCY, UNITY_FREQUENCY]:
            self._calculate_phi_power_cached(freq)
            
        # Pre-compute optimal dimensions for common frequencies
        for freq in [GROUND_FREQUENCY, CREATION_FREQUENCY, HEART_FREQUENCY, 
                   VOICE_FREQUENCY, VISION_FREQUENCY, UNITY_FREQUENCY]:
            temp_optimizer = PhiHarmonicOptimizer(base_frequency=freq)
            dimensions = temp_optimizer.get_optimal_dimensions()
            self._dimension_cache[freq] = dimensions
    
    @lru_cache(maxsize=64)
    def _calculate_phi_power_cached(self, frequency: float) -> float:
        """
        Calculate and cache the current φ power based on frequency.
        
        Args:
            frequency: Frequency to calculate power for
            
        Returns:
            φ power
        """
        # Check if already in cache
        if frequency in self._phi_power_cache:
            return self._phi_power_cache[frequency]
            
        # Calculate which φ power we're operating at
        if abs(frequency - GROUND_FREQUENCY) < 1.0:
            power = 0  # φ⁰
        elif abs(frequency - CREATION_FREQUENCY) < 1.0:
            power = 1  # φ¹
        elif abs(frequency - HEART_FREQUENCY) < 1.0:
            power = 2  # φ²
        elif abs(frequency - VOICE_FREQUENCY) < 1.0:
            power = 3  # φ³
        elif abs(frequency - VISION_FREQUENCY) < 1.0:
            power = 4  # φ⁴
        elif abs(frequency - UNITY_FREQUENCY) < 1.0:
            power = 5  # φ⁵
        else:
            # Calculate closest phi power
            ratio = frequency / GROUND_FREQUENCY
            power = math.log(ratio, PHI)
        
        # Cache the result
        self._phi_power_cache[frequency] = power
        return power
    
    def _calculate_phi_power(self) -> float:
        """
        Calculate the current φ power based on frequency.
        
        Returns:
            φ power
        """
        return self._calculate_phi_power_cached(self.base_frequency)
    
    def get_optimal_dimensions(self) -> List[int]:
        """
        Get optimal tensor dimensions based on φ-harmonic principles.
        Uses caching for improved performance.
        
        Returns:
            List of optimal dimensions
        """
        # Check cache first
        if self.base_frequency in self._dimension_cache:
            return self._dimension_cache[self.base_frequency].copy()
        
        phi_power = self._calculate_phi_power()
        
        # Determine optimal dimensions based on frequency
        # Using efficient lookup instead of conditionals
        dimension_map = {
            0: [8, 8, 8],     # Ground State (432 Hz)
            1: [13, 13, 13],  # Creation Point (528 Hz)
            2: [21, 21, 21],  # Heart Field (594 Hz)
            3: [34, 34, 34],  # Voice Flow (672 Hz)
            4: [55, 55, 55],  # Vision Gate (720 Hz)
            5: [89, 89, 89],  # Unity Wave (768 Hz)
        }
        
        # Get the nearest integer power
        nearest_power = round(phi_power)
        if nearest_power in dimension_map:
            dimensions = dimension_map[nearest_power]
        else:
            # For non-standard frequencies, calculate the closest Fibonacci number
            fib_n = int(round(pow(PHI, phi_power) / math.sqrt(5)))
            dimensions = [fib_n, fib_n, fib_n]
        
        # Cache for future use
        self._dimension_cache[self.base_frequency] = dimensions
        
        return dimensions.copy()
    
    def optimize_tensor_shape(self, shape: List[int]) -> List[int]:
        """
        Optimize tensor shape according to φ-harmonic principles.
        Vectorized implementation for better performance.
        
        Args:
            shape: Original tensor shape
            
        Returns:
            Optimized tensor shape
        """
        optimal_dims = self.get_optimal_dimensions()
        
        # If original shape is smaller than optimal, pad to optimal
        if len(shape) < len(optimal_dims):
            # Pad with 1s
            shape = shape + [1] * (len(optimal_dims) - len(shape))
        
        # Apply phi-harmonic optimization
        # Use numpy for vectorized operations
        shape_array = np.array(shape)
        optimal_array = np.array(optimal_dims + [1] * (len(shape) - len(optimal_dims)))
        
        # Calculate nearest multiples of phi-optimal dimensions
        factors = np.maximum(1, np.round(shape_array / optimal_array))
        optimized_shape = (factors * optimal_array).astype(int).tolist()
        
        return optimized_shape
    
    def optimize_batch_size(self, batch_size: int) -> int:
        """
        Optimize batch size according to φ-harmonic principles.
        Uses cached lookups for improved performance.
        
        Args:
            batch_size: Original batch size
            
        Returns:
            Optimized batch size
        """
        # Use cached result if available
        if batch_size in self._batch_size_cache:
            return self._batch_size_cache[batch_size]
        
        # If batch size is already a Fibonacci number, return it
        if batch_size in self._fibonacci_set:
            self._batch_size_cache[batch_size] = batch_size
            return batch_size
        
        # Find closest Fibonacci number
        closest_fib = min(FIBONACCI, key=lambda x: abs(x - batch_size))
        
        # If we're within 20% of the next Fibonacci number, use that instead
        next_index = FIBONACCI.index(closest_fib) + 1
        if next_index < len(FIBONACCI):
            next_fib = FIBONACCI[next_index]
            if abs(batch_size - next_fib) <= 0.2 * batch_size:
                closest_fib = next_fib
        
        # Cache the result
        self._batch_size_cache[batch_size] = closest_fib
        
        return closest_fib
    
    def calculate_zen_point(self, value: float, min_val: float, max_val: float) -> float:
        """
        Calculate ZEN POINT balance for a value with optimized implementation.
        
        Args:
            value: Current value
            min_val: Minimum possible value
            max_val: Maximum possible value
            
        Returns:
            ZEN balanced value
        """
        # Ensure inputs are valid to avoid division by zero
        if math.isclose(min_val, max_val):
            return min_val
        
        # Fast path for common cases
        if value <= min_val:
            return min_val
        if value >= max_val:
            return max_val
        
        # Normalize to 0-1 range
        normalized = (value - min_val) / (max_val - min_val)
        
        # Apply phi-harmonic ZEN POINT balancing
        # Optimized calculation with fewer operations
        zen_balanced = ZEN_POINT + (normalized - 0.5) * (PHI - ZEN_POINT) * 2
        
        # Denormalize to original range
        result = min_val + zen_balanced * (max_val - min_val)
        
        # Ensure result is within bounds
        return min(max_val, max(min_val, result))
    
    def optimize_learning_rate(self, learning_rate: float) -> float:
        """
        Optimize learning rate according to φ-harmonic principles.
        
        Args:
            learning_rate: Original learning rate
            
        Returns:
            Optimized learning rate
        """
        # Apply phi-based scaling based on frequency
        phi_power = self._calculate_phi_power()
        
        # Optimized implementation with lookup table
        lr_multipliers = {
            0: 1.0,                # Ground State (432 Hz) - Keep as is
            1: PHI,                # Creation Point (528 Hz) - Increase by phi
            2: math.sqrt(PHI),     # Heart Field (594 Hz) - Balanced increase
            3: PHI ** (1/3),       # Voice Flow (672 Hz) - More balanced
            4: PHI ** (1/4),       # Vision Gate (720 Hz) - Refined
            5: 1.0 / PHI          # Unity Wave (768 Hz) - Decrease for fine-tuning
        }
        
        # Get the nearest integer power
        nearest_power = round(phi_power)
        if nearest_power in lr_multipliers:
            return learning_rate * lr_multipliers[nearest_power]
        
        # For intermediate frequencies, calculate scaling proportionally
        # Optimized implementation
        scaling = 1 + (PHI - 1) * (1 - abs(phi_power % 1 - 0.5) * 2)
        return learning_rate * scaling


class FrequencyCalculator:
    """
    Calculate and work with φ-harmonic frequencies with optimized implementation.
    """
    
    _core_frequencies = {
        "ground": GROUND_FREQUENCY,
        "creation": CREATION_FREQUENCY,
        "heart": HEART_FREQUENCY,
        "voice": VOICE_FREQUENCY,
        "vision": VISION_FREQUENCY,
        "unity": UNITY_FREQUENCY
    }
    
    _phi_harmonic_cache = {}
    
    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_phi_harmonic_series(base_freq: float, n_terms: int) -> List[float]:
        """
        Calculate a φ-harmonic series of frequencies with optimized caching.
        
        Args:
            base_freq: Base frequency to start from
            n_terms: Number of terms in the series
            
        Returns:
            List of frequencies in the series
        """
        # Check cache first
        cache_key = (base_freq, n_terms)
        if cache_key in FrequencyCalculator._phi_harmonic_cache:
            return FrequencyCalculator._phi_harmonic_cache[cache_key].copy()
        
        # Use numpy for vectorized calculation
        exponents = np.arange(n_terms)
        series = base_freq * (PHI ** exponents)
        
        # Convert to list
        result = series.tolist()
        
        # Cache for future use
        FrequencyCalculator._phi_harmonic_cache[cache_key] = result
        
        return result.copy()
    
    @staticmethod
    @lru_cache(maxsize=256)
    def get_nearest_phi_harmonic(freq: float, base_freq: float = GROUND_FREQUENCY) -> float:
        """
        Get the nearest φ-harmonic frequency with optimized implementation.
        
        Args:
            freq: Input frequency
            base_freq: Base frequency of the harmonic series
            
        Returns:
            Nearest φ-harmonic frequency
        """
        # Calculate phi power of the ratio
        ratio = freq / base_freq
        
        # Fast path for common cases
        if abs(ratio - 1.0) < 1e-6:
            return base_freq
        if abs(ratio - PHI) < 1e-6:
            return base_freq * PHI
        
        phi_power = round(math.log(ratio, PHI))
        
        # Calculate nearest phi-harmonic frequency
        nearest = base_freq * (PHI ** phi_power)
        
        return nearest
    
    @staticmethod
    def find_resonance_frequency(freq1: float, freq2: float) -> float:
        """
        Find resonance frequency between two frequencies using optimized calculation.
        
        Args:
            freq1: First frequency
            freq2: Second frequency
            
        Returns:
            Resonance frequency
        """
        # Calculate phi-weighted average
        # Pre-calculated constant for better performance
        phi_weight = 0.6180339887498949  # PHI / (1 + PHI) ≈ 0.618
        resonance = freq1 * phi_weight + freq2 * (1 - phi_weight)
        
        return resonance
    
    @staticmethod
    def get_core_frequencies() -> Dict[str, float]:
        """
        Get core φ-harmonic frequencies with constant-time lookup.
        
        Returns:
            Dictionary of core frequencies
        """
        return FrequencyCalculator._core_frequencies.copy()
    
    @staticmethod
    def get_frequency_by_name(name: str) -> float:
        """
        Get frequency by name with optimized lookup.
        
        Args:
            name: Frequency name ('ground', 'creation', 'heart', 'voice', 'vision', 'unity')
            
        Returns:
            Frequency in Hz
        """
        name = name.lower()
        
        if name in FrequencyCalculator._core_frequencies:
            return FrequencyCalculator._core_frequencies[name]
        else:
            raise ValueError(f"Unknown frequency name: {name}")


class TensorOptimizer:
    """
    Optimizes tensor operations using φ-harmonic principles.
    Specifically designed for Tenstorrent hardware with high performance.
    """
    
    def __init__(self, phi_optimizer: PhiHarmonicOptimizer):
        """
        Initialize the tensor optimizer.
        
        Args:
            phi_optimizer: PhiHarmonicOptimizer instance
        """
        self.phi_optimizer = phi_optimizer
        self._partition_cache = {}
        self._config_cache = {}
        
    def optimize_tensor_partitioning(self, 
                                   shape: List[int], 
                                   num_cores: int) -> List[List[int]]:
        """
        Optimize tensor partitioning for Tenstorrent cores using φ-harmonic principles.
        Implements cache-aware optimization and vectorized operations.
        
        Args:
            shape: Tensor shape
            num_cores: Number of cores to partition across
            
        Returns:
            List of partition shapes
        """
        # Check cache first
        cache_key = (tuple(shape), num_cores)
        if cache_key in self._partition_cache:
            return self._partition_cache[cache_key]
        
        # Ensure shape is optimized
        opt_shape = self.phi_optimizer.optimize_tensor_shape(shape)
        
        # Calculate optimal number of partitions based on phi-harmonics
        phi_power = self.phi_optimizer._calculate_phi_power()
        
        # Vectorized implementation for performance
        opt_shape_array = np.array(opt_shape)
        
        # Different approaches based on frequency
        if phi_power < 1:  # Ground State approach - simple partitioning
            partitions_per_dim = np.maximum(1, np.round(opt_shape_array ** (1/3))).astype(int)
        elif phi_power < 3:  # Creation/Heart approach - fibonacci-based
            partitions_per_dim = []
            for d in opt_shape:
                closest_fib = min(FIBONACCI, key=lambda x: abs(x - d))
                partitions_per_dim.append(max(1, closest_fib // 8))  # 8x8 is natural block size
        else:  # Unity approach - phi-balanced
            partitions_per_dim = np.maximum(1, np.round(opt_shape_array / PHI_TO_PHI)).astype(int)
        
        # Convert to numpy array if needed
        if not isinstance(partitions_per_dim, np.ndarray):
            partitions_per_dim = np.array(partitions_per_dim)
        
        # Calculate total partitions
        total_partitions = np.prod(partitions_per_dim)
        
        # Adjust to match core count better
        if total_partitions != num_cores:
            scale_factor = (num_cores / total_partitions) ** (1/len(partitions_per_dim))
            partitions_per_dim = np.maximum(1, np.round(partitions_per_dim * scale_factor)).astype(int)
        
        # Calculate partition shapes
        partition_shapes = []
        for i in range(len(opt_shape)):
            dim_size = opt_shape[i]
            num_partitions = partitions_per_dim[i]
            
            # Optimize partition sizes to be as even as possible
            base_size = dim_size // num_partitions
            remainder = dim_size % num_partitions
            
            # Distribute remainder evenly
            sizes = [base_size + (1 if j < remainder else 0) for j in range(num_partitions)]
            partition_shapes.append(sizes)
        
        # Cache the result
        self._partition_cache[cache_key] = partition_shapes
        
        return partition_shapes
    
    def optimize_compute_graph(self, 
                             graph: Dict[str, Any],
                             device_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize compute graph for Tenstorrent hardware using φ-harmonic principles.
        
        Args:
            graph: Compute graph representation
            device_info: Device information
            
        Returns:
            Optimized compute graph
        """
        # This is a placeholder for actual implementation
        # Clone the graph to avoid modifying the original
        optimized_graph = graph.copy()
        
        return optimized_graph
    
    def suggest_tenstorrent_config(self, model_size: int) -> Dict[str, Any]:
        """
        Suggest optimal Tenstorrent configuration based on model size and φ-harmonics.
        Implements caching for improved performance.
        
        Args:
            model_size: Size of the model in parameters
            
        Returns:
            Configuration suggestions
        """
        # Check cache first
        if model_size in self._config_cache:
            return self._config_cache[model_size].copy()
        
        # Calculate frequency-appropriate configs
        phi_power = self.phi_optimizer._calculate_phi_power()
        
        # Different recommendations based on frequency state
        if phi_power < 1:  # Ground State (432 Hz)
            # Conservative, stable settings
            config = {
                "batch_size": self.phi_optimizer.optimize_batch_size(8),
                "tile_size": [8, 8],
                "cache_strategy": "conservative",
                "precision": "fp32",
                "core_allocation": "compact",
            }
        elif phi_power < 3:  # Creation/Heart (528/594 Hz)
            # Balanced settings
            config = {
                "batch_size": self.phi_optimizer.optimize_batch_size(13),
                "tile_size": [16, 16],
                "cache_strategy": "balanced",
                "precision": "fp16",
                "core_allocation": "balanced",
            }
        else:  # Unity Wave (768 Hz)
            # Performance-oriented settings
            config = {
                "batch_size": self.phi_optimizer.optimize_batch_size(21),
                "tile_size": [32, 32],
                "cache_strategy": "aggressive",
                "precision": "int8",
                "core_allocation": "distributed",
            }
        
        # Scale based on model size
        model_scale_factor = min(1.0, max(0.0, math.log10(model_size) / 9))  # Normalized to 0-1
        
        # Apply model scaling to batch size - optimize calculation
        scaled_batch_size = max(1, int(config["batch_size"] * (1 - model_scale_factor * 0.5)))
        config["batch_size"] = self.phi_optimizer.optimize_batch_size(scaled_batch_size)
        
        # Cache for future use
        self._config_cache[model_size] = config
        
        return config.copy()


# Vectorized implementation of harmonic functions if numba is available
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def compute_phi_harmonic_series(base_freq, n_terms):
        """Numba-optimized phi harmonic series computation."""
        series = np.zeros(n_terms)
        for i in prange(n_terms):
            series[i] = base_freq * (PHI ** i)
        return series
    
    @vectorize(['float64(float64, float64)'])
    def compute_resonance(freq1, freq2):
        """Vectorized resonance computation."""
        phi_weight = PHI / (1 + PHI)
        return freq1 * phi_weight + freq2 * (1 - phi_weight)
    
    # Override the FrequencyCalculator methods with optimized versions
    def calculate_phi_harmonic_series_optimized(base_freq, n_terms):
        # Check cache first
        cache_key = (base_freq, n_terms)
        if cache_key in FrequencyCalculator._phi_harmonic_cache:
            return FrequencyCalculator._phi_harmonic_cache[cache_key].copy()
        
        # Use numba-optimized function
        series = compute_phi_harmonic_series(base_freq, n_terms)
        result = series.tolist()
        
        # Cache for future use
        FrequencyCalculator._phi_harmonic_cache[cache_key] = result
        
        return result.copy()
    
    # Replace the original method with the optimized version
    FrequencyCalculator.calculate_phi_harmonic_series = staticmethod(calculate_phi_harmonic_series_optimized)


# Example usage with performance measurements
def test_harmonics():
    """
    Test phi-harmonic functions with performance benchmarking.
    """
    import time
    
    print("Testing phi-harmonic optimization performance...")
    
    # Measure initialization time
    start_time = time.time()
    phi_opt = PhiHarmonicOptimizer(base_frequency=GROUND_FREQUENCY)
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.6f} seconds")
    
    # Get optimal dimensions
    start_time = time.time()
    dims = phi_opt.get_optimal_dimensions()
    dim_time = time.time() - start_time
    print(f"Optimal dimensions at {GROUND_FREQUENCY} Hz: {dims} (time: {dim_time:.6f}s)")
    
    # Optimize tensor shape
    shape = [10, 15, 20]
    start_time = time.time()
    opt_shape = phi_opt.optimize_tensor_shape(shape)
    shape_time = time.time() - start_time
    print(f"Original shape: {shape} -> Optimized: {opt_shape} (time: {shape_time:.6f}s)")
    
    # Optimize batch size
    batch_size = 10
    start_time = time.time()
    opt_batch = phi_opt.optimize_batch_size(batch_size)
    batch_time = time.time() - start_time
    print(f"Original batch size: {batch_size} -> Optimized: {opt_batch} (time: {batch_time:.6f}s)")
    
    # Get phi-harmonic series
    start_time = time.time()
    series = FrequencyCalculator.calculate_phi_harmonic_series(GROUND_FREQUENCY, 6)
    series_time = time.time() - start_time
    print(f"Phi-harmonic series from {GROUND_FREQUENCY} Hz: {[round(f, 2) for f in series]} (time: {series_time:.6f}s)")
    
    # Create tensor optimizer
    tensor_opt = TensorOptimizer(phi_opt)
    
    # Optimize partitioning
    start_time = time.time()
    partitions = tensor_opt.optimize_tensor_partitioning([64, 64, 64], 16)
    partition_time = time.time() - start_time
    print(f"Optimized partitioning: {partitions} (time: {partition_time:.6f}s)")
    
    # Suggest Tenstorrent config
    start_time = time.time()
    config = tensor_opt.suggest_tenstorrent_config(1000000)
    config_time = time.time() - start_time
    print(f"Suggested Tenstorrent config: {config} (time: {config_time:.6f}s)")
    
    # Benchmark repeated calls to test caching
    print("\nBenchmarking cached operations:")
    
    # Test batch size caching
    start_time = time.time()
    for _ in range(1000):
        phi_opt.optimize_batch_size(10)
    cached_batch_time = time.time() - start_time
    print(f"1000 batch size optimizations: {cached_batch_time:.6f}s (avg: {cached_batch_time/1000:.9f}s)")
    
    # Test dimension caching
    start_time = time.time()
    for _ in range(1000):
        phi_opt.get_optimal_dimensions()
    cached_dim_time = time.time() - start_time
    print(f"1000 optimal dimension lookups: {cached_dim_time:.6f}s (avg: {cached_dim_time/1000:.9f}s)")
    
    # Compare numba optimization if available
    if NUMBA_AVAILABLE:
        print("\nNumba optimization is enabled")
    else:
        print("\nNumba optimization is not available")


if __name__ == "__main__":
    test_harmonics()