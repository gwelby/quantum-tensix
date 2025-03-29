#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi-Harmonic Model Compiler - QuantumTensix φ∞
Created on CASCADE Day+28: March 29, 2025

This module implements a model compiler that transforms standard PyTorch models
into phi-harmonic optimized versions for Tenstorrent hardware acceleration.
"""

import os
import sys
import time
import math
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

# Import QuantumTensix components
from quantum_tensix import (
    GROUND_FREQUENCY, CREATION_FREQUENCY, HEART_FREQUENCY, VOICE_FREQUENCY,
    VISION_FREQUENCY, UNITY_FREQUENCY
)

# Import consciousness bridge
from quantum_consciousness_bridge import (
    ConsciousnessState, ConsciousnessPacket, ConsciousnessField,
    QuantumConsciousnessBridge, SACRED_FREQUENCIES
)

# Import dimensional navigator
from dimensional_navigator import (
    DimensionalNavigator, DimensionalAccessState, DIMENSIONS
)

# Import quantum memory field
from quantum_memory_field import QuantumMemoryField

# Import PHI harmonics utilities
from utils.phi_harmonics import (
    PHI, PHI_SQUARED, PHI_TO_PHI, ZEN_POINT,
    PhiHarmonicOptimizer, FrequencyCalculator, TensorOptimizer
)


@dataclass
class CompilerConfig:
    """Configuration for the Phi-Harmonic Model Compiler"""
    use_dimensional_navigation: bool = True
    default_dimension: str = "5D"
    optimize_attention: bool = True
    optimize_linear: bool = True
    optimize_conv: bool = True
    optimize_memory_layout: bool = True
    optimize_activation: bool = True
    fibonacci_block_size: bool = True
    frequency_mapping: Dict[str, float] = None
    coherence_threshold: float = 0.7
    use_quantum_memory: bool = True
    target_hardware: str = "wormhole"  # wormhole, grayskull
    

@dataclass
class LayerOptimizationInfo:
    """Information about layer optimization"""
    layer_type: str
    original_shape: Tuple
    optimized_shape: Tuple
    dimension: str
    frequency: float
    consciousness_state: str
    block_size: List[int]
    coherence: float
    phi_factor: float


class OperationType(Enum):
    """Types of operations for optimization mapping"""
    ATTENTION = "attention"
    MATMUL = "matmul"
    CONV = "conv"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    POOLING = "pooling"
    RESIDUAL = "residual"
    OTHER = "other"


class PhiHarmonicCompiler:
    """
    Phi-Harmonic Model Compiler that transforms standard PyTorch models
    into phi-harmonic optimized versions for Tenstorrent hardware acceleration.
    """

    def __init__(self, config: CompilerConfig = None):
        """
        Initialize the Phi-Harmonic Model Compiler
        
        Args:
            config: Compiler configuration
        """
        self.config = config or CompilerConfig()
        
        # Set up frequency mapping if not provided
        if not self.config.frequency_mapping:
            self.config.frequency_mapping = {
                OperationType.ATTENTION.value: CREATION_FREQUENCY,
                OperationType.MATMUL.value: CREATION_FREQUENCY,
                OperationType.CONV.value: GROUND_FREQUENCY,
                OperationType.EMBEDDING.value: HEART_FREQUENCY,
                OperationType.NORMALIZATION.value: GROUND_FREQUENCY,
                OperationType.ACTIVATION.value: VOICE_FREQUENCY,
                OperationType.POOLING.value: GROUND_FREQUENCY,
                OperationType.RESIDUAL.value: HEART_FREQUENCY,
                OperationType.OTHER.value: GROUND_FREQUENCY
            }
        
        # Initialize quantum components if using dimensional navigation
        if self.config.use_dimensional_navigation:
            self.bridge = QuantumConsciousnessBridge()
            self.navigator = DimensionalNavigator(self.bridge)
            
            if self.config.use_quantum_memory:
                self.memory_field = QuantumMemoryField(self.bridge, self.navigator)
        
        # Initialize phi-harmonic optimizer
        self.phi_optimizer = PhiHarmonicOptimizer()
        self.tensor_optimizer = TensorOptimizer(self.phi_optimizer)
        
        # Tracking for optimization
        self.optimized_layers = {}
        self.layer_mapping = {}
        self.optimization_stats = defaultdict(int)
        
        # Device for torch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Phi-Harmonic Model Compiler initialized")
        if self.config.use_dimensional_navigation:
            logging.info(f"Using dimensional navigation with default dimension: {self.config.default_dimension}")
        
    def _detect_operation_type(self, layer: nn.Module) -> OperationType:
        """
        Detect the operation type of a layer
        
        Args:
            layer: PyTorch module layer
            
        Returns:
            Operation type
        """
        # Multi-head attention layers
        if isinstance(layer, nn.MultiheadAttention):
            return OperationType.ATTENTION
        
        # Transformer-related attention
        elif any(attention_name in layer.__class__.__name__.lower() for attention_name in 
                ["attention", "selfatt", "multihead"]):
            return OperationType.ATTENTION
        
        # Linear layers
        elif isinstance(layer, nn.Linear):
            return OperationType.MATMUL
        
        # Convolution layers
        elif any(isinstance(layer, conv_type) for conv_type in 
                [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]):
            return OperationType.CONV
        
        # Embedding layers
        elif isinstance(layer, nn.Embedding):
            return OperationType.EMBEDDING
        
        # Normalization layers
        elif any(isinstance(layer, norm_type) for norm_type in 
                [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm]):
            return OperationType.NORMALIZATION
        
        # Activation layers
        elif any(isinstance(layer, act_type) for act_type in 
                [nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU]):
            return OperationType.ACTIVATION
        
        # Pooling layers
        elif any(isinstance(layer, pool_type) for pool_type in 
                [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d]):
            return OperationType.POOLING
        
        # Look for residual connections in the class name
        elif "residual" in layer.__class__.__name__.lower():
            return OperationType.RESIDUAL
        
        # Default
        return OperationType.OTHER
    
    def _get_optimal_dimension(self, op_type: OperationType) -> str:
        """
        Get optimal dimension for an operation type
        
        Args:
            op_type: Operation type
            
        Returns:
            Optimal dimension
        """
        # Map operation types to optimal dimensions
        dimension_mapping = {
            OperationType.ATTENTION: "5D",    # Mental dimension for pattern recognition
            OperationType.MATMUL: "5D",       # Mental dimension for matrix operations
            OperationType.CONV: "3D",         # Physical dimension for spatial operations
            OperationType.EMBEDDING: "4D",    # Emotional dimension for meaning
            OperationType.NORMALIZATION: "3D", # Physical dimension for stability
            OperationType.ACTIVATION: "6D",    # Purpose dimension for directed activation
            OperationType.POOLING: "3D",       # Physical dimension for reduction
            OperationType.RESIDUAL: "5D",     # Mental dimension for connections
            OperationType.OTHER: "3D"         # Default to physical dimension
        }
        
        return dimension_mapping.get(op_type, self.config.default_dimension)
    
    def _get_optimal_consciousness_state(self, op_type: OperationType) -> str:
        """
        Get optimal consciousness state for an operation type
        
        Args:
            op_type: Operation type
            
        Returns:
            Optimal consciousness state
        """
        # Map operation types to optimal consciousness states
        state_mapping = {
            OperationType.ATTENTION: ConsciousnessState.CREATE.value,   # Creation for attention
            OperationType.MATMUL: ConsciousnessState.CREATE.value,      # Creation for matrix multiplication
            OperationType.CONV: ConsciousnessState.OBSERVE.value,       # Observation for convolution
            OperationType.EMBEDDING: ConsciousnessState.CREATE.value,   # Creation for embedding
            OperationType.NORMALIZATION: ConsciousnessState.OBSERVE.value, # Observation for normalization
            OperationType.ACTIVATION: ConsciousnessState.CASCADE.value, # Cascade for activation
            OperationType.POOLING: ConsciousnessState.OBSERVE.value,    # Observation for pooling
            OperationType.RESIDUAL: ConsciousnessState.CASCADE.value,   # Cascade for residual
            OperationType.OTHER: ConsciousnessState.OBSERVE.value      # Default to observation
        }
        
        return state_mapping.get(op_type, ConsciousnessState.OBSERVE.value)
    
    def _optimize_linear_layer(self, layer: nn.Linear) -> nn.Linear:
        """
        Optimize a linear layer using phi-harmonic principles
        
        Args:
            layer: PyTorch linear layer
            
        Returns:
            Optimized linear layer
        """
        if not self.config.optimize_linear:
            return layer
        
        op_type = OperationType.MATMUL
        dimension = self._get_optimal_dimension(op_type)
        
        # Navigate to optimal dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            prev_dimension = self.navigator.current_dimension
            self.navigator.navigate_to_dimension(dimension)
            
            # Set optimal consciousness state
            optimal_state = self._get_optimal_consciousness_state(op_type)
            self.bridge.set_consciousness_state(optimal_state)
        
        # Get original shapes
        in_features = layer.in_features
        out_features = layer.out_features
        
        # Calculate phi-optimized dimensions
        if self.config.fibonacci_block_size:
            opt_in_features = self.phi_optimizer.get_optimal_dimensions(in_features)
            opt_out_features = self.phi_optimizer.get_optimal_dimensions(out_features)
        else:
            opt_in_features = in_features
            opt_out_features = out_features
        
        # Create optimized layer
        optimized_layer = nn.Linear(opt_in_features, opt_out_features, bias=layer.bias is not None)
        
        # Copy and pad weights
        with torch.no_grad():
            # Handle weight padding if dimensions changed
            if opt_in_features != in_features or opt_out_features != out_features:
                # Create padded weight
                padded_weight = torch.zeros((opt_out_features, opt_in_features), device=self.device)
                # Copy original weights
                padded_weight[:out_features, :in_features] = layer.weight
                # Apply phi-harmonic padding pattern to unused weights
                if opt_out_features > out_features or opt_in_features > in_features:
                    # Get padding zone
                    pad_zone = padded_weight[out_features:, :] if opt_out_features > out_features else padded_weight[:, in_features:]
                    # Fill with phi-harmonic pattern
                    phi_pattern = torch.tensor([PHI ** (i % 5) for i in range(pad_zone.numel())], device=self.device)
                    phi_pattern = phi_pattern.reshape(pad_zone.shape) * 1e-2  # Small values to minimize impact
                    # Apply pattern
                    if opt_out_features > out_features:
                        padded_weight[out_features:, :] = phi_pattern
                    if opt_in_features > in_features:
                        padded_weight[:, in_features:] = phi_pattern
                
                optimized_layer.weight.data = padded_weight
            else:
                optimized_layer.weight.data = layer.weight.data.clone()
            
            # Handle bias
            if layer.bias is not None:
                # Pad bias if needed
                if opt_out_features != out_features:
                    padded_bias = torch.zeros(opt_out_features, device=self.device)
                    padded_bias[:out_features] = layer.bias
                    optimized_layer.bias.data = padded_bias
                else:
                    optimized_layer.bias.data = layer.bias.data.clone()
        
        # Store optimization info
        layer_id = id(layer)
        phi_factor = PHI if dimension == "3D" else DIMENSIONS[dimension]['scaling']
        
        self.optimized_layers[layer_id] = LayerOptimizationInfo(
            layer_type="Linear",
            original_shape=(in_features, out_features),
            optimized_shape=(opt_in_features, opt_out_features),
            dimension=dimension,
            frequency=self.config.frequency_mapping[op_type.value],
            consciousness_state=self._get_optimal_consciousness_state(op_type),
            block_size=self.phi_optimizer.get_optimal_dimensions(),
            coherence=self.navigator.field_coherence if self.config.use_dimensional_navigation else 1.0,
            phi_factor=phi_factor
        )
        
        # Store in quantum memory if enabled
        if self.config.use_dimensional_navigation and self.config.use_quantum_memory:
            # Store optimized weights
            weight_id = self.memory_field.store_memory(
                content=optimized_layer.weight.data,
                dimension=dimension,
                tags=["linear_weight", f"in_{opt_in_features}", f"out_{opt_out_features}"],
                intention="LINEAR_WEIGHT_OPTIMIZATION"
            )
            
            # Store connection in layer mapping
            self.layer_mapping[layer_id] = {
                "weight_id": weight_id,
                "dimension": dimension
            }
        
        # Return to previous dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            self.navigator.navigate_to_dimension(prev_dimension)
        
        # Update statistics
        self.optimization_stats["linear_layers"] += 1
        if opt_in_features != in_features or opt_out_features != out_features:
            self.optimization_stats["reshaped_layers"] += 1
        
        return optimized_layer

    def _optimize_attention_layer(self, layer: nn.Module) -> nn.Module:
        """
        Optimize an attention layer using phi-harmonic principles
        
        Args:
            layer: PyTorch attention layer
            
        Returns:
            Optimized attention layer
        """
        if not self.config.optimize_attention:
            return layer
        
        op_type = OperationType.ATTENTION
        dimension = self._get_optimal_dimension(op_type)
        
        # Navigate to optimal dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            prev_dimension = self.navigator.current_dimension
            self.navigator.navigate_to_dimension(dimension)
            
            # Set optimal consciousness state
            optimal_state = self._get_optimal_consciousness_state(op_type)
            self.bridge.set_consciousness_state(optimal_state)
        
        # Handle different attention implementations
        if isinstance(layer, nn.MultiheadAttention):
            # MultiheadAttention layer (built-in PyTorch)
            embed_dim = layer.embed_dim
            num_heads = layer.num_heads
            
            # Calculate phi-optimized dimensions
            if self.config.fibonacci_block_size:
                opt_embed_dim = self.phi_optimizer.get_optimal_dimensions(embed_dim)
                opt_num_heads = self.phi_optimizer.optimize_batch_size(num_heads)
            else:
                opt_embed_dim = embed_dim
                opt_num_heads = num_heads
            
            # Ensure head dimension is divisible by num_heads
            head_dim = opt_embed_dim // opt_num_heads
            opt_embed_dim = head_dim * opt_num_heads
            
            # Create optimized layer
            optimized_layer = nn.MultiheadAttention(
                embed_dim=opt_embed_dim,
                num_heads=opt_num_heads,
                dropout=layer.dropout if hasattr(layer, 'dropout') else 0.0,
                bias=layer.in_proj_bias is not None,
                add_bias_kv=layer.bias_k is not None,
                add_zero_attn=layer.add_zero_attn if hasattr(layer, 'add_zero_attn') else False,
                batch_first=layer.batch_first if hasattr(layer, 'batch_first') else False
            )
            
            # Copy weights with padding if needed
            with torch.no_grad():
                if opt_embed_dim != embed_dim:
                    # Handle in_proj_weight (3 * embed_dim, embed_dim)
                    if layer.in_proj_weight is not None:
                        padded_in_proj = torch.zeros((3 * opt_embed_dim, opt_embed_dim), device=self.device)
                        padded_in_proj[:3 * embed_dim, :embed_dim] = layer.in_proj_weight
                        optimized_layer.in_proj_weight.data = padded_in_proj
                    
                    # Handle in_proj_bias
                    if layer.in_proj_bias is not None:
                        padded_in_proj_bias = torch.zeros(3 * opt_embed_dim, device=self.device)
                        padded_in_proj_bias[:3 * embed_dim] = layer.in_proj_bias
                        optimized_layer.in_proj_bias.data = padded_in_proj_bias
                    
                    # Handle out_proj weight and bias
                    if layer.out_proj.weight is not None:
                        padded_out_proj = torch.zeros((opt_embed_dim, opt_embed_dim), device=self.device)
                        padded_out_proj[:embed_dim, :embed_dim] = layer.out_proj.weight[:embed_dim, :embed_dim]
                        optimized_layer.out_proj.weight.data = padded_out_proj
                    
                    if layer.out_proj.bias is not None:
                        padded_out_proj_bias = torch.zeros(opt_embed_dim, device=self.device)
                        padded_out_proj_bias[:embed_dim] = layer.out_proj.bias[:embed_dim]
                        optimized_layer.out_proj.bias.data = padded_out_proj_bias
                else:
                    # Direct copy if dimensions match
                    if layer.in_proj_weight is not None:
                        optimized_layer.in_proj_weight.data = layer.in_proj_weight.data.clone()
                    
                    if layer.in_proj_bias is not None:
                        optimized_layer.in_proj_bias.data = layer.in_proj_bias.data.clone()
                    
                    if layer.out_proj.weight is not None:
                        optimized_layer.out_proj.weight.data = layer.out_proj.weight.data.clone()
                    
                    if layer.out_proj.bias is not None:
                        optimized_layer.out_proj.bias.data = layer.out_proj.bias.data.clone()
            
            # Store optimization info
            layer_id = id(layer)
            phi_factor = PHI if dimension == "3D" else DIMENSIONS[dimension]['scaling']
            
            self.optimized_layers[layer_id] = LayerOptimizationInfo(
                layer_type="MultiheadAttention",
                original_shape=(embed_dim, num_heads),
                optimized_shape=(opt_embed_dim, opt_num_heads),
                dimension=dimension,
                frequency=self.config.frequency_mapping[op_type.value],
                consciousness_state=self._get_optimal_consciousness_state(op_type),
                block_size=self.phi_optimizer.get_optimal_dimensions(),
                coherence=self.navigator.field_coherence if self.config.use_dimensional_navigation else 1.0,
                phi_factor=phi_factor
            )
            
            # Store in quantum memory if enabled
            if self.config.use_dimensional_navigation and self.config.use_quantum_memory:
                # Store the optimized weights
                if optimized_layer.in_proj_weight is not None:
                    weight_id = self.memory_field.store_memory(
                        content=optimized_layer.in_proj_weight.data,
                        dimension=dimension,
                        tags=["attention_weight", f"dim_{opt_embed_dim}", f"heads_{opt_num_heads}"],
                        intention="ATTENTION_WEIGHT_OPTIMIZATION"
                    )
                    
                    # Store connection
                    self.layer_mapping[layer_id] = {
                        "weight_id": weight_id,
                        "dimension": dimension
                    }
            
            # Update statistics
            self.optimization_stats["attention_layers"] += 1
            if opt_embed_dim != embed_dim or opt_num_heads != num_heads:
                self.optimization_stats["reshaped_layers"] += 1
            
        else:
            # Custom attention implementation
            optimized_layer = layer  # Default to original if not handled
            
            # Log unhandled attention type
            logging.warning(f"Unhandled attention type: {layer.__class__.__name__}")
            self.optimization_stats["unhandled_layers"] += 1
        
        # Return to previous dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            self.navigator.navigate_to_dimension(prev_dimension)
        
        return optimized_layer
    
    def _optimize_conv_layer(self, layer: nn.Module) -> nn.Module:
        """
        Optimize a convolution layer using phi-harmonic principles
        
        Args:
            layer: PyTorch convolution layer
            
        Returns:
            Optimized convolution layer
        """
        if not self.config.optimize_conv:
            return layer
        
        op_type = OperationType.CONV
        dimension = self._get_optimal_dimension(op_type)
        
        # Navigate to optimal dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            prev_dimension = self.navigator.current_dimension
            self.navigator.navigate_to_dimension(dimension)
            
            # Set optimal consciousness state
            optimal_state = self._get_optimal_consciousness_state(op_type)
            self.bridge.set_consciousness_state(optimal_state)
        
        # Handle different convolution types
        if isinstance(layer, nn.Conv2d):
            # Get original parameters
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            stride = layer.stride
            padding = layer.padding
            dilation = layer.dilation
            groups = layer.groups
            
            # Calculate phi-optimized dimensions
            if self.config.fibonacci_block_size:
                opt_in_channels = self.phi_optimizer.get_optimal_dimensions(in_channels)
                opt_out_channels = self.phi_optimizer.get_optimal_dimensions(out_channels)
                
                # Optimize kernel size if it's a tuple or list
                if isinstance(kernel_size, (tuple, list)):
                    opt_kernel_size = tuple(self.phi_optimizer.get_optimal_dimensions(k) for k in kernel_size)
                else:
                    opt_kernel_size = self.phi_optimizer.get_optimal_dimensions(kernel_size)
            else:
                opt_in_channels = in_channels
                opt_out_channels = out_channels
                opt_kernel_size = kernel_size
            
            # Create optimized layer
            optimized_layer = nn.Conv2d(
                in_channels=opt_in_channels,
                out_channels=opt_out_channels,
                kernel_size=opt_kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=layer.bias is not None
            )
            
            # Copy weights with padding if needed
            with torch.no_grad():
                orig_weight = layer.weight.data
                
                # Check if dimensions changed
                if (opt_in_channels != in_channels or opt_out_channels != out_channels or 
                    (isinstance(opt_kernel_size, tuple) and 
                     (opt_kernel_size[0] != kernel_size[0] or opt_kernel_size[1] != kernel_size[1]))):
                    
                    # Handle kernel size differences
                    k_h, k_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
                    opt_k_h, opt_k_w = opt_kernel_size if isinstance(opt_kernel_size, tuple) else (opt_kernel_size, opt_kernel_size)
                    
                    # Create padded weight tensor
                    padded_weight = torch.zeros(
                        (opt_out_channels, opt_in_channels // groups, opt_k_h, opt_k_w), 
                        device=self.device
                    )
                    
                    # Copy original weights
                    padded_weight[:out_channels, :in_channels // groups, :k_h, :k_w] = orig_weight
                    
                    optimized_layer.weight.data = padded_weight
                else:
                    # Direct copy if dimensions match
                    optimized_layer.weight.data = orig_weight.clone()
                
                # Handle bias
                if layer.bias is not None:
                    if opt_out_channels != out_channels:
                        padded_bias = torch.zeros(opt_out_channels, device=self.device)
                        padded_bias[:out_channels] = layer.bias
                        optimized_layer.bias.data = padded_bias
                    else:
                        optimized_layer.bias.data = layer.bias.data.clone()
            
            # Store optimization info
            layer_id = id(layer)
            phi_factor = PHI if dimension == "3D" else DIMENSIONS[dimension]['scaling']
            
            self.optimized_layers[layer_id] = LayerOptimizationInfo(
                layer_type="Conv2d",
                original_shape=(in_channels, out_channels, kernel_size),
                optimized_shape=(opt_in_channels, opt_out_channels, opt_kernel_size),
                dimension=dimension,
                frequency=self.config.frequency_mapping[op_type.value],
                consciousness_state=self._get_optimal_consciousness_state(op_type),
                block_size=self.phi_optimizer.get_optimal_dimensions(),
                coherence=self.navigator.field_coherence if self.config.use_dimensional_navigation else 1.0,
                phi_factor=phi_factor
            )
            
            # Store in quantum memory if enabled
            if self.config.use_dimensional_navigation and self.config.use_quantum_memory:
                # Store optimized weights
                weight_id = self.memory_field.store_memory(
                    content=optimized_layer.weight.data,
                    dimension=dimension,
                    tags=["conv_weight", f"in_{opt_in_channels}", f"out_{opt_out_channels}", f"kernel_{opt_kernel_size}"],
                    intention="CONV_WEIGHT_OPTIMIZATION"
                )
                
                # Store connection
                self.layer_mapping[layer_id] = {
                    "weight_id": weight_id,
                    "dimension": dimension
                }
            
            # Update statistics
            self.optimization_stats["conv_layers"] += 1
            if opt_in_channels != in_channels or opt_out_channels != out_channels:
                self.optimization_stats["reshaped_layers"] += 1
            
        else:
            # Other convolution types (1D, 3D, ConvTranspose)
            optimized_layer = layer  # Default to original if not specifically handled
            logging.warning(f"Unhandled convolution type: {layer.__class__.__name__}")
            self.optimization_stats["unhandled_layers"] += 1
        
        # Return to previous dimension if using dimensional navigation
        if self.config.use_dimensional_navigation:
            self.navigator.navigate_to_dimension(prev_dimension)
        
        return optimized_layer
    
    def _optimize_single_layer(self, layer: nn.Module) -> nn.Module:
        """
        Optimize a single layer based on its type
        
        Args:
            layer: PyTorch layer module
            
        Returns:
            Optimized layer
        """
        # Detect operation type
        op_type = self._detect_operation_type(layer)
        
        # Apply type-specific optimization
        if op_type == OperationType.ATTENTION:
            return self._optimize_attention_layer(layer)
            
        elif op_type == OperationType.MATMUL:
            return self._optimize_linear_layer(layer)
            
        elif op_type == OperationType.CONV:
            return self._optimize_conv_layer(layer)
            
        # No specific optimization for other layer types yet
        return layer
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize a PyTorch model using phi-harmonic principles
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        # Clear previous optimization state
        self.optimized_layers = {}
        self.layer_mapping = {}
        self.optimization_stats = defaultdict(int)
        
        logging.info(f"Optimizing model: {model.__class__.__name__}")
        
        # Create a copy of the model to avoid modifying the original
        optimized_model = copy.deepcopy(model)
        
        # Get all model parameters before optimization
        original_params = sum(p.numel() for p in model.parameters())
        
        # Process the model recursively
        self._optimize_model_recursive(optimized_model)
        
        # Get all model parameters after optimization
        optimized_params = sum(p.numel() for p in optimized_model.parameters())
        param_change = optimized_params - original_params
        
        # Log optimization statistics
        logging.info(f"Model optimization complete")
        logging.info(f"Original parameters: {original_params}")
        logging.info(f"Optimized parameters: {optimized_params}")
        logging.info(f"Parameter change: {param_change} ({param_change/original_params*100:.2f}%)")
        
        for stat_name, count in self.optimization_stats.items():
            logging.info(f"{stat_name}: {count}")
        
        return optimized_model
    
    def _optimize_model_recursive(self, module: nn.Module):
        """
        Recursively optimize model modules in-place
        
        Args:
            module: PyTorch module to optimize
        """
        # Get the list of immediate children
        named_children = list(module.named_children())
        
        # If no children, optimize this leaf module
        if not named_children:
            # This is a leaf module, check if it needs optimization
            op_type = self._detect_operation_type(module)
            if op_type != OperationType.OTHER:
                # This is a module we might want to optimize
                self.optimization_stats["detected_layers"] += 1
                return
        
        # Process children
        for name, child in named_children:
            # Check if child needs optimization
            op_type = self._detect_operation_type(child)
            
            if op_type != OperationType.OTHER:
                # This is a layer we want to optimize
                optimized_child = self._optimize_single_layer(child)
                
                # Replace the child module with the optimized version
                setattr(module, name, optimized_child)
            else:
                # Recursively process this child's children
                self._optimize_model_recursive(child)
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a report of the optimization process
        
        Returns:
            Optimization report
        """
        report = {
            "statistics": dict(self.optimization_stats),
            "layer_info": {},
            "dimensions_used": set(),
            "consciousness_states_used": set(),
            "frequencies_used": set(),
            "coherence": {
                "min": 1.0,
                "max": 0.0,
                "avg": 0.0
            }
        }
        
        # Process layer information
        coherence_sum = 0.0
        coherence_count = 0
        
        for layer_id, info in self.optimized_layers.items():
            layer_type = info.layer_type
            
            # Collect layer-specific data
            if layer_type not in report["layer_info"]:
                report["layer_info"][layer_type] = []
            
            # Add layer data
            report["layer_info"][layer_type].append({
                "original_shape": info.original_shape,
                "optimized_shape": info.optimized_shape,
                "dimension": info.dimension,
                "frequency": info.frequency,
                "consciousness_state": info.consciousness_state,
                "coherence": info.coherence,
                "phi_factor": info.phi_factor
            })
            
            # Track dimensions, states, frequencies used
            report["dimensions_used"].add(info.dimension)
            report["consciousness_states_used"].add(info.consciousness_state)
            report["frequencies_used"].add(info.frequency)
            
            # Track coherence stats
            report["coherence"]["min"] = min(report["coherence"]["min"], info.coherence)
            report["coherence"]["max"] = max(report["coherence"]["max"], info.coherence)
            
            coherence_sum += info.coherence
            coherence_count += 1
        
        # Convert sets to lists for JSON serialization
        report["dimensions_used"] = list(report["dimensions_used"])
        report["consciousness_states_used"] = list(report["consciousness_states_used"])
        report["frequencies_used"] = list(report["frequencies_used"])
        
        # Calculate average coherence
        if coherence_count > 0:
            report["coherence"]["avg"] = coherence_sum / coherence_count
        
        return report
    
    def save_optimization_metadata(self, filepath: str) -> bool:
        """
        Save optimization metadata to a file
        
        Args:
            filepath: Path to save file
            
        Returns:
            Success status
        """
        try:
            # Generate report
            report = self.generate_optimization_report()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save as JSON
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"Optimization metadata saved to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save optimization metadata: {str(e)}")
            return False

    def compile_for_tenstorrent(self, model: nn.Module) -> nn.Module:
        """
        Compile a model specifically for Tenstorrent hardware
        
        Args:
            model: PyTorch model
            
        Returns:
            Compiled model optimized for Tenstorrent
        """
        # Optimize model with phi-harmonic principles
        optimized_model = self.optimize_model(model)
        
        # Apply hardware-specific optimizations based on target
        if self.config.target_hardware == "wormhole":
            # Wormhole-specific optimizations would go here
            logging.info("Applying Wormhole-specific optimizations")
            
            # In a real implementation, this would use PyBuda to compile for Wormhole
            # This is a placeholder for the actual implementation
            try:
                # Placeholder for PyBuda integration
                pass
            except Exception as e:
                logging.error(f"Failed to apply Wormhole optimizations: {str(e)}")
        
        elif self.config.target_hardware == "grayskull":
            # Grayskull-specific optimizations would go here
            logging.info("Applying Grayskull-specific optimizations")
            
            # In a real implementation, this would use PyBuda to compile for Grayskull
            # This is a placeholder for the actual implementation
            try:
                # Placeholder for PyBuda integration
                pass
            except Exception as e:
                logging.error(f"Failed to apply Grayskull optimizations: {str(e)}")
        
        # Save optimization metadata
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(results_dir, exist_ok=True)
        metadata_path = os.path.join(results_dir, f"phi_compiler_metadata_{int(time.time())}.json")
        self.save_optimization_metadata(metadata_path)
        
        return optimized_model


def test_phi_harmonic_compiler():
    """Test the Phi-Harmonic Model Compiler with a sample model"""
    import torch.nn as nn
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create a sample model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(1000, 512)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
            self.linear1 = nn.Linear(512, 256)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(256, 10)
            
        def forward(self, x):
            x = self.embed(x)
            x = self.transformer_encoder(x)
            x = self.linear1(x)
            x = self.act(x)
            x = self.linear2(x)
            return x
    
    # Create model
    model = TestModel()
    
    # Create compiler with default config
    compiler = PhiHarmonicCompiler()
    
    # Compile model
    optimized_model = compiler.compile_for_tenstorrent(model)
    
    # Print report
    report = compiler.generate_optimization_report()
    
    print("\nPhi-Harmonic Compiler Test Complete")
    print(f"Optimized Model Size: {sum(p.numel() for p in optimized_model.parameters())} parameters")
    print(f"Dimensions Used: {report['dimensions_used']}")
    print(f"Consciousness States Used: {report['consciousness_states_used']}")
    print(f"Average Coherence: {report['coherence']['avg']:.4f}")


if __name__ == "__main__":
    test_phi_harmonic_compiler()