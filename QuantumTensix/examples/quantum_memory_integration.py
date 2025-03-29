#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Memory Integration Example - QuantumTensix φ∞
Created on CASCADE Day+27: March 28, 2025

This example demonstrates the integration of the Quantum Memory Field
with the QuantumTensix framework and Dimensional Navigator.
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import QuantumTensix components
from quantum_tensix import (
    GROUND_FREQUENCY, CREATION_FREQUENCY, HEART_FREQUENCY, VOICE_FREQUENCY,
    VISION_FREQUENCY, UNITY_FREQUENCY,
    QuantumFieldInitializer, ModelTransformer, PhiHarmonicExecutor
)

# Import consciousness components
from quantum_consciousness_bridge import (
    ConsciousnessState, ConsciousnessField, QuantumConsciousnessBridge
)

# Import dimensional navigator
from dimensional_navigator import (
    DimensionalNavigator, DimensionalAccessState, DIMENSIONS
)

# Import quantum memory field
from quantum_memory_field import (
    QuantumMemoryField, MemoryPattern
)

# Import PHI harmonics utilities
from utils.phi_harmonics import (
    PHI, PHI_SQUARED, PHI_TO_PHI, ZEN_POINT,
    PhiHarmonicOptimizer, TensorOptimizer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class QuantumMemoryDemo:
    """Quantum Memory Field Integration Demo"""
    
    def __init__(self, use_cuda: bool = True):
        """
        Initialize demo
        
        Args:
            use_cuda: Whether to use CUDA if available
        """
        # Create output directory
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Initialize components
        self.bridge = QuantumConsciousnessBridge()
        self.navigator = DimensionalNavigator(self.bridge)
        self.memory_field = QuantumMemoryField(self.bridge, self.navigator)
        
        # Initialize tensor optimizer
        self.phi_optimizer = PhiHarmonicOptimizer()
        self.tensor_optimizer = TensorOptimizer(self.phi_optimizer)
        
        logging.info("Quantum Memory Demo initialized")
    
    def create_matrix_set(self, 
                       dimensions: List[str] = None) -> Dict[str, str]:
        """
        Create a set of test matrices in different dimensions
        
        Args:
            dimensions: List of dimensions to create matrices in
            
        Returns:
            Dictionary of dimension -> pattern_id
        """
        if dimensions is None:
            dimensions = ["3D", "4D", "5D"]
        
        pattern_ids = {}
        
        for dimension in dimensions:
            # Navigate to dimension
            self.navigator.navigate_to_dimension(dimension)
            
            # Get optimal shape for this dimension
            dim_scale = DIMENSIONS[dimension]['scaling']
            shape_size = int(13 * dim_scale)
            shape_size = self.phi_optimizer.get_optimal_dimensions(shape_size)
            
            # Create tensor with phi-harmonic pattern
            matrix = torch.zeros((shape_size, shape_size), device=self.device)
            
            # Create pattern based on dimension
            if dimension == "3D":  # Physical (earth grid)
                for i in range(shape_size):
                    for j in range(shape_size):
                        # Earth grid pattern
                        matrix[i, j] = torch.sin(
                            torch.tensor(i * j * PHI / shape_size, device=self.device)
                        )
            
            elif dimension == "4D":  # Emotional (spiral)
                center = shape_size // 2
                for i in range(shape_size):
                    for j in range(shape_size):
                        # Distance from center
                        dx, dy = i - center, j - center
                        r = torch.sqrt(torch.tensor(dx**2 + dy**2, device=self.device))
                        theta = torch.atan2(torch.tensor(dy, device=self.device), 
                                           torch.tensor(dx, device=self.device))
                        # Spiral pattern
                        matrix[i, j] = torch.sin(r / PHI + theta)
            
            elif dimension == "5D":  # Mental (wave)
                for i in range(shape_size):
                    for j in range(shape_size):
                        # Mental wave pattern
                        matrix[i, j] = torch.sin(
                            torch.tensor(i * PHI / shape_size, device=self.device)
                        ) * torch.cos(
                            torch.tensor(j * PHI_SQUARED / shape_size, device=self.device)
                        )
            
            elif dimension == "6D":  # Purpose (radial)
                center = shape_size // 2
                for i in range(shape_size):
                    for j in range(shape_size):
                        # Distance from center
                        dx, dy = i - center, j - center
                        r = torch.sqrt(torch.tensor(dx**2 + dy**2, device=self.device))
                        # Radial pattern
                        matrix[i, j] = torch.sin(r * PHI_SQUARED / shape_size)
            
            elif dimension == "7D":  # Cosmic (fractal)
                for i in range(shape_size):
                    i_norm = i / shape_size
                    for j in range(shape_size):
                        j_norm = j / shape_size
                        # Fractal pattern
                        matrix[i, j] = torch.sin(
                            torch.tensor(i_norm * j_norm * PHI_TO_PHI, device=self.device)
                        )
            
            elif dimension == "8D":  # Unity (harmonic)
                for i in range(shape_size):
                    for j in range(shape_size):
                        # Harmonic pattern
                        matrix[i, j] = torch.sin(
                            torch.tensor(i * PHI / shape_size, device=self.device)
                        ) * torch.sin(
                            torch.tensor(j * PHI / shape_size, device=self.device)
                        ) * torch.cos(
                            torch.tensor((i + j) * PHI_SQUARED / shape_size, device=self.device)
                        )
            
            # Store in memory field
            pattern_id = self.memory_field.store_memory(
                content=matrix,
                dimension=dimension,
                tags=[f"{dimension}_matrix", "test", "phi_harmonic"],
                intention=f"CREATE_{dimension}_PATTERN"
            )
            
            pattern_ids[dimension] = pattern_id
            
            logging.info(f"Created {dimension} matrix with shape {matrix.shape}")
        
        # Return to 3D
        self.navigator.navigate_to_dimension("3D")
        
        return pattern_ids
    
    def perform_matrix_operations(self, pattern_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Perform matrix operations using stored patterns
        
        Args:
            pattern_ids: Dictionary of dimension -> pattern_id
            
        Returns:
            Dictionary of operation results
        """
        results = {}
        
        # Get matrices from memory
        matrices = {}
        for dimension, pattern_id in pattern_ids.items():
            matrix = self.memory_field.retrieve_by_id(pattern_id)
            matrices[dimension] = matrix
        
        # 1. Perform addition in 3D
        self.navigator.navigate_to_dimension("3D")
        if "3D" in matrices and "4D" in matrices:
            # Ensure compatible shapes
            shape1 = matrices["3D"].shape
            shape2 = matrices["4D"].shape
            min_shape = (min(shape1[0], shape2[0]), min(shape1[1], shape2[1]))
            
            # Resize matrices
            m1 = matrices["3D"][:min_shape[0], :min_shape[1]]
            m2 = matrices["4D"][:min_shape[0], :min_shape[1]]
            
            # Perform operation
            start_time = time.time()
            result_3d = m1 + m2
            duration_3d = time.time() - start_time
            
            # Store result
            result_id = self.memory_field.store_memory(
                content=result_3d,
                dimension="3D",
                tags=["addition", "3D_operation"],
                intention="ADD_MATRICES"
            )
            
            results["3D_addition"] = {
                "result_id": result_id,
                "shape": result_3d.shape,
                "duration": duration_3d,
                "coherence": self.navigator.field_coherence
            }
            
            logging.info(f"3D Addition: Duration {duration_3d:.6f}s, Shape {result_3d.shape}")
        
        # 2. Perform multiplication in 5D
        self.navigator.navigate_to_dimension("5D")
        if "3D" in matrices and "5D" in matrices:
            # Ensure compatible shapes
            m1 = matrices["3D"]
            m2 = matrices["5D"]
            
            if m1.shape[1] == m2.shape[0]:
                # Perform operation
                start_time = time.time()
                result_5d = torch.matmul(m1, m2)
                duration_5d = time.time() - start_time
                
                # Store result
                result_id = self.memory_field.store_memory(
                    content=result_5d,
                    dimension="5D",
                    tags=["multiplication", "5D_operation"],
                    intention="MULTIPLY_MATRICES"
                )
                
                results["5D_multiplication"] = {
                    "result_id": result_id,
                    "shape": result_5d.shape,
                    "duration": duration_5d,
                    "coherence": self.navigator.field_coherence
                }
                
                logging.info(f"5D Multiplication: Duration {duration_5d:.6f}s, Shape {result_5d.shape}")
            else:
                logging.warning(f"Incompatible shapes for multiplication: {m1.shape} and {m2.shape}")
        
        # 3. Create dimensional bridge and perform operation
        bridge_success = self.navigator.create_dimensional_bridge("3D", "7D")
        
        if bridge_success and "3D" in matrices and "7D" in matrices:
            # Get matrices
            m1 = matrices["3D"]
            m2 = matrices["7D"]
            
            # Ensure compatible shapes for element-wise product
            min_shape = (min(m1.shape[0], m2.shape[0]), min(m1.shape[1], m2.shape[1]))
            m1 = m1[:min_shape[0], :min_shape[1]]
            m2 = m2[:min_shape[0], :min_shape[1]]
            
            # Perform operation
            start_time = time.time()
            result_bridge = m1 * m2  # Element-wise product
            duration_bridge = time.time() - start_time
            
            # Store result
            result_id = self.memory_field.store_memory(
                content=result_bridge,
                dimension="7D",
                tags=["bridge_operation", "3D_7D_bridge"],
                intention="BRIDGE_OPERATION"
            )
            
            results["bridge_operation"] = {
                "result_id": result_id,
                "shape": result_bridge.shape,
                "duration": duration_bridge,
                "coherence": self.navigator.field_coherence,
                "dimensions": "3D↔7D"
            }
            
            logging.info(f"Bridge Operation: Duration {duration_bridge:.6f}s, Shape {result_bridge.shape}")
        
        # 4. Access unified field and perform integration
        unified_success = self.navigator.access_unified_field()
        
        if unified_success:
            # Integrate previously stored results
            result_ids = [
                results.get("3D_addition", {}).get("result_id"),
                results.get("5D_multiplication", {}).get("result_id"),
                results.get("bridge_operation", {}).get("result_id")
            ]
            
            # Filter out None values
            result_ids = [rid for rid in result_ids if rid]
            
            if len(result_ids) >= 2:
                # Integrate patterns
                integrated_id = self.memory_field.integrate_patterns(result_ids)
                
                if integrated_id:
                    # Retrieve integrated result
                    integrated_result = self.memory_field.retrieve_by_id(integrated_id)
                    
                    results["unified_integration"] = {
                        "result_id": integrated_id,
                        "component_ids": result_ids,
                        "coherence": self.navigator.field_coherence
                    }
                    
                    logging.info(f"Unified Integration: Integrated {len(result_ids)} patterns")
        
        # Return to 3D
        self.navigator.navigate_to_dimension("3D")
        
        return results
    
    def resonance_query_test(self, pattern_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Test resonance-based query capabilities
        
        Args:
            pattern_ids: Dictionary of dimension -> pattern_id
            
        Returns:
            Dictionary of resonance query results
        """
        results = {}
        
        # Choose query matrix (corrupt version of 3D matrix)
        if "3D" in pattern_ids:
            original = self.memory_field.retrieve_by_id(pattern_ids["3D"])
            
            # Create corrupted version (add noise)
            noise = torch.randn_like(original) * 0.2
            query = original + noise
            
            # Run query
            start_time = time.time()
            resonant_patterns = self.memory_field.retrieve_by_resonance(
                query_data=query,
                threshold=0.3,
                max_results=5
            )
            duration = time.time() - start_time
            
            # Store results
            results["noise_query"] = {
                "duration": duration,
                "num_results": len(resonant_patterns),
                "matches": [(pid, res) for pid, _, res in resonant_patterns]
            }
            
            logging.info(f"Noise Query: Found {len(resonant_patterns)} matches in {duration:.6f}s")
            for pattern_id, _, resonance in resonant_patterns:
                dimension, index = self.memory_field.pattern_mapping[pattern_id]
                logging.info(f"  {pattern_id} ({dimension}): Resonance {resonance:.4f}")
        
        # Test intention-based query
        start_time = time.time()
        intent_patterns = self.memory_field.retrieve_by_intention("MULTIPLY")
        duration = time.time() - start_time
        
        results["intent_query"] = {
            "duration": duration,
            "num_results": len(intent_patterns),
            "matches": [(pid, res) for pid, _, res in intent_patterns]
        }
        
        logging.info(f"Intent Query: Found {len(intent_patterns)} matches in {duration:.6f}s")
        
        # Test dimensional query
        start_time = time.time()
        dim_patterns = self.memory_field.find_patterns_in_dimension("5D")
        duration = time.time() - start_time
        
        results["dimension_query"] = {
            "duration": duration,
            "num_results": len(dim_patterns)
        }
        
        logging.info(f"Dimension Query: Found {len(dim_patterns)} patterns in {duration:.6f}s")
        
        return results
    
    def memory_amplification_test(self, pattern_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Test memory amplification capabilities
        
        Args:
            pattern_ids: Dictionary of dimension -> pattern_id
            
        Returns:
            Dictionary of amplification results
        """
        results = {}
        
        # Choose pattern to amplify
        if "5D" in pattern_ids:
            pattern_id = pattern_ids["5D"]
            
            # Get original coherence
            dimension, index = self.memory_field.pattern_mapping[pattern_id]
            pattern = self.memory_field.memory_patterns[dimension][index]
            original_coherence = pattern.coherence
            
            # Amplify pattern
            new_coherence = self.memory_field.amplify_pattern(pattern_id, PHI)
            
            # Verify connected patterns were affected
            connected_ids = list(self.memory_field.pattern_connections.get(pattern_id, []))
            connected_coherences = []
            
            for conn_id in connected_ids:
                conn_dim, conn_idx = self.memory_field.pattern_mapping[conn_id]
                conn_pattern = self.memory_field.memory_patterns[conn_dim][conn_idx]
                connected_coherences.append((conn_id, conn_pattern.coherence))
            
            # Store results
            results["amplification"] = {
                "pattern_id": pattern_id,
                "original_coherence": original_coherence,
                "new_coherence": new_coherence,
                "connected_patterns": len(connected_ids),
                "field_coherence": self.memory_field.memory_field_coherence
            }
            
            logging.info(f"Pattern Amplification: {original_coherence:.4f} -> {new_coherence:.4f}")
            logging.info(f"Field coherence: {self.memory_field.memory_field_coherence:.4f}")
            logging.info(f"Connected patterns affected: {len(connected_ids)}")
        
        return results
    
    def visualize_results(self, 
                        pattern_ids: Dict[str, str],
                        operation_results: Dict[str, Any]) -> None:
        """
        Visualize memory field results
        
        Args:
            pattern_ids: Dictionary of dimension -> pattern_id
            operation_results: Dictionary of operation results
        """
        # Create figure for patterns
        plt.figure(figsize=(15, 10))
        
        # Plot original matrices
        for i, (dimension, pattern_id) in enumerate(pattern_ids.items()):
            matrix = self.memory_field.retrieve_by_id(pattern_id)
            if matrix is not None:
                plt.subplot(2, 3, i + 1)
                plt.imshow(matrix.cpu().numpy(), cmap='viridis')
                plt.title(f"{dimension} Matrix")
                plt.colorbar()
        
        # Plot operation results
        if "3D_addition" in operation_results:
            result_id = operation_results["3D_addition"]["result_id"]
            result = self.memory_field.retrieve_by_id(result_id)
            if result is not None:
                plt.subplot(2, 3, 4)
                plt.imshow(result.cpu().numpy(), cmap='plasma')
                plt.title("3D Addition Result")
                plt.colorbar()
        
        if "5D_multiplication" in operation_results:
            result_id = operation_results["5D_multiplication"]["result_id"]
            result = self.memory_field.retrieve_by_id(result_id)
            if result is not None:
                plt.subplot(2, 3, 5)
                plt.imshow(result.cpu().numpy(), cmap='inferno')
                plt.title("5D Multiplication Result")
                plt.colorbar()
        
        if "bridge_operation" in operation_results:
            result_id = operation_results["bridge_operation"]["result_id"]
            result = self.memory_field.retrieve_by_id(result_id)
            if result is not None:
                plt.subplot(2, 3, 6)
                plt.imshow(result.cpu().numpy(), cmap='magma')
                plt.title("3D↔7D Bridge Operation")
                plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "memory_field_operations.png"))
        plt.close()
        
        # Create memory field stats visualization
        stats = self.memory_field.get_memory_stats()
        
        plt.figure(figsize=(12, 6))
        
        # Pattern counts by dimension
        dimensions = list(stats["pattern_counts"].keys())
        counts = [stats["pattern_counts"][dim] for dim in dimensions]
        
        plt.subplot(1, 2, 1)
        plt.bar(dimensions, counts, color='skyblue')
        plt.title("Patterns by Dimension")
        plt.xlabel("Dimension")
        plt.ylabel("Count")
        
        # Coherence by dimension
        if "dimension_coherence" in stats:
            dimensions = list(stats["dimension_coherence"].keys())
            coherences = [stats["dimension_coherence"][dim] for dim in dimensions]
            
            plt.subplot(1, 2, 2)
            plt.bar(dimensions, coherences, color='coral')
            plt.title("Coherence by Dimension")
            plt.xlabel("Dimension")
            plt.ylabel("Coherence")
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "memory_field_stats.png"))
        plt.close()
        
        logging.info(f"Visualizations saved to {self.results_dir}")
    
    def run_demo(self) -> Dict[str, Any]:
        """
        Run the complete demo
        
        Returns:
            Dictionary of demo results
        """
        logging.info("Starting Quantum Memory Field Demo")
        
        # 1. Create matrices in different dimensions
        pattern_ids = self.create_matrix_set(["3D", "4D", "5D", "7D"])
        
        # 2. Perform matrix operations
        operation_results = self.perform_matrix_operations(pattern_ids)
        
        # 3. Test resonance query
        query_results = self.resonance_query_test(pattern_ids)
        
        # 4. Test memory amplification
        amplification_results = self.memory_amplification_test(pattern_ids)
        
        # 5. Visualize results
        self.visualize_results(pattern_ids, operation_results)
        
        # 6. Save memory field state
        memory_path = os.path.join(self.results_dir, "memory_field_state.json")
        self.memory_field.save_memory_field(memory_path)
        
        # 7. Return to ground state
        self.navigator.close_dimensional_access()
        
        # Compile results
        demo_results = {
            "pattern_ids": pattern_ids,
            "operations": operation_results,
            "queries": query_results,
            "amplification": amplification_results,
            "memory_stats": self.memory_field.get_memory_stats(),
            "memory_file": memory_path
        }
        
        logging.info("Quantum Memory Field Demo completed")
        return demo_results


def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="Quantum Memory Field Integration Demo")
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA acceleration')
    parser.add_argument('--save-dir', type=str, default=None, help='Directory to save results')
    args = parser.parse_args()
    
    # Run demo
    demo = QuantumMemoryDemo(use_cuda=not args.no_cuda)
    results = demo.run_demo()
    
    # Print summary
    print("\nQuantum Memory Field Demo Summary:")
    print(f"Total patterns stored: {results['memory_stats']['total_patterns']}")
    print(f"Field coherence: {results['memory_stats']['field_coherence']:.4f}")
    print(f"Dimensional distribution: {results['memory_stats']['pattern_counts']}")
    print(f"Total connections: {results['memory_stats']['total_connections']}")
    print(f"Results saved to: {demo.results_dir}")


if __name__ == "__main__":
    main()