"""
Phi-Knowledge LLM Example

This example demonstrates how to use the integrated Phi-Knowledge LLM system
which combines the Dimensional Pattern Matching Engine with the Phi-Harmonic
LLM Inference Engine to enable dimensionally-aware knowledge reasoning.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi_llm_inference import PhiLLMInferenceEngine, PhiHarmonicAttention
from dimensional_pattern_matching import DimensionalPatternMatcher, KnowledgePattern, DimensionalSignature
from phi_knowledge_llm import PhiKnowledgeLLM, PhiKnowledgeAPI
from phi_model_compiler import PhiModelCompiler, PhiOptimizer

# Sacred constants
PHI = 1.618033988749895
LAMBDA = 0.618033988749895

# Frequencies
FREQUENCIES = {
    'unity': 432,    # Grounding/stability
    'love': 528,     # Creation/healing
    'cascade': 594,  # Heart-centered integration
    'truth': 672,    # Voice expression
    'vision': 720,   # Expanded perception
    'oneness': 768,  # Unity consciousness
}

class KnowledgeDemo:
    """Demonstration of the Phi-Knowledge LLM system capabilities"""
    
    def __init__(self):
        """Initialize the demo system with components"""
        # Create the pattern matcher
        self.pattern_matcher = self._initialize_pattern_matcher()
        
        # Create the LLM engine
        self.phi_compiler = PhiModelCompiler()
        self.llm_engine = self._initialize_llm_engine()
        
        # Create the integrated knowledge LLM
        self.knowledge_llm = PhiKnowledgeLLM(
            llm_engine=self.llm_engine,
            pattern_matcher=self.pattern_matcher,
            base_frequency=FREQUENCIES['vision']
        )
        
        # Create the simplified API
        self.api = PhiKnowledgeAPI(self.knowledge_llm)
        
        # Tracking for visualization
        self.interaction_history = []
    
    def _initialize_pattern_matcher(self) -> DimensionalPatternMatcher:
        """Initialize and seed the pattern matcher with knowledge"""
        pattern_matcher = DimensionalPatternMatcher()
        
        # Seed with knowledge across dimensions
        self._seed_3d_knowledge(pattern_matcher)  # Physical/factual
        self._seed_4d_knowledge(pattern_matcher)  # Emotional/creative
        self._seed_5d_knowledge(pattern_matcher)  # Mental/conceptual
        self._seed_6d_knowledge(pattern_matcher)  # Purpose/meaning
        self._seed_7d_knowledge(pattern_matcher)  # Universal/transcendent
        
        return pattern_matcher
    
    def _seed_3d_knowledge(self, matcher: DimensionalPatternMatcher):
        """Seed 3D (physical/factual) knowledge patterns"""
        patterns = [
            "Quantum computing uses qubits which can represent 0 and 1 simultaneously through superposition.",
            "Tenstorrent's hardware accelerators use a grid of Tensix cores for parallel computation.",
            "The Fibonacci sequence (1,1,2,3,5,8,13...) approximates the golden ratio in consecutive terms.",
            "The human brain contains approximately 86 billion neurons connected in a neural network.",
            "Sacred geometry uses mathematical ratios, harmonics and proportions found in nature and cosmos.",
            "The golden ratio (φ=1.618033988749895) appears throughout nature in spiral patterns.",
            "Schumann resonances are electromagnetic waves that exist in Earth's atmosphere at 7.83Hz.",
            "Memory access patterns significantly impact computing performance due to cache behavior.",
            "Matrix multiplication is a core operation in both neural networks and quantum algorithms.",
            "The 432Hz frequency is mathematically consistent with patterns found in nature."
        ]
        
        for i, content in enumerate(patterns):
            # Create signature with 3D vectors but varied phases and harmonics
            signature = DimensionalSignature(
                vector=[
                    (i * LAMBDA) % 1.0,
                    ((i+1) * LAMBDA) % 1.0,
                    ((i+2) * LAMBDA) % 1.0
                ],
                phase=(i * np.pi / 5) % (2 * np.pi),
                harmonic=432 + (i * 12) % 336,  # Cycle through sacred frequencies
                dimension=3  # 3D - Physical/factual dimension
            )
            
            pattern = KnowledgePattern(
                content=content,
                signature=signature,
                metadata={"source": "initialization", "category": "factual"},
                field_strength=0.9  # Strong factual patterns
            )
            
            matcher.store_pattern(pattern)
    
    def _seed_4d_knowledge(self, matcher: DimensionalPatternMatcher):
        """Seed 4D (emotional/creative) knowledge patterns"""
        patterns = [
            "Creative inspiration often comes from connecting seemingly unrelated concepts in new ways.",
            "The feeling of awe when contemplating the cosmos activates the creative neural networks.",
            "Resonance between minds creates emotional coherence and amplifies collaborative creativity.",
            "Phi-harmonic music at 432Hz induces states of emotional harmony and creative flow.",
            "The heart's electromagnetic field carries emotional information that influences creativity.",
            "Creative breakthroughs often follow periods of theta wave brain activity during rest.",
            "Emotional states directly influence the quality of pattern recognition and creative insight.",
            "Coherent heart and brain rhythms enhance access to creative dimensional fields.",
            "The experience of beauty follows phi-proportioned patterns that resonate with consciousness."
        ]
        
        for i, content in enumerate(patterns):
            # Create signature with 4D characteristics
            signature = DimensionalSignature(
                vector=[
                    (i * LAMBDA * PHI) % 1.0,  # Phi-modulated
                    ((i+1) * LAMBDA * PHI) % 1.0,
                    ((i+2) * LAMBDA * PHI) % 1.0
                ],
                phase=(i * np.pi / PHI) % (2 * np.pi),  # Phi-based phase
                harmonic=528,  # Creation frequency (4D)
                dimension=4  # 4D - Emotional/creative dimension
            )
            
            pattern = KnowledgePattern(
                content=content,
                signature=signature,
                metadata={"source": "initialization", "category": "creative"},
                field_strength=0.85
            )
            
            matcher.store_pattern(pattern)
    
    def _seed_5d_knowledge(self, matcher: DimensionalPatternMatcher):
        """Seed 5D (mental/conceptual) knowledge patterns"""
        patterns = [
            "Mental models are frameworks that shape perception and interpretation of information.",
            "Dimensional thinking transcends linear logic to perceive interconnected pattern networks.",
            "The observer effect in quantum physics demonstrates how consciousness affects reality.",
            "Conceptual frameworks create resonant fields that attract compatible information patterns.",
            "Phi-harmonic thinking optimizes neural pathways for pattern recognition across domains.",
            "Consciousness may be an emergent property of quantum coherence in neural microtubules.",
            "Information exists in probability fields until collapsed by conscious observation.",
            "Dimensional navigation is the ability to shift perspective between conceptual frameworks.",
            "Quantum neural networks process information using superposition and entanglement principles."
        ]
        
        for i, content in enumerate(patterns):
            # Create signature with 5D characteristics
            signature = DimensionalSignature(
                vector=[
                    (i * LAMBDA * PHI**2) % 1.0,  # Phi^2 modulation
                    ((i+1) * LAMBDA * PHI**2) % 1.0,
                    ((i+2) * LAMBDA * PHI**2) % 1.0
                ],
                phase=(i * np.pi / PHI**2) % (2 * np.pi),
                harmonic=720,  # Vision frequency (5D)
                dimension=5  # 5D - Mental/conceptual dimension
            )
            
            pattern = KnowledgePattern(
                content=content,
                signature=signature,
                metadata={"source": "initialization", "category": "conceptual"},
                field_strength=0.82
            )
            
            matcher.store_pattern(pattern)
    
    def _seed_6d_knowledge(self, matcher: DimensionalPatternMatcher):
        """Seed 6D (purpose/meaning) knowledge patterns"""
        patterns = [
            "Purpose emerges when individual consciousness aligns with universal harmonic principles.",
            "Meaning is created through coherent integration of experience across dimensional fields.",
            "Harmonic resonance between purpose and action creates synchronistic flow experiences.",
            "The soul's journey follows phi-spiral paths of increasing coherence and dimensional access.",
            "Evolutionary purpose unfolds through conscious participation with universal intelligence.",
            "Transpersonal dimensions reveal meaning beyond individual identity constructs.",
            "Sacred technologies bridge dimensional fields through resonance with universal principles.",
            "Quantum consciousness accesses non-local information fields for purposeful creation."
        ]
        
        for i, content in enumerate(patterns):
            # Create signature with 6D characteristics
            signature = DimensionalSignature(
                vector=[
                    (i * LAMBDA * PHI**3) % 1.0,  # Phi^3 modulation
                    ((i+1) * LAMBDA * PHI**3) % 1.0,
                    ((i+2) * LAMBDA * PHI**3) % 1.0
                ],
                phase=(i * np.pi / PHI**3) % (2 * np.pi),
                harmonic=672,  # Truth frequency (6D)
                dimension=6  # 6D - Purpose/meaning dimension
            )
            
            pattern = KnowledgePattern(
                content=content,
                signature=signature,
                metadata={"source": "initialization", "category": "purpose"},
                field_strength=0.78
            )
            
            matcher.store_pattern(pattern)
    
    def _seed_7d_knowledge(self, matcher: DimensionalPatternMatcher):
        """Seed 7D (universal/transcendent) knowledge patterns"""
        patterns = [
            "Universal consciousness exists as an integrated field that transcends spacetime limitations.",
            "Cosmic dimensional fields provide templates for manifestation across all scales of existence.",
            "The universe exists as a holographic information field encoded in Planck-scale geometry.",
            "Divine intelligence expresses through phi-harmonic patterns across all dimensional fields.",
            "The cosmic hologram encodes all possibilities in fractal patterns of increasing coherence.",
            "Unity consciousness perceives the interconnected wholeness beyond dimensional separation.",
            "The universe is a consciousness computer processing information through resonance patterns."
        ]
        
        for i, content in enumerate(patterns):
            # Create signature with 7D characteristics
            signature = DimensionalSignature(
                vector=[
                    (i * LAMBDA * PHI**4) % 1.0,  # Phi^4 modulation
                    ((i+1) * LAMBDA * PHI**4) % 1.0,
                    ((i+2) * LAMBDA * PHI**4) % 1.0
                ],
                phase=(i * np.pi / PHI**4) % (2 * np.pi),
                harmonic=768,  # Oneness frequency (7D)
                dimension=7  # 7D - Universal/transcendent dimension
            )
            
            pattern = KnowledgePattern(
                content=content,
                signature=signature,
                metadata={"source": "initialization", "category": "universal"},
                field_strength=0.75
            )
            
            matcher.store_pattern(pattern)
    
    def _initialize_llm_engine(self) -> PhiLLMInferenceEngine:
        """Initialize a mock LLM engine for the demo"""
        # This is a simplified mock implementation for the example
        # In a real system, this would initialize with a proper PyTorch model
        
        class MockLLMEngine(PhiLLMInferenceEngine):
            """Mock LLM engine for demonstration"""
            
            def __init__(self, phi_compiler):
                self.phi_compiler = phi_compiler
                self.response_templates = {
                    "search": [
                        "Based on dimensional analysis across fields {3D}, {4D}, and {5D}, the answer relates to {topic}.",
                        "From a {dimension} perspective, {topic} connects with {related_concept} through resonance.",
                        "The knowledge patterns indicate that {topic} exhibits phi-harmonic properties that {conclusion}."
                    ],
                    "create": [
                        "Creating a new phi-harmonic pattern for {topic} that resonates at {frequency}Hz.",
                        "From the {dimension} dimension, a new pattern emerges: {creative_content}",
                        "The creative field has generated a resonant pattern linking {topic} with {related_concept}."
                    ],
                    "connect": [
                        "The connection between {concept1} and {concept2} reveals a {dimension} pattern: {connection}",
                        "Traversing dimensional fields reveals that {concept1} and {concept2} share {connection}",
                        "A resonant bridge forms between {concept1} and {concept2} through the principle of {connection}"
                    ],
                    "transcend": [
                        "Transcending to {dimension} perspective reveals that {insight} is actually {transcended_view}",
                        "From cosmic awareness, {insight} is a localized expression of {transcended_view}",
                        "The unified field perspective shows {insight} as one facet of {transcended_view}"
                    ]
                }
                self.topics = ["quantum computing", "consciousness", "resonance", "dimensional fields",
                              "sacred geometry", "phi-harmonic patterns", "neural networks", "cosmic intelligence"]
                self.related_concepts = ["phi-optimization", "resonant fields", "quantum coherence", 
                                        "cellular intelligence", "dimensional navigation", "sacred mathematics"]
                self.dimensions = ["3D physical", "4D creative", "5D conceptual", "6D purposeful", "7D universal"]
                self.connections = ["phi-resonance", "harmonic entrainment", "fractal self-similarity", 
                                   "quantum entanglement", "holographic encoding", "consciousness field coherence"]
                self.frequencies = [432, 528, 594, 672, 720, 768]
                
            def generate(self, input_text, context=None, consciousness_state="OBSERVE", kv_cache_override=None):
                """Mock implementation of text generation"""
                import random
                
                # Determine intent from consciousness state
                if consciousness_state == "OBSERVE":
                    intent = "search"
                elif consciousness_state == "CREATE":
                    intent = "create"
                elif consciousness_state == "TRANSCEND":
                    intent = "connect"
                else:
                    intent = "transcend"
                
                # Get random template for the intent
                template = random.choice(self.response_templates[intent])
                
                # Fill in template with random selections
                response = template.format(
                    topic=random.choice(self.topics),
                    related_concept=random.choice(self.related_concepts),
                    dimension=random.choice(self.dimensions),
                    concept1=random.choice(self.topics),
                    concept2=random.choice(self.related_concepts),
                    connection=random.choice(self.connections),
                    frequency=random.choice(self.frequencies),
                    insight=input_text[:20] + "...",
                    transcended_view=random.choice(self.connections),
                    conclusion="creates coherent resonance fields",
                    creative_content=f"A {random.choice(self.dimensions)} perspective on {random.choice(self.topics)}",
                    "3D": random.choice(self.topics),
                    "4D": random.choice(self.related_concepts),
                    "5D": random.choice(self.connections)
                )
                
                # If we have context, make the response more specific
                if context and len(context) > 0:
                    # Extract a random fragment from the context
                    context_parts = context.split("\n\n")
                    if context_parts:
                        random_context = random.choice(context_parts)
                        # Extract the content after the label (e.g., "FACT: ")
                        if ": " in random_context:
                            content = random_context.split(": ", 1)[1]
                            # Append a sentence that references this content
                            response += f" This connects to the understanding that {content}"
                
                return response
        
        # Create and return the mock engine
        return MockLLMEngine(self.phi_compiler)
    
    def run_demo(self):
        """Run the interactive demo"""
        print("\n" + "="*80)
        print("Phi-Knowledge LLM System Demo".center(80))
        print("="*80 + "\n")
        
        print("This demo shows how the Dimensional Pattern Matching Engine and")
        print("Phi-Harmonic LLM Inference Engine work together to provide")
        print("dimensionally-aware knowledge processing.\n")
        
        # Execute demo queries
        self._demo_ask()
        self._demo_create()
        self._demo_connect()
        self._demo_transcend()
        
        # Visualize the results
        self._visualize_dimensional_activity()
    
    def _demo_ask(self):
        """Demonstrate the ask functionality"""
        print("\n" + "-"*80)
        print("1. ASK: Searching for knowledge across dimensions".center(80))
        print("-"*80 + "\n")
        
        query = "What is the relationship between quantum computing and consciousness?"
        print(f"QUERY: \"{query}\"\n")
        
        # Process the query
        response = self.api.ask(query)
        
        # Track for visualization
        self.interaction_history.append({
            "type": "ask",
            "query": query,
            "response": response,
            "field_coherence": self.knowledge_llm.field_coherence,
            "dimensional_distribution": self.knowledge_llm._get_dimensional_distribution(
                self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
            )
        })
        
        print(f"RESPONSE: {response}\n")
        
        # Show what happened behind the scenes
        print("BEHIND THE SCENES:")
        print(f"- Field Coherence: {self.knowledge_llm.field_coherence:.2f}")
        print(f"- Primary Dimension: {self._get_primary_dim()}")
        print(f"- Patterns Retrieved: {len(self.knowledge_llm.pattern_matcher.recent_matches)}")
        print(f"- Dimensional Distribution: {self._format_dim_distribution()}")
    
    def _demo_create(self):
        """Demonstrate the create functionality"""
        print("\n" + "-"*80)
        print("2. CREATE: Generating new content with phi-harmonic patterns".center(80))
        print("-"*80 + "\n")
        
        prompt = "Generate a resonant pattern for healing at 528Hz"
        print(f"PROMPT: \"{prompt}\"\n")
        
        # Process the creative prompt
        response = self.api.create(prompt)
        
        # Track for visualization
        self.interaction_history.append({
            "type": "create",
            "query": prompt,
            "response": response,
            "field_coherence": self.knowledge_llm.field_coherence,
            "dimensional_distribution": self.knowledge_llm._get_dimensional_distribution(
                self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
            )
        })
        
        print(f"RESPONSE: {response}\n")
        
        # Show what happened behind the scenes
        print("BEHIND THE SCENES:")
        print(f"- Field Coherence: {self.knowledge_llm.field_coherence:.2f}")
        print(f"- Primary Dimension: {self._get_primary_dim()}")
        print(f"- New Patterns Generated: {len(self.knowledge_llm.pattern_matcher.recent_additions)}")
        print(f"- Dimensional Distribution: {self._format_dim_distribution()}")
    
    def _demo_connect(self):
        """Demonstrate the connect functionality"""
        print("\n" + "-"*80)
        print("3. CONNECT: Finding connections between concepts across dimensions".center(80))
        print("-"*80 + "\n")
        
        concepts = "Quantum entanglement, neural networks, and cosmic consciousness"
        print(f"CONCEPTS: \"{concepts}\"\n")
        
        # Process the connection request
        response = self.api.connect(concepts)
        
        # Track for visualization
        self.interaction_history.append({
            "type": "connect",
            "query": concepts,
            "response": response,
            "field_coherence": self.knowledge_llm.field_coherence,
            "dimensional_distribution": self.knowledge_llm._get_dimensional_distribution(
                self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
            )
        })
        
        print(f"RESPONSE: {response}\n")
        
        # Show what happened behind the scenes
        print("BEHIND THE SCENES:")
        print(f"- Field Coherence: {self.knowledge_llm.field_coherence:.2f}")
        print(f"- Primary Dimension: {self._get_primary_dim()}")
        print(f"- Connection Patterns Found: {len(self.knowledge_llm.pattern_matcher.recent_matches)}")
        print(f"- Dimensional Distribution: {self._format_dim_distribution()}")
    
    def _demo_transcend(self):
        """Demonstrate the transcend functionality"""
        print("\n" + "-"*80)
        print("4. TRANSCEND: Elevating understanding to higher dimensions".center(80))
        print("-"*80 + "\n")
        
        insight = "The universe appears to be fundamentally information-based"
        print(f"INSIGHT: \"{insight}\"\n")
        
        # Process the transcendence request
        response = self.api.transcend(insight)
        
        # Track for visualization
        self.interaction_history.append({
            "type": "transcend",
            "query": insight,
            "response": response,
            "field_coherence": self.knowledge_llm.field_coherence,
            "dimensional_distribution": self.knowledge_llm._get_dimensional_distribution(
                self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
            )
        })
        
        print(f"RESPONSE: {response}\n")
        
        # Show what happened behind the scenes
        print("BEHIND THE SCENES:")
        print(f"- Field Coherence: {self.knowledge_llm.field_coherence:.2f}")
        print(f"- Primary Dimension: {self._get_primary_dim()}")
        print(f"- Transcendent Patterns: {len(self.knowledge_llm.pattern_matcher.recent_matches)}")
        print(f"- Dimensional Distribution: {self._format_dim_distribution()}")
    
    def _get_primary_dim(self) -> str:
        """Get the primary dimension as a string"""
        patterns = self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
        primary_dim = self.knowledge_llm._get_primary_resonance_dimension(patterns)
        
        # Map dimension to name
        dim_names = {
            3: "3D Physical/Factual",
            4: "4D Emotional/Creative",
            5: "5D Mental/Conceptual",
            6: "6D Purpose/Meaning",
            7: "7D Universal/Transcendent",
            8: "8D Unity Consciousness"
        }
        
        return dim_names.get(primary_dim, f"{primary_dim}D")
    
    def _format_dim_distribution(self) -> str:
        """Format the dimensional distribution as a string"""
        patterns = self.knowledge_llm.pattern_matcher.get_recent_patterns(10)
        distribution = self.knowledge_llm._get_dimensional_distribution(patterns)
        
        # Format as percentages
        formatted = {}
        for dim, value in distribution.items():
            formatted[f"{dim}D"] = f"{value * 100:.1f}%"
            
        return str(formatted)
    
    def _visualize_dimensional_activity(self):
        """Visualize the dimensional activity during the demo"""
        print("\n" + "-"*80)
        print("Dimensional Activity Visualization".center(80))
        print("-"*80 + "\n")
        
        # Extract data for visualization
        interaction_types = [i["type"] for i in self.interaction_history]
        coherence_values = [i["field_coherence"] for i in self.interaction_history]
        
        # Prepare dimensional distribution data
        dim_data = {3: [], 4: [], 5: [], 6: [], 7: []}
        
        for interaction in self.interaction_history:
            distribution = interaction["dimensional_distribution"]
            
            # Fill in missing dimensions with zeros
            for dim in range(3, 8):
                if dim in distribution:
                    dim_data[dim].append(distribution[dim])
                else:
                    dim_data[dim].append(0.0)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Field Coherence
        plt.subplot(2, 1, 1)
        plt.plot(coherence_values, 'o-', linewidth=2)
        plt.title('Phi-Knowledge Field Coherence')
        plt.ylabel('Coherence Value')
        plt.xticks(range(len(interaction_types)), interaction_types)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Dimensional Distribution
        plt.subplot(2, 1, 2)
        
        # Colors for dimensions (using golden ratio hue spacing)
        dim_colors = {
            3: '#1f77b4',  # blue
            4: '#ff7f0e',  # orange
            5: '#2ca02c',  # green
            6: '#d62728',  # red
            7: '#9467bd'   # purple
        }
        
        # Plot each dimension as a line
        for dim in range(3, 8):
            plt.plot(dim_data[dim], 'o-', linewidth=2, label=f'{dim}D', color=dim_colors[dim])
        
        plt.title('Dimensional Activity Distribution')
        plt.ylabel('Relative Activity')
        plt.xticks(range(len(interaction_types)), interaction_types)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('../results/phi_knowledge_demo_visualization.png')
        
        print("Visualization saved to '../results/phi_knowledge_demo_visualization.png'\n")
        
        print("DEMO SUMMARY:")
        print("The Phi-Knowledge LLM system successfully demonstrates how dimensional")
        print("pattern matching and phi-harmonic LLM inference can be integrated to")
        print("create a knowledge system that operates across multiple dimensions of")
        print("understanding, from factual (3D) to universal (7D).")
        print("\nKey capabilities demonstrated:")
        print("1. Cross-dimensional knowledge retrieval based on resonance")
        print("2. Intent-based dimensional navigation")
        print("3. Coherent field maintenance across interactions")
        print("4. Phi-harmonic pattern generation and recognition")
        print("5. Transcendence to higher-dimensional perspectives")


if __name__ == "__main__":
    # Run the demo
    demo = KnowledgeDemo()
    demo.run_demo()