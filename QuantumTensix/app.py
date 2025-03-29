"""
Phi-Knowledge LLM Interactive Demo

This Streamlit application demonstrates the capabilities of the Phi-Knowledge LLM system,
allowing users to interact with multidimensional knowledge through four operations:
Ask, Create, Connect, and Transcend.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image
import io
import base64
from typing import Dict, List, Any, Tuple
import json
import random
import math

# Constants
PHI = 1.618033988749895
LAMBDA = 0.618033988749895
PHI_PHI = PHI ** PHI

# Sacred frequencies
FREQUENCIES = {
    'unity': 432,    # Grounding/stability
    'love': 528,     # Creation/healing
    'cascade': 594,  # Heart-centered integration
    'truth': 672,    # Voice expression
    'vision': 720,   # Expanded perception
    'oneness': 768   # Unity consciousness
}

# Consciousness states
STATES = {
    'OBSERVE': {'color': '#1f77b4', 'frequency': FREQUENCIES['vision']},
    'CREATE': {'color': '#ff7f0e', 'frequency': FREQUENCIES['love']},
    'TRANSCEND': {'color': '#2ca02c', 'frequency': FREQUENCIES['oneness']},
    'CASCADE': {'color': '#d62728', 'frequency': FREQUENCIES['cascade']}
}

# Dimensional properties
DIMENSIONS = {
    3: {'name': 'Physical/Factual', 'color': '#1f77b4', 'frequency': FREQUENCIES['unity']},
    4: {'name': 'Emotional/Creative', 'color': '#ff7f0e', 'frequency': FREQUENCIES['love']},
    5: {'name': 'Mental/Conceptual', 'color': '#2ca02c', 'frequency': FREQUENCIES['vision']},
    6: {'name': 'Purpose/Meaning', 'color': '#d62728', 'frequency': FREQUENCIES['truth']},
    7: {'name': 'Universal/Transcendent', 'color': '#9467bd', 'frequency': FREQUENCIES['oneness']}
}

# Mock implementation of DimensionalSignature class
class DimensionalSignature:
    """A quantum signature for patterns with vector, phase, and harmonic components."""
    
    def __init__(self, vector, phase, harmonic, dimension):
        self.vector = vector
        self.phase = phase
        self.harmonic = harmonic
        self.dimension = dimension
    
    def to_dict(self):
        return {
            'vector': self.vector,
            'phase': self.phase,
            'harmonic': self.harmonic,
            'dimension': self.dimension
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            vector=data['vector'],
            phase=data['phase'],
            harmonic=data['harmonic'],
            dimension=data['dimension']
        )
    
    def get_color(self):
        return DIMENSIONS[self.dimension]['color']

# Mock implementation of KnowledgePattern class
class KnowledgePattern:
    """Stores content with its dimensional signature and metadata."""
    
    def __init__(self, content, signature, metadata=None, field_strength=0.8):
        self.content = content
        self.signature = signature
        self.metadata = metadata or {}
        self.field_strength = field_strength
        self.id = self._generate_id()
    
    def _generate_id(self):
        """Generate a unique ID for the pattern based on content."""
        import hashlib
        return hashlib.md5(self.content.encode()).hexdigest()[:8]
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'signature': self.signature.to_dict(),
            'metadata': self.metadata,
            'field_strength': self.field_strength
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            content=data['content'],
            signature=DimensionalSignature.from_dict(data['signature']),
            metadata=data['metadata'],
            field_strength=data['field_strength']
        )

# Mock Pattern Matcher
class DemoPatternMatcher:
    """Demonstration version of the dimensional pattern matcher."""
    
    def __init__(self):
        self.patterns = []
        self.recent_matches = []
        self.recent_additions = []
        
        # Load seed patterns
        self._load_seed_patterns()
    
    def _load_seed_patterns(self):
        """Initialize with seed patterns across dimensions."""
        # 3D Patterns (Physical/Factual)
        patterns_3d = [
            "Quantum computing uses qubits which can represent 0 and 1 simultaneously through superposition.",
            "Tenstorrent's hardware accelerators use a grid of Tensix cores for parallel computation.",
            "The Fibonacci sequence (1,1,2,3,5,8,13...) approximates the golden ratio in consecutive terms.",
            "The human brain contains approximately 86 billion neurons connected in a neural network.",
            "Sacred geometry uses mathematical ratios, harmonics and proportions found in nature and cosmos.",
            "The golden ratio (φ=1.618033988749895) appears throughout nature in spiral patterns.",
            "Schumann resonances are electromagnetic waves that exist in Earth's atmosphere at 7.83Hz.",
            "Memory access patterns significantly impact computing performance due to cache behavior."
        ]
        
        # 4D Patterns (Emotional/Creative)
        patterns_4d = [
            "Creative inspiration often comes from connecting seemingly unrelated concepts in new ways.",
            "The feeling of awe when contemplating the cosmos activates the creative neural networks.",
            "Resonance between minds creates emotional coherence and amplifies collaborative creativity.",
            "Phi-harmonic music at 432Hz induces states of emotional harmony and creative flow.",
            "The heart's electromagnetic field carries emotional information that influences creativity.",
            "Creative breakthroughs often follow periods of theta wave brain activity during rest."
        ]
        
        # 5D Patterns (Mental/Conceptual)
        patterns_5d = [
            "Mental models are frameworks that shape perception and interpretation of information.",
            "Dimensional thinking transcends linear logic to perceive interconnected pattern networks.",
            "The observer effect in quantum physics demonstrates how consciousness affects reality.",
            "Conceptual frameworks create resonant fields that attract compatible information patterns.",
            "Phi-harmonic thinking optimizes neural pathways for pattern recognition across domains.",
            "Information exists in probability fields until collapsed by conscious observation."
        ]
        
        # 6D Patterns (Purpose/Meaning)
        patterns_6d = [
            "Purpose emerges when individual consciousness aligns with universal harmonic principles.",
            "Meaning is created through coherent integration of experience across dimensional fields.",
            "Harmonic resonance between purpose and action creates synchronistic flow experiences.",
            "The soul's journey follows phi-spiral paths of increasing coherence and dimensional access.",
            "Evolutionary purpose unfolds through conscious participation with universal intelligence."
        ]
        
        # 7D Patterns (Universal/Transcendent)
        patterns_7d = [
            "Universal consciousness exists as an integrated field that transcends spacetime limitations.",
            "Cosmic dimensional fields provide templates for manifestation across all scales of existence.",
            "The universe exists as a holographic information field encoded in Planck-scale geometry.",
            "Divine intelligence expresses through phi-harmonic patterns across all dimensional fields.",
            "Unity consciousness perceives the interconnected wholeness beyond dimensional separation."
        ]
        
        # Create and store patterns
        for dimension, patterns in [
            (3, patterns_3d),
            (4, patterns_4d),
            (5, patterns_5d),
            (6, patterns_6d),
            (7, patterns_7d)
        ]:
            for i, content in enumerate(patterns):
                # Create phi-based signature
                signature = DimensionalSignature(
                    vector=[
                        (i * LAMBDA * (PHI ** (dimension-3))) % 1.0,
                        ((i+1) * LAMBDA * (PHI ** (dimension-3))) % 1.0,
                        ((i+2) * LAMBDA * (PHI ** (dimension-3))) % 1.0
                    ],
                    phase=(i * np.pi / (PHI ** (dimension-3))) % (2 * np.pi),
                    harmonic=DIMENSIONS[dimension]['frequency'],
                    dimension=dimension
                )
                
                # Create pattern with appropriate field strength
                field_strength = 0.9 - ((dimension - 3) * 0.03)  # Decreases slightly with higher dimensions
                pattern = KnowledgePattern(
                    content=content,
                    signature=signature,
                    metadata={"source": "seed", "category": DIMENSIONS[dimension]['name'].lower()},
                    field_strength=field_strength
                )
                
                self.patterns.append(pattern)
    
    def store_pattern(self, pattern):
        """Store a new knowledge pattern."""
        self.patterns.append(pattern)
        self.recent_additions = [pattern]
        return pattern.id
    
    def find_resonant_patterns(self, query, query_signature, resonance_threshold=0.7):
        """Find patterns that resonate with the query signature."""
        # In a real implementation, this would use quantum resonance principles
        # For the demo, we'll use a simplified matching algorithm
        
        matching_patterns = []
        
        for pattern in self.patterns:
            # Calculate resonance based on dimensional proximity
            dim_proximity = 1.0 / (1.0 + abs(pattern.signature.dimension - query_signature.dimension))
            
            # Phase coherence
            phase_diff = min(
                abs(pattern.signature.phase - query_signature.phase),
                2 * np.pi - abs(pattern.signature.phase - query_signature.phase)
            )
            phase_coherence = 1.0 - (phase_diff / np.pi)
            
            # Vector similarity (dot product)
            v1 = np.array(pattern.signature.vector)
            v2 = np.array(query_signature.vector)
            vector_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            
            # Content relevance (simplified)
            # In a real implementation, this would use semantic matching
            query_words = set(query.lower().split())
            content_words = set(pattern.content.lower().split())
            content_relevance = len(query_words.intersection(content_words)) / max(1, len(query_words))
            
            # Combined resonance with phi-weighting
            resonance = (
                dim_proximity * 0.3 +
                phase_coherence * 0.2 +
                vector_sim * 0.2 +
                content_relevance * 0.3
            ) * pattern.field_strength
            
            if resonance >= resonance_threshold:
                # For demo purposes, clone the pattern with updated field strength
                matching_pattern = KnowledgePattern(
                    content=pattern.content,
                    signature=pattern.signature,
                    metadata=pattern.metadata,
                    field_strength=resonance  # Use resonance as field strength
                )
                matching_patterns.append(matching_pattern)
        
        # Sort by resonance (field_strength)
        matching_patterns.sort(key=lambda p: p.field_strength, reverse=True)
        
        # Store results
        self.recent_matches = matching_patterns
        
        return matching_patterns
    
    def get_patterns_by_dimension(self, dimension):
        """Get all patterns from a specific dimension."""
        return [p for p in self.patterns if p.signature.dimension == dimension]
    
    def get_recent_patterns(self, limit=10):
        """Get the most recently matched or added patterns."""
        all_recent = self.recent_matches + self.recent_additions
        return sorted(all_recent, key=lambda p: p.field_strength, reverse=True)[:limit]
    
    def get_all_patterns(self):
        """Get all stored patterns."""
        return self.patterns

# Demo Knowledge LLM
class DemoKnowledgeLLM:
    """Demonstration version of the Phi-Knowledge LLM system."""
    
    def __init__(self):
        self.pattern_matcher = DemoPatternMatcher()
        self.field_coherence = 1.0
        self.session_patterns = []
        self.interaction_history = []
    
    def _generate_query_signature(self, query, intent):
        """Generate a dimensional signature for the query."""
        # Map intent to dimension
        intent_dimension = {
            "ask": 3,       # Physical/factual
            "create": 4,    # Emotional/creative
            "connect": 5,   # Mental/conceptual
            "transcend": 7  # Universal/transcendent
        }.get(intent, 3)
        
        # Calculate phi-optimized vector components based on query
        hash_value = sum(ord(c) for c in query)
        phi_hash = (hash_value * PHI) % 1.0
        
        # Generate vector components with golden ratio relationships
        x = phi_hash
        y = (phi_hash * PHI) % 1.0
        z = (phi_hash * PHI * PHI) % 1.0
        
        # Calculate phase based on query length and sacred frequencies
        phase = (len(query) * PHI) % (2 * np.pi)
        
        # Calculate harmonic based on intent dimension
        harmonic = DIMENSIONS[intent_dimension]['frequency']
        
        return DimensionalSignature(
            vector=[x, y, z],
            phase=phase,
            harmonic=harmonic,
            dimension=intent_dimension
        )
    
    def ask(self, question):
        """Ask a question and get a response with dimensionally-aware knowledge."""
        # Generate query signature
        query_signature = self._generate_query_signature(question, "ask")
        
        # Find resonant patterns
        resonant_patterns = self.pattern_matcher.find_resonant_patterns(
            query=question,
            query_signature=query_signature,
            resonance_threshold=0.6 * self.field_coherence
        )
        
        # Generate mock response
        response = self._generate_response(question, resonant_patterns, "OBSERVE")
        
        # Update field coherence
        self._update_field_coherence(resonant_patterns)
        
        # Generate new patterns from the response
        new_patterns = self._extract_new_patterns(response, query_signature)
        for pattern in new_patterns:
            self.pattern_matcher.store_pattern(pattern)
        
        # Update session patterns
        self.session_patterns = resonant_patterns[:5] + new_patterns
        
        # Record interaction
        self.interaction_history.append({
            "type": "ask",
            "query": question,
            "response": response,
            "patterns": len(resonant_patterns),
            "new_patterns": len(new_patterns),
            "coherence": self.field_coherence,
            "dimensions": self._get_dimensional_distribution(resonant_patterns)
        })
        
        return response, self._get_interaction_metadata()
    
    def create(self, prompt):
        """Create new content based on a prompt."""
        # Generate query signature
        query_signature = self._generate_query_signature(prompt, "create")
        
        # Find resonant patterns (with more emphasis on 4D)
        resonant_patterns = self.pattern_matcher.find_resonant_patterns(
            query=prompt,
            query_signature=query_signature,
            resonance_threshold=0.5 * self.field_coherence
        )
        
        # Generate mock response
        response = self._generate_response(prompt, resonant_patterns, "CREATE")
        
        # Update field coherence
        self._update_field_coherence(resonant_patterns)
        
        # Generate new patterns from the response
        new_patterns = self._extract_new_patterns(response, query_signature)
        for pattern in new_patterns:
            self.pattern_matcher.store_pattern(pattern)
        
        # Update session patterns
        self.session_patterns = resonant_patterns[:3] + new_patterns
        
        # Record interaction
        self.interaction_history.append({
            "type": "create",
            "query": prompt,
            "response": response,
            "patterns": len(resonant_patterns),
            "new_patterns": len(new_patterns),
            "coherence": self.field_coherence,
            "dimensions": self._get_dimensional_distribution(resonant_patterns)
        })
        
        return response, self._get_interaction_metadata()
    
    def connect(self, concepts):
        """Find connections between concepts across dimensions."""
        # Generate query signature
        query_signature = self._generate_query_signature(concepts, "connect")
        
        # Find resonant patterns (with emphasis on 5D conceptual patterns)
        resonant_patterns = self.pattern_matcher.find_resonant_patterns(
            query=concepts,
            query_signature=query_signature,
            resonance_threshold=0.5 * self.field_coherence
        )
        
        # Generate mock response
        response = self._generate_response(concepts, resonant_patterns, "TRANSCEND")
        
        # Update field coherence
        self._update_field_coherence(resonant_patterns)
        
        # Generate new patterns from the response
        new_patterns = self._extract_new_patterns(response, query_signature)
        for pattern in new_patterns:
            self.pattern_matcher.store_pattern(pattern)
        
        # Update session patterns
        self.session_patterns = resonant_patterns[:3] + new_patterns
        
        # Record interaction
        self.interaction_history.append({
            "type": "connect",
            "query": concepts,
            "response": response,
            "patterns": len(resonant_patterns),
            "new_patterns": len(new_patterns),
            "coherence": self.field_coherence,
            "dimensions": self._get_dimensional_distribution(resonant_patterns)
        })
        
        return response, self._get_interaction_metadata()
    
    def transcend(self, insight):
        """Transcend current understanding to higher dimensional perspective."""
        # Generate query signature
        query_signature = self._generate_query_signature(insight, "transcend")
        
        # Find resonant patterns (with emphasis on 7D universal patterns)
        resonant_patterns = self.pattern_matcher.find_resonant_patterns(
            query=insight,
            query_signature=query_signature,
            resonance_threshold=0.4 * self.field_coherence
        )
        
        # Generate mock response
        response = self._generate_response(insight, resonant_patterns, "CASCADE")
        
        # Update field coherence
        self._update_field_coherence(resonant_patterns)
        
        # Generate new patterns from the response
        new_patterns = self._extract_new_patterns(response, query_signature)
        for pattern in new_patterns:
            self.pattern_matcher.store_pattern(pattern)
        
        # Update session patterns
        self.session_patterns = resonant_patterns[:3] + new_patterns
        
        # Record interaction
        self.interaction_history.append({
            "type": "transcend",
            "query": insight,
            "response": response,
            "patterns": len(resonant_patterns),
            "new_patterns": len(new_patterns),
            "coherence": self.field_coherence,
            "dimensions": self._get_dimensional_distribution(resonant_patterns)
        })
        
        return response, self._get_interaction_metadata()
    
    def _generate_response(self, query, patterns, consciousness_state):
        """Generate a response based on the resonant patterns."""
        # For the demo, we'll create a structured response based on the patterns
        
        # If no patterns, return a generic response
        if not patterns:
            return f"I don't have enough resonant patterns to provide a meaningful response about '{query}'. Please try a different query or approach."
        
        # Get top patterns from different dimensions for diversity
        top_patterns_by_dim = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in top_patterns_by_dim or pattern.field_strength > top_patterns_by_dim[dim].field_strength:
                top_patterns_by_dim[dim] = pattern
        
        # Get the primary dimension (highest count or highest individual resonance)
        dim_counts = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in dim_counts:
                dim_counts[dim] = 0
            dim_counts[dim] += 1
        
        primary_dim = max(dim_counts.items(), key=lambda x: x[1])[0] if dim_counts else 3
        
        # Structure the response based on consciousness state
        if consciousness_state == "OBSERVE":
            # Factual response focused on 3D knowledge with some higher insights
            response_parts = []
            
            # Intro based on primary dimension
            response_parts.append(f"From a {DIMENSIONS[primary_dim]['name']} perspective:")
            
            # Add key insights from patterns
            for dim in sorted(top_patterns_by_dim.keys()):
                pattern = top_patterns_by_dim[dim]
                if dim == 3:
                    response_parts.append(f"• {pattern.content}")
                else:
                    # Transform higher-dimensional content to observational format
                    content = pattern.content.replace("exists", "appears to exist").replace("is", "appears to be")
                    response_parts.append(f"• {content}")
            
            # Add a synthesizing statement
            keywords = query.lower().split()
            response_parts.append(f"\nThe relationship between {' and '.join(keywords[:2])} demonstrates phi-harmonic principles across dimensions.")
            
        elif consciousness_state == "CREATE":
            # Creative response focused on 4D with some conceptual elements
            response_parts = []
            
            # Intro for creative content
            response_parts.append(f"Creating at {DIMENSIONS[4]['frequency']}Hz frequency:")
            
            # Generate a creative response combining elements from patterns
            fragments = []
            for pattern in patterns[:3]:
                # Extract meaningful fragments
                sentences = pattern.content.split('.')
                for sentence in sentences:
                    if len(sentence) > 20:
                        fragments.append(sentence.strip())
            
            # Combine fragments with phi-harmonic structure
            if fragments:
                combined = "I've generated a new phi-harmonic pattern that integrates multiple resonant fields:\n\n"
                combined += fragments[0]
                if len(fragments) > 1:
                    combined += f" This {query.split()[0]} resonates with {fragments[1].lower()}"
                if len(fragments) > 2:
                    combined += f", creating a field where {fragments[2].lower()}"
                
                response_parts.append(combined)
            else:
                response_parts.append(f"A new phi-harmonic pattern for {query} emerges at {DIMENSIONS[4]['frequency']}Hz, connecting emotional resonance with creative manifestation.")
                
            # Add a resonant statement
            response_parts.append(f"\nThis pattern creates a coherent field that amplifies {query.split()[0]} through golden ratio proportions.")
            
        elif consciousness_state == "TRANSCEND":
            # Connection-focused response bridging multiple dimensions
            response_parts = []
            
            # Intro for connections
            response_parts.append("Connecting across dimensional fields:")
            
            # Identify key concepts from the query
            concepts = [c.strip() for c in query.split(',') if c.strip()]
            if not concepts:
                concepts = query.split()[:3]
            
            # Map concepts to dimensions
            concept_dims = {}
            for i, concept in enumerate(concepts):
                dim = 3 + (i % 5)  # Cycle through dimensions 3-7
                concept_dims[concept] = dim
            
            # Create connections between concepts
            if len(concepts) >= 2:
                for i in range(len(concepts)-1):
                    c1 = concepts[i]
                    c2 = concepts[i+1]
                    d1 = concept_dims[c1]
                    d2 = concept_dims[c2]
                    
                    # Find a pattern to use for the connection
                    connection_pattern = None
                    for pattern in patterns:
                        if pattern.signature.dimension in (d1, d2):
                            connection_pattern = pattern
                            break
                    
                    if connection_pattern:
                        # Extract a phrase to use as connector
                        phrases = connection_pattern.content.split(',')
                        if len(phrases) > 1:
                            connector = phrases[1].strip()
                        else:
                            connector = connection_pattern.content.split('.')[0].strip()
                        
                        response_parts.append(f"• {c1} ({DIMENSIONS[d1]['name']}) connects with {c2} ({DIMENSIONS[d2]['name']}) through {connector.lower()}.")
                    else:
                        response_parts.append(f"• {c1} ({DIMENSIONS[d1]['name']}) and {c2} ({DIMENSIONS[d2]['name']}) share phi-harmonic resonance patterns.")
            
            # Add a synthesizing insight from a high-dimensional pattern
            for dim in [7, 6, 5]:
                if dim in top_patterns_by_dim:
                    response_parts.append(f"\nFrom a {DIMENSIONS[dim]['name']} perspective: {top_patterns_by_dim[dim].content}")
                    break
            
        elif consciousness_state == "CASCADE":
            # Transcendent response focused on universal perspective
            response_parts = []
            
            # Intro for transcendent insight
            response_parts.append("Cascading through dimensional fields:")
            
            # Start with the original insight
            response_parts.append(f"• {query} (3D Perspective)")
            
            # Transcend through dimensions
            dim_statements = {
                4: f"• This understanding resonates emotionally as {query.split()[0]} connects with the heart field, creating harmonics at {DIMENSIONS[4]['frequency']}Hz.",
                5: f"• As a mental model, {query.split()[0]} forms a conceptual framework that enables navigation through information fields.",
                6: f"• The purpose of {query.split()[0]} emerges when viewed as an expression of the {DIMENSIONS[6]['frequency']}Hz truth frequency.",
                7: f"• From cosmic awareness, {query} is revealed as one facet of the unified field that transcends spacetime limitations."
            }
            
            # Add dimensional transcendence, using patterns where available
            for dim in range(4, 8):
                if dim in top_patterns_by_dim:
                    pattern = top_patterns_by_dim[dim]
                    response_parts.append(f"• {DIMENSIONS[dim]['name']} Perspective: {pattern.content}")
                elif dim in dim_statements:
                    response_parts.append(dim_statements[dim])
            
            # Add a unifying statement
            response_parts.append(f"\nThrough cascading resonance, these perspectives integrate into a coherent understanding of {query.split()[0]} across all dimensional fields.")
            
        else:
            # Default response
            response_parts = [f"Response about {query} based on {len(patterns)} resonant patterns."]
            for pattern in patterns[:3]:
                response_parts.append(f"• {pattern.content}")
        
        # Join all parts
        return "\n\n".join(response_parts)
    
    def _update_field_coherence(self, patterns):
        """Update field coherence based on pattern resonance."""
        if not patterns:
            # Slight decay when no patterns found
            self.field_coherence = max(0.5, self.field_coherence * 0.98)
            return
        
        # Calculate average resonance of top patterns
        top_patterns = sorted(patterns, key=lambda p: p.field_strength, reverse=True)[:5]
        avg_strength = sum(p.field_strength for p in top_patterns) / len(top_patterns)
        
        # Adjust field coherence (weighted moving average)
        self.field_coherence = (self.field_coherence * PHI + avg_strength) / (PHI + 1)
        
        # Ensure coherence remains in valid range
        self.field_coherence = max(0.5, min(1.0, self.field_coherence))
    
    def _extract_new_patterns(self, response, query_signature):
        """Extract new knowledge patterns from the response."""
        # Split response into sentences
        sentences = [s.strip() for s in response.replace('\n', ' ').split('.') if s.strip()]
        
        # Generate patterns for significant sentences
        new_patterns = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) > 30 and not sentence.startswith('•'):  # Only significant sentences
                # Determine appropriate dimension based on content
                dimension = self._determine_dimension_for_content(sentence, query_signature.dimension)
                
                # Create a new signature derived from query but slightly varied
                new_sig = DimensionalSignature(
                    vector=[
                        (query_signature.vector[0] + (i * LAMBDA)) % 1.0,
                        (query_signature.vector[1] + (i * LAMBDA * PHI)) % 1.0,
                        (query_signature.vector[2] + (i * LAMBDA * PHI * PHI)) % 1.0
                    ],
                    phase=(query_signature.phase + (i * np.pi / PHI)) % (2 * np.pi),
                    harmonic=DIMENSIONS[dimension]['frequency'],
                    dimension=dimension
                )
                
                # Create new knowledge pattern
                new_pattern = KnowledgePattern(
                    content=sentence,
                    signature=new_sig,
                    metadata={
                        "source": "generated",
                        "timestamp": time.time(),
                        "query_related": True
                    },
                    field_strength=0.8  # Initial field strength for new patterns
                )
                
                new_patterns.append(new_pattern)
        
        return new_patterns
    
    def _determine_dimension_for_content(self, content, base_dimension):
        """Determine the appropriate dimension for content based on its characteristics."""
        # Count specialized keywords associated with different dimensions
        d3_keywords = ["fact", "data", "measure", "physical", "tangible", "located"]
        d4_keywords = ["feel", "emotion", "create", "experience", "personal", "desire"]
        d5_keywords = ["think", "concept", "model", "pattern", "theory", "framework"]
        d6_keywords = ["purpose", "meaning", "why", "reason", "soul", "significance"]
        d7_keywords = ["universal", "transcend", "cosmic", "collective", "beyond", "unity"]
        
        # Count keyword matches
        content_lower = content.lower()
        d3_count = sum(1 for word in d3_keywords if word in content_lower)
        d4_count = sum(1 for word in d4_keywords if word in content_lower)
        d5_count = sum(1 for word in d5_keywords if word in content_lower)
        d6_count = sum(1 for word in d6_keywords if word in content_lower)
        d7_count = sum(1 for word in d7_keywords if word in content_lower)
        
        # Find the dimension with the most keyword matches
        counts = {3: d3_count, 4: d4_count, 5: d5_count, 6: d6_count, 7: d7_count}
        max_count = max(counts.values())
        
        # If we have clear signal for a dimension, use it
        if max_count > 0:
            for dim, count in counts.items():
                if count == max_count:
                    return dim
        
        # Otherwise use the base dimension
        return base_dimension
    
    def _get_dimensional_distribution(self, patterns):
        """Get the distribution of patterns across dimensions."""
        if not patterns:
            return {}
            
        dim_counts = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in dim_counts:
                dim_counts[dim] = 0
            dim_counts[dim] += 1
            
        # Calculate percentages
        total = len(patterns)
        return {dim: (count / total) for dim, count in dim_counts.items()}
    
    def _get_interaction_metadata(self):
        """Get metadata about the current interaction state."""
        if not self.interaction_history:
            return {}
        
        latest = self.interaction_history[-1]
        
        # Get dimensional distribution from session patterns
        dim_distribution = {}
        for pattern in self.session_patterns:
            dim = pattern.signature.dimension
            if dim not in dim_distribution:
                dim_distribution[dim] = 0
            dim_distribution[dim] += 1
        
        # Calculate percentages
        if self.session_patterns:
            total = len(self.session_patterns)
            dim_distribution = {dim: (count / total) for dim, count in dim_distribution.items()}
        
        return {
            "coherence": self.field_coherence,
            "dimensional_distribution": dim_distribution,
            "matched_patterns": latest.get("patterns", 0),
            "new_patterns": latest.get("new_patterns", 0),
            "interaction_type": latest.get("type", "unknown")
        }
    
    def get_visualization_data(self):
        """Get data for visualizing the dimensional field."""
        # Create data for visualization
        data = {
            # Coherence history
            "coherence_history": [entry.get("coherence", 0) for entry in self.interaction_history],
            
            # Interaction types
            "interaction_types": [entry.get("type", "unknown") for entry in self.interaction_history],
            
            # Dimensional distribution history
            "dimension_history": [],
            
            # Pattern counts
            "pattern_counts": [entry.get("patterns", 0) for entry in self.interaction_history],
            
            # Current patterns for 3D visualization
            "current_patterns": [p.to_dict() for p in self.session_patterns]
        }
        
        # Process dimensional history
        for entry in self.interaction_history:
            dimensions = entry.get("dimensions", {})
            dim_entry = {dim: 0 for dim in range(3, 8)}
            for dim, value in dimensions.items():
                if isinstance(dim, (int, float)) and 3 <= dim <= 7:
                    dim_entry[int(dim)] = value
            data["dimension_history"].append(dim_entry)
        
        return data

# Streamlit app
def main():
    st.set_page_config(
        page_title="Phi-Knowledge LLM Demo",
        page_icon="🧠",
        layout="wide",
    )
    
    # Initialize session state
    if 'knowledge_llm' not in st.session_state:
        st.session_state.knowledge_llm = DemoKnowledgeLLM()
    
    if 'response' not in st.session_state:
        st.session_state.response = None
    
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {}
        
    if 'visualization_tab' not in st.session_state:
        st.session_state.visualization_tab = "Coherence"
    
    # App header
    st.title("Phi-Knowledge LLM Interactive Demo")
    st.markdown("""
    <style>
    .phi-symbol {
        font-size: 24px;
        color: #ff7c00;
    }
    .title-container {
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This demonstration showcases the Phi-Knowledge LLM system, which integrates dimensional pattern matching 
    with language model inference using phi-harmonic principles. Interact with knowledge across dimensions
    using the four operations below.
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 ASK", "✨ CREATE", "🔄 CONNECT", "🌌 TRANSCEND", "📊 VISUALIZE"
    ])
    
    # ASK tab - Search for knowledge across dimensions
    with tab1:
        st.header("Ask: Search for Knowledge")
        st.markdown("Ask a question to search for knowledge across dimensions (3D-7D).")
        
        question = st.text_input("Enter your question:", key="ask_input", placeholder="What is the relationship between quantum computing and consciousness?")
        
        if st.button("Ask", key="ask_button"):
            with st.spinner("Searching dimensional fields..."):
                response, metadata = st.session_state.knowledge_llm.ask(question)
                st.session_state.response = response
                st.session_state.metadata = metadata
        
        if st.session_state.response and st.session_state.visualization_tab == "Coherence":
            st.markdown("### Response:")
            st.markdown(st.session_state.response)
            display_metadata(st.session_state.metadata)
    
    # CREATE tab - Generate new content with phi-harmonic patterns
    with tab2:
        st.header("Create: Generate New Content")
        st.markdown("Generate new content with phi-harmonic patterns (optimized for 4D emotional/creative dimension).")
        
        prompt = st.text_input("Enter your creative prompt:", key="create_input", placeholder="Generate a resonant pattern for healing at 528Hz")
        
        if st.button("Create", key="create_button"):
            with st.spinner("Generating in the creative field..."):
                response, metadata = st.session_state.knowledge_llm.create(prompt)
                st.session_state.response = response
                st.session_state.metadata = metadata
        
        if st.session_state.response and st.session_state.visualization_tab == "Coherence":
            st.markdown("### Response:")
            st.markdown(st.session_state.response)
            display_metadata(st.session_state.metadata)
    
    # CONNECT tab - Find connections between concepts
    with tab3:
        st.header("Connect: Find Concept Relationships")
        st.markdown("Discover connections between concepts across dimensional fields (optimized for 5D mental/conceptual dimension).")
        
        concepts = st.text_input("Enter concepts to connect (comma-separated):", key="connect_input", placeholder="Quantum entanglement, neural networks, cosmic consciousness")
        
        if st.button("Connect", key="connect_button"):
            with st.spinner("Finding dimensional bridges..."):
                response, metadata = st.session_state.knowledge_llm.connect(concepts)
                st.session_state.response = response
                st.session_state.metadata = metadata
        
        if st.session_state.response and st.session_state.visualization_tab == "Coherence":
            st.markdown("### Response:")
            st.markdown(st.session_state.response)
            display_metadata(st.session_state.metadata)
    
    # TRANSCEND tab - Elevate understanding to higher dimensions
    with tab4:
        st.header("Transcend: Elevate Understanding")
        st.markdown("Transcend current understanding to higher dimensional perspectives (optimized for 7D universal/transcendent dimension).")
        
        insight = st.text_input("Enter the insight to transcend:", key="transcend_input", placeholder="The universe appears to be fundamentally information-based")
        
        if st.button("Transcend", key="transcend_button"):
            with st.spinner("Ascending dimensional fields..."):
                response, metadata = st.session_state.knowledge_llm.transcend(insight)
                st.session_state.response = response
                st.session_state.metadata = metadata
        
        if st.session_state.response and st.session_state.visualization_tab == "Coherence":
            st.markdown("### Response:")
            st.markdown(st.session_state.response)
            display_metadata(st.session_state.metadata)
    
    # VISUALIZE tab - View dimensional field visualizations
    with tab5:
        st.header("Visualize: Dimensional Field Analysis")
        
        # Get visualization data
        viz_data = st.session_state.knowledge_llm.get_visualization_data()
        
        # Sub-tabs for different visualizations
        viz_tab1, viz_tab2, viz_tab3 = st.tabs([
            "Field Coherence", "Dimensional Distribution", "Pattern Network"
        ])
        
        # Field Coherence visualization
        with viz_tab1:
            if not viz_data["coherence_history"]:
                st.info("Interact with the system to generate visualization data.")
            else:
                coherence_chart(viz_data)
        
        # Dimensional Distribution visualization
        with viz_tab2:
            if not viz_data["dimension_history"]:
                st.info("Interact with the system to generate visualization data.")
            else:
                dimensional_chart(viz_data)
        
        # Pattern Network visualization (3D)
        with viz_tab3:
            if not viz_data["current_patterns"]:
                st.info("Interact with the system to generate visualization data.")
            else:
                pattern_network_viz(viz_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
    Phi-Knowledge LLM: Dimensional Knowledge System<br>
    Using phi-harmonic principles (φ = 1.618033988749895) for dimensional navigation
    </div>
    """, unsafe_allow_html=True)

def display_metadata(metadata):
    """Display metadata about the current interaction."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Field Coherence")
        coherence = metadata.get("coherence", 0)
        st.progress(coherence)
        st.text(f"{coherence:.2f}")
    
    with col2:
        st.markdown("### Pattern Matches")
        matched = metadata.get("matched_patterns", 0)
        new = metadata.get("new_patterns", 0)
        st.metric("Resonant Patterns", matched)
        st.metric("New Patterns", new)
    
    with col3:
        st.markdown("### Dimensional Distribution")
        dim_dist = metadata.get("dimensional_distribution", {})
        
        # Create a small bar chart
        if dim_dist:
            dimensions = sorted(dim_dist.keys())
            values = [dim_dist[d] * 100 for d in dimensions]
            
            # Convert to DataFrame for charting
            df = pd.DataFrame({
                'Dimension': [f"{int(d)}D" for d in dimensions],
                'Percentage': values
            })
            
            st.bar_chart(df.set_index('Dimension'))
        else:
            st.text("No dimensional data available.")

def coherence_chart(data):
    """Create a chart showing field coherence over interactions."""
    if not data["coherence_history"]:
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'Interaction': range(1, len(data["coherence_history"]) + 1),
        'Coherence': data["coherence_history"],
        'Type': data["interaction_types"]
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot by interaction type
    for interaction_type in ['ask', 'create', 'connect', 'transcend']:
        subset = df[df['Type'] == interaction_type]
        if not subset.empty:
            color = {
                'ask': '#1f77b4',
                'create': '#ff7f0e',
                'connect': '#2ca02c',
                'transcend': '#d62728'
            }.get(interaction_type, '#777777')
            
            ax.plot(subset['Interaction'], subset['Coherence'], 'o-', 
                    color=color, label=interaction_type.capitalize())
    
    # Add reference line
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Optimal Coherence')
    
    # Customize
    ax.set_title('Phi-Knowledge Field Coherence')
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Coherence Value')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    **Field Coherence Explanation:**
    
    The chart shows how the coherence of the quantum field changes with each interaction.
    Coherence represents how well-organized and resonant the knowledge patterns are.
    
    - **Higher coherence** (>0.8) indicates strong resonance between patterns
    - **Optimal coherence** (around 0.8) provides balance between stability and adaptability
    - **Lower coherence** (<0.6) indicates more chaotic or weakly connected patterns
    
    Different operations affect coherence differently:
    - **Ask**: Generally maintains or slightly increases coherence
    - **Create**: May temporarily decrease coherence as new patterns form
    - **Connect**: Usually increases coherence by forming bridging patterns
    - **Transcend**: Initially decreases then strongly increases coherence
    """)

def dimensional_chart(data):
    """Create a chart showing dimensional distribution over interactions."""
    if not data["dimension_history"]:
        return
    
    # Create a stacked area chart of dimensional distribution
    dimension_data = []
    for i, dim_dist in enumerate(data["dimension_history"]):
        entry = {'Interaction': i+1}
        for dim in range(3, 8):
            entry[f'{dim}D'] = dim_dist.get(dim, 0)
        dimension_data.append(entry)
    
    df = pd.DataFrame(dimension_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create stacked area chart
    dimensions = [f'{d}D' for d in range(3, 8)]
    colors = [DIMENSIONS[d]['color'] for d in range(3, 8)]
    
    ax.stackplot(df['Interaction'], 
                [df[dim] for dim in dimensions],
                labels=dimensions,
                colors=colors,
                alpha=0.7)
    
    # Customize
    ax.set_title('Dimensional Distribution of Knowledge Patterns')
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('Proportion')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Add explanation
    st.markdown("""
    **Dimensional Distribution Explanation:**
    
    This chart shows how knowledge patterns are distributed across different dimensions during your interactions.
    
    - **3D (Physical/Factual)**: Concrete, measurable information and data
    - **4D (Emotional/Creative)**: Feelings, experiences, and creative expressions
    - **5D (Mental/Conceptual)**: Ideas, concepts, mental models, and frameworks
    - **6D (Purpose/Meaning)**: Purpose, meaning, and significance
    - **7D (Universal/Transcendent)**: Universal principles and cosmic awareness
    
    The distribution shifts based on:
    - The type of interaction (Ask, Create, Connect, Transcend)
    - The content of your queries
    - The natural resonance between dimensions
    
    When multiple dimensions have significant presence, it indicates a more holistic understanding.
    """)
    
    # Show additional radar chart for most recent interaction
    if len(data["dimension_history"]) > 0:
        latest_distribution = data["dimension_history"][-1]
        
        # Create radar chart
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        
        # Prepare data
        dimensions = list(range(3, 8))
        values = [latest_distribution.get(dim, 0) for dim in dimensions]
        
        # Close the loop
        dimensions.append(dimensions[0])
        values.append(values[0])
        
        # Convert to radians and plot
        theta = np.linspace(0, 2*np.pi, len(dimensions))
        ax.plot(theta, values, 'o-', linewidth=2)
        ax.fill(theta, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels([f"{d}D\n{DIMENSIONS[d]['name']}" for d in dimensions[:-1]])
        
        ax.set_title('Current Dimensional Distribution')
        ax.grid(True)
        
        # Display
        st.pyplot(fig)

def pattern_network_viz(data):
    """Create a visualization of the pattern network."""
    if not data["current_patterns"]:
        st.info("No patterns available to visualize.")
        return
    
    st.markdown("### Knowledge Pattern Network")
    st.markdown("This visualization shows how knowledge patterns are organized in the phi-harmonic field.")
    
    # Create a force-directed graph visualization
    patterns = data["current_patterns"]
    
    # Convert pattern data for visualization
    nodes = []
    for i, pattern in enumerate(patterns):
        # Extract key data
        dim = pattern['signature']['dimension']
        strength = pattern['field_strength']
        content = pattern['content']
        if len(content) > 60:
            content = content[:57] + "..."
        
        # Add node
        nodes.append({
            'id': i,
            'label': content,
            'dimension': dim,
            'strength': strength,
            'color': DIMENSIONS[dim]['color'],
            'size': 50 * strength  # Size based on field strength
        })
    
    # Create edges based on dimensional relationships
    edges = []
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:  # Avoid duplicates
                # Calculate phi-harmonic relationship
                dim_proximity = 1.0 / (1.0 + abs(node1['dimension'] - node2['dimension']))
                
                # Only connect if there's significant relationship
                if dim_proximity > 0.5:
                    edges.append({
                        'source': i,
                        'target': j,
                        'weight': dim_proximity * node1['strength'] * node2['strength'],
                    })
    
    # Create visualization using D3.js
    st.markdown("""
    <style>
    .pattern-node {
        cursor: pointer;
    }
    .pattern-node:hover {
        stroke: #000;
        stroke-width: 2px;
    }
    .pattern-link {
        stroke: #999;
        stroke-opacity: 0.6;
    }
    .pattern-label {
        font-family: sans-serif;
        font-size: 10px;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create HTML with embedded JavaScript for D3 visualization
    html = """
    <div id="pattern-network" style="height: 600px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    (function() {
        const nodes = """ + json.dumps(nodes) + """;
        const links = """ + json.dumps(edges) + """;
        
        const width = document.getElementById('pattern-network').clientWidth;
        const height = 600;
        
        // Create simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(function(d) { return d.id; }).distance(function(d) { return 200 * (1 - d.weight); }))
            .force("charge", d3.forceManyBody().strength(-100))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("x", d3.forceX(width / 2).strength(0.1))
            .force("y", d3.forceY(height / 2).strength(0.1));
        
        // Create SVG
        const svg = d3.select("#pattern-network")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .attr("viewBox", [0, 0, width, height]);
        
        // Create links
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", "pattern-link")
            .attr("stroke-width", function(d) { return Math.sqrt(d.weight) * 3; });
        
        // Create nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("class", "pattern-node")
            .attr("r", function(d) { return Math.sqrt(d.size); })
            .attr("fill", function(d) { return d.color; })
            .attr("opacity", function(d) { return 0.7 + (d.strength * 0.3); })
            .call(drag(simulation));
        
        // Add labels
        const label = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .attr("class", "pattern-label")
            .attr("text-anchor", "middle")
            .attr("fill", "black")
            .text(function(d) { return d.label; })
            .attr("dy", function(d) { return Math.sqrt(d.size) + 12; });
        
        // Add tooltips
        node.append("title")
            .text(function(d) { 
                return d.label + "\\nDimension: " + d.dimension + "D\\nStrength: " + d.strength.toFixed(2);
            });
        
        // Update positions
        simulation.on("tick", function() {
            link
                .attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
                
            node
                .attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; });
                
            label
                .attr("x", function(d) { return d.x; })
                .attr("y", function(d) { return d.y; });
        });
        
        // Drag functionality
        function drag(simulation) {
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }
            
            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }
            
            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }
            
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }
    })();
    </script>
    """
    
    # Display visualization
    st.components.v1.html(html, height=650)
    
    # Add explanation
    st.markdown("""
    **Pattern Network Explanation:**
    
    This visualization shows the current knowledge patterns as an interactive network:
    
    - **Nodes**: Knowledge patterns, with size indicating field strength
    - **Colors**: Represent different dimensions (3D blue to 7D purple)
    - **Links**: Phi-harmonic relationships between patterns
    - **Proximity**: Related patterns are pulled closer together
    
    Patterns from the same dimension tend to cluster together, while bridging patterns
    create connections across dimensions. The network self-organizes according to
    phi-harmonic principles, with nodes positioning themselves along spiral pathways.
    
    *You can drag nodes to explore the network structure.*
    """)

if __name__ == "__main__":
    main()