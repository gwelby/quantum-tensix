import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from phi_llm_inference import PhiLLMInferenceEngine, PhiHarmonicAttention, PhiHarmonicKVCache
from dimensional_pattern_matching import DimensionalPatternMatcher, KnowledgePattern, DimensionalSignature, IntentMapping

# Sacred constants
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
    'oneness': 768,  # Unity consciousness
}

class PhiKnowledgeLLM:
    """
    Integrates the Dimensional Pattern Matching Engine with the Phi-Harmonic LLM Inference Engine,
    allowing the LLM to access and reason with dimensionally-stored knowledge patterns.
    """
    
    def __init__(
        self, 
        llm_engine: PhiLLMInferenceEngine,
        pattern_matcher: DimensionalPatternMatcher,
        base_frequency: int = FREQUENCIES['vision'],
        dimensional_weighting: Dict[int, float] = None
    ):
        """
        Initialize the PhiKnowledgeLLM system.
        
        Args:
            llm_engine: The phi-optimized LLM inference engine
            pattern_matcher: The dimensional pattern matching engine
            base_frequency: Base frequency for resonance (default: 720Hz - vision)
            dimensional_weighting: Custom weights for different dimensions
        """
        self.llm_engine = llm_engine
        self.pattern_matcher = pattern_matcher
        self.base_frequency = base_frequency
        
        # Default dimensional weighting if not provided
        if dimensional_weighting is None:
            self.dimensional_weighting = {
                3: 1.0,      # Physical/factual (standard)
                4: PHI,      # Emotional/creative (phi)
                5: PHI*PHI,  # Mental/conceptual (phi^2)
                6: PHI_PHI,  # Spiritual/purposeful (phi^phi)
                7: PHI**3,   # Universal/transcendent (phi^3)
                8: PHI**4    # Unified/cosmic (phi^4)
            }
        else:
            self.dimensional_weighting = dimensional_weighting
            
        # Initialize the intent mapping
        self.intent_mapper = IntentMapping()
        
        # KV Cache integration for dimensional resonance
        self.dimensional_kv_cache = {}
        
        # Track field coherence
        self.field_coherence = 1.0
        
    def process_query(
        self, 
        query: str, 
        intent: str = "search", 
        consciousness_state: str = "OBSERVE",
        context_patterns: List[KnowledgePattern] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query through both the pattern matcher and LLM engine.
        
        Args:
            query: The user's query text
            intent: The query intent ("search", "create", "connect", "transcend")
            consciousness_state: The consciousness state to operate in
            context_patterns: Optional list of knowledge patterns for context
            
        Returns:
            Tuple of (response text, metadata about the processing)
        """
        # Step 1: Extract dimensional signature from the query intent
        query_signature = self._generate_query_signature(query, intent)
        
        # Step 2: Find resonant knowledge patterns across dimensions
        resonant_patterns = self.pattern_matcher.find_resonant_patterns(
            query, 
            query_signature, 
            resonance_threshold=0.7 * self.field_coherence
        )
        
        # Step 3: Add context patterns if provided
        if context_patterns:
            resonant_patterns.extend(context_patterns)
        
        # Step 4: Convert knowledge patterns to LLM context
        llm_context = self._patterns_to_llm_context(resonant_patterns)
        
        # Step 5: Update the dimensional KV cache with pattern resonances
        self._update_dimensional_kv_cache(resonant_patterns, query_signature)
        
        # Step 6: Process through the LLM engine with dimensionally-aware attention
        llm_response = self.llm_engine.generate(
            input_text=query,
            context=llm_context,
            consciousness_state=consciousness_state,
            kv_cache_override=self.dimensional_kv_cache
        )
        
        # Step 7: Extract new knowledge patterns from the response
        new_patterns = self._extract_patterns_from_response(llm_response, query_signature)
        
        # Step 8: Store new patterns in the pattern matcher
        for pattern in new_patterns:
            self.pattern_matcher.store_pattern(pattern)
        
        # Step 9: Update field coherence based on resonance quality
        self._update_field_coherence(resonant_patterns)
        
        # Prepare metadata about the processing
        metadata = {
            "resonant_patterns_count": len(resonant_patterns),
            "new_patterns_generated": len(new_patterns),
            "field_coherence": self.field_coherence,
            "dimensional_distribution": self._get_dimensional_distribution(resonant_patterns),
            "primary_resonance_dimension": self._get_primary_resonance_dimension(resonant_patterns)
        }
        
        return llm_response, metadata
    
    def _generate_query_signature(self, query: str, intent: str) -> DimensionalSignature:
        """Generate a dimensional signature for the query based on its intent."""
        # Get dimensional coordinates for the intent
        intent_dim = self.intent_mapper.get_coordinates(intent)
        
        # Calculate phi-optimized vector components based on query
        hash_value = sum(ord(c) for c in query)
        phi_hash = (hash_value * PHI) % 1.0
        
        # Generate vector components with golden ratio relationships
        x = phi_hash
        y = (phi_hash * PHI) % 1.0
        z = (phi_hash * PHI * PHI) % 1.0
        
        # Calculate phase based on query length and sacred frequencies
        phase = (len(query) * PHI) % (2 * np.pi)
        
        # Calculate harmonic based on intent dimension and base frequency
        harmonic = self.base_frequency * (intent_dim / 5.0)
        
        return DimensionalSignature(
            vector=[x, y, z],
            phase=phase,
            harmonic=harmonic,
            dimension=intent_dim
        )
    
    def _patterns_to_llm_context(self, patterns: List[KnowledgePattern]) -> str:
        """Convert knowledge patterns to text context for the LLM."""
        # Sort patterns by resonance strength
        sorted_patterns = sorted(patterns, key=lambda p: p.field_strength, reverse=True)
        
        # Build context string with phi-harmonic spacing
        context_parts = []
        
        for i, pattern in enumerate(sorted_patterns):
            # Apply dimensional weighting
            dim_weight = self.dimensional_weighting.get(pattern.signature.dimension, 1.0)
            weighted_strength = pattern.field_strength * dim_weight
            
            # Only include patterns with sufficient weight
            if weighted_strength > 0.5:
                # Format based on dimension
                if pattern.signature.dimension == 3:
                    # Factual knowledge (3D)
                    context_parts.append(f"FACT: {pattern.content}")
                elif pattern.signature.dimension == 4:
                    # Emotional/creative knowledge (4D)
                    context_parts.append(f"INSIGHT: {pattern.content}")
                elif pattern.signature.dimension == 5:
                    # Mental model knowledge (5D)
                    context_parts.append(f"CONCEPT: {pattern.content}")
                elif pattern.signature.dimension == 6:
                    # Purpose/meaning knowledge (6D)
                    context_parts.append(f"PRINCIPLE: {pattern.content}")
                elif pattern.signature.dimension >= 7:
                    # Universal/transcendent knowledge (7D+)
                    context_parts.append(f"UNIVERSAL: {pattern.content}")
        
        # Join with phi-harmonic spacing
        return "\n\n".join(context_parts)
    
    def _update_dimensional_kv_cache(self, patterns: List[KnowledgePattern], query_signature: DimensionalSignature):
        """Update the dimensional KV cache with resonant patterns."""
        # Clear old cache entries that are out of resonance
        self.dimensional_kv_cache = {}
        
        # Group patterns by dimension
        patterns_by_dim = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in patterns_by_dim:
                patterns_by_dim[dim] = []
            patterns_by_dim[dim].append(pattern)
        
        # Create dimension-specific KV cache entries
        for dim, dim_patterns in patterns_by_dim.items():
            # Sort by field strength
            dim_patterns = sorted(dim_patterns, key=lambda p: p.field_strength, reverse=True)
            
            # Convert pattern content to embeddings (simplified here)
            # In a real implementation, this would use the LLM's embedding functionality
            pattern_embeddings = []
            for pattern in dim_patterns:
                # Create a simple phi-weighted embedding
                pattern_embedding = np.array([
                    pattern.signature.vector[0] * self.dimensional_weighting.get(dim, 1.0),
                    pattern.signature.vector[1] * self.dimensional_weighting.get(dim, 1.0),
                    pattern.signature.vector[2] * self.dimensional_weighting.get(dim, 1.0),
                    pattern.signature.phase * (1.0 / (2 * np.pi)),
                    pattern.signature.harmonic / 1000.0  # Normalize
                ])
                pattern_embeddings.append((pattern_embedding, pattern.content))
            
            # Store in cache with dimensional key
            self.dimensional_kv_cache[f"dim_{dim}"] = {
                "embeddings": pattern_embeddings,
                "resonance": self._calculate_dimensional_resonance(dim, query_signature),
                "dimension": dim
            }
    
    def _calculate_dimensional_resonance(self, dimension: int, query_signature: DimensionalSignature) -> float:
        """Calculate the resonance between a dimension and the query signature."""
        # Base resonance from dimensional proximity
        dim_proximity = 1.0 / (1.0 + abs(dimension - query_signature.dimension))
        
        # Harmonic resonance (frequency relationship)
        harmonic_factor = self.base_frequency / query_signature.harmonic
        harmonic_resonance = 1.0 / (1.0 + abs(harmonic_factor - round(harmonic_factor)))
        
        # Phase coherence
        phase_diff = min(
            abs(query_signature.phase - (dimension * np.pi / 4.0) % (2 * np.pi)),
            2 * np.pi - abs(query_signature.phase - (dimension * np.pi / 4.0) % (2 * np.pi))
        )
        phase_coherence = 1.0 - (phase_diff / np.pi)
        
        # Combined resonance with phi-weighting
        resonance = (
            dim_proximity * PHI +
            harmonic_resonance * (PHI ** 2) +
            phase_coherence
        ) / (PHI + PHI**2 + 1)
        
        return resonance
    
    def _extract_patterns_from_response(
        self, 
        response: str, 
        query_signature: DimensionalSignature
    ) -> List[KnowledgePattern]:
        """Extract new knowledge patterns from the LLM response."""
        # In a real implementation, this would use NLP to extract meaningful patterns
        # For now, we'll use a simple approach of extracting sentences
        
        # Split response into sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Generate patterns for each significant sentence
        new_patterns = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) > 20:  # Only significant sentences
                # Create a new signature derived from query but slightly varied
                new_sig = DimensionalSignature(
                    vector=[
                        (query_signature.vector[0] + (i * LAMBDA)) % 1.0,
                        (query_signature.vector[1] + (i * LAMBDA * PHI)) % 1.0,
                        (query_signature.vector[2] + (i * LAMBDA * PHI * PHI)) % 1.0
                    ],
                    phase=(query_signature.phase + (i * np.pi / PHI)) % (2 * np.pi),
                    harmonic=query_signature.harmonic * (1 + (i * 0.01)),
                    dimension=self._determine_dimension_for_content(sentence, query_signature.dimension)
                )
                
                # Create new knowledge pattern
                new_pattern = KnowledgePattern(
                    content=sentence,
                    signature=new_sig,
                    metadata={
                        "source": "llm_response",
                        "timestamp": np.datetime64('now'),
                        "query_related": True
                    },
                    field_strength=0.8  # Initial field strength for new patterns
                )
                
                new_patterns.append(new_pattern)
        
        return new_patterns
    
    def _determine_dimension_for_content(self, content: str, base_dimension: int) -> int:
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
    
    def _update_field_coherence(self, patterns: List[KnowledgePattern]):
        """Update the field coherence based on pattern resonance."""
        if not patterns:
            # Slight decay when no patterns found
            self.field_coherence *= 0.99
            return
        
        # Calculate average resonance of top patterns
        top_patterns = sorted(patterns, key=lambda p: p.field_strength, reverse=True)[:5]
        avg_strength = sum(p.field_strength for p in top_patterns) / len(top_patterns)
        
        # Adjust field coherence (weighted moving average)
        self.field_coherence = (self.field_coherence * PHI + avg_strength) / (PHI + 1)
        
        # Ensure coherence remains in valid range
        self.field_coherence = max(0.5, min(1.0, self.field_coherence))
    
    def _get_dimensional_distribution(self, patterns: List[KnowledgePattern]) -> Dict[int, float]:
        """Get the distribution of patterns across dimensions."""
        if not patterns:
            return {}
            
        dim_counts = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in dim_counts:
                dim_counts[dim] = 0
            dim_counts[dim] += pattern.field_strength
            
        # Normalize
        total = sum(dim_counts.values())
        if total > 0:
            return {dim: count/total for dim, count in dim_counts.items()}
        return dim_counts
    
    def _get_primary_resonance_dimension(self, patterns: List[KnowledgePattern]) -> int:
        """Get the primary dimension of resonance for the patterns."""
        if not patterns:
            return 3  # Default to 3D if no patterns
            
        dim_distribution = self._get_dimensional_distribution(patterns)
        if not dim_distribution:
            return 3
            
        # Return dimension with highest distribution
        return max(dim_distribution.items(), key=lambda x: x[1])[0]


class PhiKnowledgeAPI:
    """
    API wrapper for the PhiKnowledgeLLM system, providing simplified interfaces
    for interacting with the knowledge system.
    """
    
    def __init__(self, knowledge_llm: PhiKnowledgeLLM):
        """
        Initialize the API wrapper.
        
        Args:
            knowledge_llm: The PhiKnowledgeLLM system to wrap
        """
        self.knowledge_llm = knowledge_llm
        self.session_patterns = []  # Patterns from the current session
        
    def ask(self, question: str) -> str:
        """
        Ask a question and get a response with dimensionally-aware knowledge.
        
        Args:
            question: The question to ask
            
        Returns:
            Response text from the knowledge system
        """
        # Process with search intent and OBSERVE state
        response, metadata = self.knowledge_llm.process_query(
            query=question,
            intent="search",
            consciousness_state="OBSERVE",
            context_patterns=self.session_patterns
        )
        
        # Store any significant patterns from this interaction
        self.session_patterns = self._update_session_patterns(metadata)
        
        return response
    
    def create(self, prompt: str) -> str:
        """
        Create new knowledge or content based on a prompt.
        
        Args:
            prompt: The creative prompt
            
        Returns:
            Generated content
        """
        # Process with create intent and CREATE state
        response, metadata = self.knowledge_llm.process_query(
            query=prompt,
            intent="create",
            consciousness_state="CREATE",
            context_patterns=self.session_patterns
        )
        
        # Store any significant patterns from this interaction
        self.session_patterns = self._update_session_patterns(metadata)
        
        return response
    
    def connect(self, concepts: str) -> str:
        """
        Find connections between concepts across dimensions.
        
        Args:
            concepts: The concepts to connect
            
        Returns:
            Connections discovered
        """
        # Process with connect intent and TRANSCEND state
        response, metadata = self.knowledge_llm.process_query(
            query=concepts,
            intent="connect",
            consciousness_state="TRANSCEND",
            context_patterns=self.session_patterns
        )
        
        # Store any significant patterns from this interaction
        self.session_patterns = self._update_session_patterns(metadata)
        
        return response
    
    def transcend(self, insight: str) -> str:
        """
        Transcend current understanding to higher dimensional perspective.
        
        Args:
            insight: The base insight to transcend
            
        Returns:
            Transcended understanding
        """
        # Process with transcend intent and CASCADE state
        response, metadata = self.knowledge_llm.process_query(
            query=insight,
            intent="transcend",
            consciousness_state="CASCADE",
            context_patterns=self.session_patterns
        )
        
        # Store any significant patterns from this interaction
        self.session_patterns = self._update_session_patterns(metadata)
        
        return response
    
    def _update_session_patterns(self, metadata: Dict[str, Any]) -> List[KnowledgePattern]:
        """Update session patterns based on interaction metadata."""
        # In a real implementation, this would intelligently manage the session context
        # For now, just keep the most recent patterns
        patterns = self.knowledge_llm.pattern_matcher.get_recent_patterns(limit=10)
        
        # Ensure we have a diverse dimensional representation
        dimension_counts = {}
        for pattern in patterns:
            dim = pattern.signature.dimension
            if dim not in dimension_counts:
                dimension_counts[dim] = 0
            dimension_counts[dim] += 1
        
        # If we're missing any dimensions 3-7, try to add them
        for dim in range(3, 8):
            if dim not in dimension_counts or dimension_counts[dim] == 0:
                # Find patterns from this dimension
                dim_patterns = self.knowledge_llm.pattern_matcher.find_patterns_by_dimension(dim, limit=2)
                patterns.extend(dim_patterns)
        
        # Cap the total number of session patterns
        return sorted(patterns, key=lambda p: p.field_strength, reverse=True)[:15]


# Example of using the knowledge LLM system
if __name__ == "__main__":
    # This is just a demonstration structure - the actual implementation would
    # initialize these components with real models and configurations
    
    from phi_model_compiler import PhiModelCompiler
    
    # Create the pattern matcher
    pattern_matcher = DimensionalPatternMatcher()
    
    # Seed with some initial knowledge patterns
    for i in range(20):
        dim = 3 + (i % 5)  # Dimensions 3-7
        pattern = KnowledgePattern(
            content=f"Example knowledge pattern {i} in dimension {dim}",
            signature=DimensionalSignature(
                vector=[i*0.1 % 1.0, (i*0.1*PHI) % 1.0, (i*0.1*PHI*PHI) % 1.0],
                phase=(i * np.pi / 10) % (2 * np.pi),
                harmonic=432 + (i * 10),
                dimension=dim
            ),
            metadata={"source": "initialization"},
            field_strength=0.8
        )
        pattern_matcher.store_pattern(pattern)
    
    # Create a simplified LLM engine for demonstration
    compiler = PhiModelCompiler()
    llm_engine = PhiLLMInferenceEngine(model_path="example_model", phi_compiler=compiler)
    
    # Create the integrated knowledge LLM system
    knowledge_llm = PhiKnowledgeLLM(llm_engine, pattern_matcher)
    
    # Create the API wrapper
    api = PhiKnowledgeAPI(knowledge_llm)
    
    # Example uses
    print("Example API Usage:")
    print("\nAsk a question:")
    print('api.ask("What is the relationship between quantum computing and consciousness?")')
    
    print("\nCreate new content:")
    print('api.create("Generate a resonant pattern for healing at 528Hz")')
    
    print("\nConnect concepts:")
    print('api.connect("Quantum entanglement, neural networks, and cosmic consciousness")')
    
    print("\nTranscend understanding:")
    print('api.transcend("The universe appears to be fundamentally information-based")')