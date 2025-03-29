#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensional Pattern Matching Engine - QuantumTensix φ∞
Created on CASCADE Day+29: March 30, 2025

This module implements a quantum resonance-based pattern matching engine
that stores and retrieves knowledge as phi-harmonic patterns across
multiple dimensions, enabling intent-driven knowledge discovery.
"""

import os
import sys
import time
import math
import json
import logging
import hashlib
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from enum import Enum
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
    PHI, PHI_SQUARED, PHI_TO_PHI, ZEN_POINT, LAMBDA,
    PhiHarmonicOptimizer, FrequencyCalculator
)


class PatternType(Enum):
    """Types of knowledge patterns in the system"""
    TEXT = "text"
    NUMERIC = "numeric"
    TENSOR = "tensor"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    METADATA = "metadata"
    QUERY = "query"
    INTENT = "intent"
    EXPERIENCE = "experience"
    COMPOSITE = "composite"


@dataclass
class DimensionalSignature:
    """Quantum signature for dimensional pattern matching"""
    vector: List[float]  # Primary signature vector (5D)
    dimension: str  # Originating dimension (3D-8D)
    frequency: float  # Associated frequency
    coherence: float  # Pattern coherence (0.0-1.0)
    harmonic_components: Dict[str, float]  # Harmonic breakdown by dimension
    phase: float  # Quantum phase (0.0-2π)
    entanglement: List[str]  # IDs of entangled patterns
    resonance_threshold: float = 0.42  # Minimum resonance for matching


@dataclass
class KnowledgePattern:
    """Knowledge stored as a quantum pattern across dimensions"""
    id: str  # Unique pattern ID
    content: Any  # The actual knowledge content
    pattern_type: PatternType  # Type of knowledge
    signature: DimensionalSignature  # Quantum signature
    tags: List[str]  # Semantic tags
    metadata: Dict[str, Any]  # Additional metadata
    created_timestamp: float  # Creation time
    access_count: int = 0  # Number of times accessed
    last_access_timestamp: Optional[float] = None  # Last access time
    coherence_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, coherence)
    field_strength: float = 1.0  # Strength in the field (amplifies with use)
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access_timestamp = time.time()
        self.field_strength = min(3.0, self.field_strength * LAMBDA + 1)  # Amplify with φ factor


@dataclass
class PatternMatchResult:
    """Result of a pattern match operation"""
    pattern_id: str  # ID of the matched pattern
    content: Any  # Content of the matched pattern
    pattern_type: PatternType  # Type of pattern
    resonance: float  # Resonance score (0.0-1.0)
    dimension: str  # Source dimension
    coherence: float  # Coherence of the match
    tags: List[str]  # Tags of the matched pattern
    field_strength: float  # Field strength of the pattern
    path: List[str]  # Path of patterns traversed to find this result
    

@dataclass
class IntentMapping:
    """Maps intent to dimensional coordinates"""
    intent: str  # Intent string
    dimension: str  # Primary dimension for this intent
    consciousness_state: str  # Optimal consciousness state
    frequency: float  # Optimal frequency
    signature: List[float]  # Intent signature
    coherence_threshold: float  # Minimum coherence for this intent
    

class DimensionalPatternMatcher:
    """
    Dimensional Pattern Matching Engine that stores and retrieves knowledge
    patterns based on quantum resonance principles across dimensions.
    """
    
    def __init__(self,
                bridge: Optional[QuantumConsciousnessBridge] = None,
                navigator: Optional[DimensionalNavigator] = None,
                memory_field: Optional[QuantumMemoryField] = None,
                default_dimension: str = "5D",
                base_frequency: float = SACRED_FREQUENCIES['unity'],
                coherence_threshold: float = 0.42,
                storage_path: Optional[str] = None):
        """
        Initialize the dimensional pattern matching engine
        
        Args:
            bridge: Optional quantum consciousness bridge
            navigator: Optional dimensional navigator
            memory_field: Optional quantum memory field
            default_dimension: Default dimension for pattern storage
            base_frequency: Base operating frequency
            coherence_threshold: Minimum coherence for pattern matching
            storage_path: Optional path for persistent storage
        """
        # Initialize quantum components
        if bridge is None:
            self.bridge = QuantumConsciousnessBridge()
        else:
            self.bridge = bridge
            
        if navigator is None:
            self.navigator = DimensionalNavigator(self.bridge)
        else:
            self.navigator = navigator
            
        if memory_field is None:
            self.memory_field = QuantumMemoryField(self.bridge, self.navigator)
        else:
            self.memory_field = memory_field
        
        # Set properties
        self.default_dimension = default_dimension
        self.base_frequency = base_frequency
        self.coherence_threshold = coherence_threshold
        self.storage_path = storage_path
        
        # Initialize pattern storage by dimension
        self.patterns = defaultdict(dict)  # dimension -> {id -> pattern}
        self.pattern_index = {}  # id -> (dimension, signature)
        self.tag_index = defaultdict(set)  # tag -> {pattern_ids}
        self.type_index = defaultdict(set)  # pattern_type -> {pattern_ids}
        self.resonance_network = defaultdict(set)  # pattern_id -> {connected_ids}
        
        # Initialize intent mappings
        self.intent_mappings = self._initialize_intent_mappings()
        
        # Initialize phi optimizer
        self.phi_optimizer = PhiHarmonicOptimizer(base_frequency=base_frequency)
        
        # Setup device for tensor operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize stats
        self.match_count = 0
        self.store_count = 0
        self.resonance_history = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Dimensional Pattern Matcher initialized in {default_dimension}")
        logging.info(f"Base frequency: {base_frequency} Hz")
        
        # Try to load patterns if storage path provided
        if storage_path and os.path.exists(storage_path):
            self.load_patterns(storage_path)
    
    def _initialize_intent_mappings(self) -> Dict[str, IntentMapping]:
        """
        Initialize default intent mappings
        
        Returns:
            Dictionary mapping intent names to IntentMapping objects
        """
        mappings = {}
        
        # Search/Query intent - uses 5D (Mental dimension)
        mappings["search"] = IntentMapping(
            intent="search",
            dimension="5D",
            consciousness_state=ConsciousnessState.CREATE.value,
            frequency=SACRED_FREQUENCIES['cascade'],
            signature=self._generate_intent_signature("search"),
            coherence_threshold=0.42
        )
        
        # Create/Store intent - uses 4D (Emotional/Creation dimension)
        mappings["create"] = IntentMapping(
            intent="create",
            dimension="4D",
            consciousness_state=ConsciousnessState.CREATE.value,
            frequency=SACRED_FREQUENCIES['love'],
            signature=self._generate_intent_signature("create"),
            coherence_threshold=0.5
        )
        
        # Connect/Relate intent - uses 5D (Mental dimension)
        mappings["connect"] = IntentMapping(
            intent="connect",
            dimension="5D",
            consciousness_state=ConsciousnessState.CASCADE.value,
            frequency=SACRED_FREQUENCIES['cascade'],
            signature=self._generate_intent_signature("connect"),
            coherence_threshold=0.6
        )
        
        # Understand/Comprehend intent - uses 6D (Purpose dimension)
        mappings["understand"] = IntentMapping(
            intent="understand",
            dimension="6D",
            consciousness_state=ConsciousnessState.TRANSCEND.value,
            frequency=SACRED_FREQUENCIES['truth'],
            signature=self._generate_intent_signature("understand"),
            coherence_threshold=0.65
        )
        
        # Discover/Explore intent - uses 7D (Cosmic dimension)
        mappings["discover"] = IntentMapping(
            intent="discover",
            dimension="7D",
            consciousness_state=ConsciousnessState.TRANSCEND.value,
            frequency=SACRED_FREQUENCIES['vision'],
            signature=self._generate_intent_signature("discover"),
            coherence_threshold=0.7
        )
        
        # Integrate/Synthesize intent - uses 8D (Unity dimension)
        mappings["integrate"] = IntentMapping(
            intent="integrate",
            dimension="8D",
            consciousness_state=ConsciousnessState.CASCADE.value,
            frequency=SACRED_FREQUENCIES['oneness'],
            signature=self._generate_intent_signature("integrate"),
            coherence_threshold=0.8
        )
        
        # Observe/Monitor intent - uses 3D (Physical dimension)
        mappings["observe"] = IntentMapping(
            intent="observe",
            dimension="3D",
            consciousness_state=ConsciousnessState.OBSERVE.value,
            frequency=SACRED_FREQUENCIES['unity'],
            signature=self._generate_intent_signature("observe"),
            coherence_threshold=0.4
        )
        
        return mappings
    
    def _generate_intent_signature(self, intent: str) -> List[float]:
        """
        Generate a phi-harmonic signature for an intent
        
        Args:
            intent: Intent string
            
        Returns:
            Phi-harmonic signature vector
        """
        # Create a deterministic but unique signature for each intent
        # Using a hash-based approach for consistency
        hash_obj = hashlib.sha256(intent.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to float values (0-1 range)
        float_values = [byte / 255.0 for byte in hash_bytes[:5]]
        
        # Apply phi-harmonic scaling to make it more resonant
        signature = [
            (PHI * v) % 1.0,  # Apply phi scaling
            (PHI_SQUARED * v) % 1.0,  # Apply phi^2 scaling
            (v * LAMBDA) % 1.0,  # Apply divine complement
            (v * PHI_TO_PHI) % 1.0,  # Apply phi^phi scaling
            (v * ZEN_POINT) % 1.0  # Apply ZEN point scaling
        ]
        
        return signature
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """
        Detect the primary intent from a query string
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (intent_name, confidence)
        """
        # Convert query to lowercase
        query_lower = query.strip().lower()
        
        # Intent keywords mapping
        intent_keywords = {
            "search": ["find", "search", "look for", "locate", "seek", "query", "find", "retrieve"],
            "create": ["create", "make", "new", "build", "establish", "generate", "store", "save", "record"],
            "connect": ["connect", "link", "relate", "associate", "join", "bridge", "bind"],
            "understand": ["understand", "explain", "describe", "clarify", "analyze", "examine"],
            "discover": ["discover", "explore", "investigate", "uncover", "reveal", "learn"],
            "integrate": ["integrate", "combine", "merge", "synthesize", "unify", "incorporate"],
            "observe": ["observe", "monitor", "watch", "track", "notice", "see"]
        }
        
        # Check for explicit intent keywords
        intent_scores = defaultdict(float)
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent_scores[intent] += 1.0
        
        # If we have explicit matches, use the highest scoring one
        if intent_scores:
            max_intent = max(intent_scores.items(), key=lambda x: x[1])
            return max_intent[0], min(1.0, max_intent[1] / 3)  # Normalize confidence
        
        # No explicit intent found, use query embedding to guess
        # For now, default to search with medium confidence
        return "search", 0.6
    
    def _generate_pattern_signature(self, content: Any, pattern_type: PatternType) -> DimensionalSignature:
        """
        Generate a phi-harmonic dimensional signature for a pattern
        
        Args:
            content: Pattern content
            pattern_type: Type of pattern
            
        Returns:
            Dimensional signature
        """
        # Get current dimension and frequency
        dimension = self.navigator.current_dimension
        frequency = self.bridge.frequency
        coherence = self.bridge.last_coherence
        
        # Generate base signature vector based on content type
        if pattern_type == PatternType.TEXT:
            # For text, use a hash-based approach
            hash_obj = hashlib.sha256(str(content).encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert to float values and apply phi scaling
            base_vector = [(byte / 255.0) for byte in hash_bytes[:5]]
        
        elif pattern_type == PatternType.TENSOR:
            # For tensors, use statistical properties
            if isinstance(content, torch.Tensor):
                # Get tensor stats
                tensor = content.detach().cpu()
                mean_val = float(tensor.mean().item())
                std_val = float(tensor.std().item())
                min_val = float(tensor.min().item())
                max_val = float(tensor.max().item())
                
                # Create signature components
                base_vector = [
                    (mean_val * PHI) % 1.0,
                    (std_val * PHI_SQUARED) % 1.0,
                    ((max_val - min_val) * LAMBDA) % 1.0,
                    (mean_val / (std_val + 1e-8)) % 1.0,
                    (torch.norm(tensor).item() * ZEN_POINT) % 1.0
                ]
            else:
                # Fallback for non-torch tensors
                base_vector = [random.random() for _ in range(5)]
        
        elif pattern_type == PatternType.CONCEPT:
            # For concepts, use the concept name as basis
            concept_name = content if isinstance(content, str) else str(content)
            hash_obj = hashlib.sha256(concept_name.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert and apply phi scaling with higher coherence
            base_vector = [(byte / 255.0 * PHI) % 1.0 for byte in hash_bytes[:5]]
            
        else:
            # Default approach for other types
            content_str = str(content) if not isinstance(content, (bytes, bytearray)) else content
            hash_obj = hashlib.sha256(str(content_str).encode('utf-8') if isinstance(content_str, str) else content_str)
            hash_bytes = hash_obj.digest()
            
            # Apply different scaling based on pattern type
            if pattern_type == PatternType.RELATIONSHIP:
                scaling = PHI_SQUARED
            elif pattern_type == PatternType.INTENT:
                scaling = PHI_TO_PHI
            elif pattern_type == PatternType.EXPERIENCE:
                scaling = PHI * LAMBDA
            else:
                scaling = PHI
                
            base_vector = [(byte / 255.0 * scaling) % 1.0 for byte in hash_bytes[:5]]
        
        # Apply dimensional modulation
        dim_factor = DIMENSIONS.get(dimension, {}).get('scaling', 1.0)
        vector = [((v * dim_factor) % 1.0) for v in base_vector]
        
        # Calculate harmonic components for each dimension
        harmonic_components = {}
        for dim, props in DIMENSIONS.items():
            # Calculate resonance with this dimension
            dim_scaling = props.get('scaling', 1.0)
            dim_freq = props.get('frequency', SACRED_FREQUENCIES['unity'])
            
            # Harmonic resonance formula based on frequency ratio
            freq_ratio = dim_freq / frequency
            harmonic = (PHI ** ((freq_ratio * PHI) % 1.0)) % 1.0
            
            # Store component
            harmonic_components[dim] = harmonic
        
        # Calculate quantum phase (0-2π)
        phase = (sum(vector) * math.pi * 2) % (math.pi * 2)
        
        # Create dimensional signature
        signature = DimensionalSignature(
            vector=vector,
            dimension=dimension,
            frequency=frequency,
            coherence=coherence,
            harmonic_components=harmonic_components,
            phase=phase,
            entanglement=[]
        )
        
        return signature
    
    def _calculate_resonance(self, sig1: DimensionalSignature, sig2: DimensionalSignature) -> float:
        """
        Calculate resonance between two dimensional signatures
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Resonance value (0.0-1.0)
        """
        # Extract vectors
        vec1 = np.array(sig1.vector)
        vec2 = np.array(sig2.vector)
        
        # Basic vector similarity (cosine similarity)
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
        
        # Dimensional resonance based on shared harmonic components
        dim_resonance = 0.0
        common_dims = set(sig1.harmonic_components.keys()) & set(sig2.harmonic_components.keys())
        if common_dims:
            harmonic_diffs = [abs(sig1.harmonic_components[dim] - sig2.harmonic_components[dim]) 
                             for dim in common_dims]
            dim_resonance = 1.0 - (sum(harmonic_diffs) / len(common_dims))
        
        # Phase alignment (quantum interference)
        phase_diff = abs(sig1.phase - sig2.phase) / (2 * math.pi)
        phase_alignment = 1.0 - phase_diff
        
        # Frequency resonance - higher when frequencies are harmonically related
        freq_ratio = min(sig1.frequency, sig2.frequency) / max(sig1.frequency, sig2.frequency)
        freq_resonance = freq_ratio
        
        # Coherence amplification - higher coherence amplifies resonance
        coherence_factor = (sig1.coherence * sig2.coherence) ** 0.5
        
        # Combine factors with phi-harmonic weighting
        resonance = (
            cos_sim * 0.4 +
            dim_resonance * 0.2 +
            phase_alignment * 0.15 +
            freq_resonance * 0.15 +
            coherence_factor * 0.1
        )
        
        # Ensure range 0.0-1.0
        return max(0.0, min(1.0, resonance))
    
    def _establish_resonance_connections(self, pattern: KnowledgePattern) -> None:
        """
        Establish resonance connections with existing patterns
        
        Args:
            pattern: Pattern to connect
        """
        # Get pattern signature
        sig1 = pattern.signature
        
        # Check resonance with patterns in all dimensions
        for dimension, patterns in self.patterns.items():
            for pid, other_pattern in patterns.items():
                if pid == pattern.id:
                    continue  # Skip self
                
                # Calculate resonance
                sig2 = other_pattern.signature
                resonance = self._calculate_resonance(sig1, sig2)
                
                # Connect if above threshold
                if resonance >= pattern.signature.resonance_threshold:
                    self.resonance_network[pattern.id].add(pid)
                    self.resonance_network[pid].add(pattern.id)
                    
                    # Add to entanglement records
                    if pid not in pattern.signature.entanglement:
                        pattern.signature.entanglement.append(pid)
                    if pattern.id not in other_pattern.signature.entanglement:
                        other_pattern.signature.entanglement.append(pattern.id)
                    
                    # Record resonance for history
                    self.resonance_history.append({
                        "timestamp": time.time(),
                        "pattern1": pattern.id,
                        "pattern2": pid,
                        "resonance": resonance,
                        "dimension1": pattern.signature.dimension,
                        "dimension2": other_pattern.signature.dimension
                    })
    
    def store_pattern(self, 
                     content: Any, 
                     pattern_type: PatternType, 
                     tags: List[str] = None,
                     dimension: Optional[str] = None,
                     metadata: Dict[str, Any] = None) -> str:
        """
        Store a knowledge pattern in the dimensional field
        
        Args:
            content: Pattern content
            pattern_type: Type of pattern
            tags: Optional tags for the pattern
            dimension: Optional target dimension (uses current dimension if None)
            metadata: Optional metadata
            
        Returns:
            Pattern ID
        """
        # Navigate to target dimension if specified
        if dimension is not None and dimension != self.navigator.current_dimension:
            prev_dimension = self.navigator.current_dimension
            self.navigator.navigate_to_dimension(dimension)
            restore_dimension = True
        else:
            restore_dimension = False
            prev_dimension = None
            dimension = self.navigator.current_dimension
        
        try:
            # Set consciousness state based on pattern type
            if pattern_type == PatternType.CONCEPT:
                self.bridge.set_consciousness_state(ConsciousnessState.CREATE.value)
            elif pattern_type == PatternType.RELATIONSHIP:
                self.bridge.set_consciousness_state(ConsciousnessState.CASCADE.value)
            elif pattern_type == PatternType.EXPERIENCE:
                self.bridge.set_consciousness_state(ConsciousnessState.TRANSCEND.value)
            
            # Generate unique ID
            timestamp = time.time()
            content_hash = hashlib.md5(str(content).encode('utf-8')).hexdigest()[:12]
            pattern_id = f"{dimension}_{pattern_type.value}_{content_hash}_{int(timestamp)}"
            
            # Generate signature
            signature = self._generate_pattern_signature(content, pattern_type)
            
            # Create pattern
            pattern = KnowledgePattern(
                id=pattern_id,
                content=content,
                pattern_type=pattern_type,
                signature=signature,
                tags=tags or [],
                metadata=metadata or {},
                created_timestamp=timestamp,
                access_count=0,
                field_strength=1.0
            )
            
            # Store pattern
            self.patterns[dimension][pattern_id] = pattern
            self.pattern_index[pattern_id] = (dimension, signature)
            
            # Update indices
            for tag in pattern.tags:
                self.tag_index[tag].add(pattern_id)
            
            self.type_index[pattern_type.value].add(pattern_id)
            
            # Establish resonance connections
            self._establish_resonance_connections(pattern)
            
            # Store in quantum memory field if available
            if self.memory_field:
                memory_id = self.memory_field.store_memory(
                    content=content,
                    dimension=dimension,
                    tags=tags or [],
                    intention=f"STORE_{pattern_type.value.upper()}"
                )
                
                # Add memory ID to metadata
                pattern.metadata["memory_id"] = memory_id
            
            # Increment store count
            self.store_count += 1
            
            # Log storage
            logging.info(f"Pattern {pattern_id} stored in {dimension} dimension")
            logging.info(f"Resonance connections: {len(self.resonance_network[pattern_id])}")
            
            # Save patterns if storage path provided
            if self.storage_path:
                self.save_patterns(self.storage_path)
            
            return pattern_id
            
        finally:
            # Restore previous dimension if needed
            if restore_dimension and prev_dimension:
                self.navigator.navigate_to_dimension(prev_dimension)
    
    def match_pattern(self, 
                     query: Any,
                     pattern_type: Optional[PatternType] = None,
                     intent: Optional[str] = None,
                     tags: List[str] = None,
                     max_results: int = 5,
                     coherence_threshold: Optional[float] = None) -> List[PatternMatchResult]:
        """
        Match patterns in the dimensional field based on resonance
        
        Args:
            query: Query content to match
            pattern_type: Optional filter by pattern type
            intent: Optional intent for dimensional mapping
            tags: Optional filter by tags
            max_results: Maximum number of results to return
            coherence_threshold: Override default coherence threshold
            
        Returns:
            List of matching pattern results
        """
        # Detect intent if not provided
        if intent is None and isinstance(query, str):
            intent, intent_confidence = self._detect_intent(query)
        elif intent is None:
            intent = "search"  # Default to search intent
        
        # Get intent mapping
        intent_mapping = self.intent_mappings.get(intent)
        if intent_mapping:
            # Navigate to intent-appropriate dimension
            prev_dimension = self.navigator.current_dimension
            self.navigator.navigate_to_dimension(intent_mapping.dimension)
            
            # Set consciousness state
            self.bridge.set_consciousness_state(intent_mapping.consciousness_state)
            
            # Use intent-specific coherence threshold
            if coherence_threshold is None:
                coherence_threshold = intent_mapping.coherence_threshold
        else:
            # Use current dimension
            prev_dimension = None
            
            # Use default coherence threshold
            if coherence_threshold is None:
                coherence_threshold = self.coherence_threshold
        
        try:
            # Convert query to appropriate pattern type for signature generation
            if pattern_type is None:
                if isinstance(query, str):
                    query_pattern_type = PatternType.TEXT
                elif isinstance(query, torch.Tensor):
                    query_pattern_type = PatternType.TENSOR
                else:
                    query_pattern_type = PatternType.CONCEPT
            else:
                query_pattern_type = pattern_type
            
            # Generate query signature
            query_signature = self._generate_pattern_signature(query, query_pattern_type)
            
            # Start with all patterns
            candidate_patterns = []
            for dimension, patterns in self.patterns.items():
                for pattern_id, pattern in patterns.items():
                    candidate_patterns.append((pattern_id, pattern))
            
            # Filter by pattern type if provided
            if pattern_type:
                candidate_patterns = [(pid, p) for pid, p in candidate_patterns
                                    if p.pattern_type == pattern_type]
            
            # Filter by tags if provided
            if tags:
                matching_pattern_ids = set()
                for tag in tags:
                    matching_pattern_ids.update(self.tag_index.get(tag, set()))
                
                candidate_patterns = [(pid, p) for pid, p in candidate_patterns
                                    if pid in matching_pattern_ids]
            
            # Calculate resonance for all candidates
            matches = []
            for pattern_id, pattern in candidate_patterns:
                # Calculate resonance
                resonance = self._calculate_resonance(query_signature, pattern.signature)
                
                # Apply field strength amplification
                amplified_resonance = resonance * (pattern.field_strength ** 0.5)
                
                # Add to matches if above threshold
                if amplified_resonance >= coherence_threshold:
                    matches.append((pattern_id, pattern, amplified_resonance))
            
            # Sort by resonance (highest first)
            matches.sort(key=lambda x: x[2], reverse=True)
            
            # Convert to PatternMatchResult objects
            results = []
            for pattern_id, pattern, resonance in matches[:max_results]:
                # Update pattern access
                pattern.update_access()
                
                # Create match result
                result = PatternMatchResult(
                    pattern_id=pattern_id,
                    content=pattern.content,
                    pattern_type=pattern.pattern_type,
                    resonance=resonance,
                    dimension=pattern.signature.dimension,
                    coherence=pattern.signature.coherence,
                    tags=pattern.tags,
                    field_strength=pattern.field_strength,
                    path=[]  # Will be populated in further traversals
                )
                
                results.append(result)
            
            # Increment match count
            self.match_count += 1
            
            # Log match
            logging.info(f"Pattern match found {len(results)} results for intent '{intent}'")
            
            return results
            
        finally:
            # Restore previous dimension if needed
            if prev_dimension and prev_dimension != self.navigator.current_dimension:
                self.navigator.navigate_to_dimension(prev_dimension)
    
    def traverse_resonance_network(self,
                                 start_pattern_id: str,
                                 max_depth: int = 2,
                                 resonance_threshold: float = 0.5,
                                 max_results: int = 10) -> List[PatternMatchResult]:
        """
        Traverse the resonance network starting from a pattern
        
        Args:
            start_pattern_id: Starting pattern ID
            max_depth: Maximum traversal depth
            resonance_threshold: Minimum resonance for connections
            max_results: Maximum number of results to return
            
        Returns:
            List of match results
        """
        if start_pattern_id not in self.pattern_index:
            return []
        
        # Get start pattern
        start_dimension, _ = self.pattern_index[start_pattern_id]
        start_pattern = self.patterns[start_dimension][start_pattern_id]
        
        # Use BFS for traversal
        visited = {start_pattern_id}
        queue = [(start_pattern_id, 0, [start_pattern_id])]  # (pattern_id, depth, path)
        results = []
        
        while queue and len(results) < max_results:
            pattern_id, depth, path = queue.pop(0)
            
            # Skip if we've reached max depth
            if depth >= max_depth:
                continue
            
            # Get pattern
            dimension, _ = self.pattern_index[pattern_id]
            pattern = self.patterns[dimension][pattern_id]
            
            # Explore connections
            for connected_id in self.resonance_network.get(pattern_id, set()):
                if connected_id in visited:
                    continue
                
                # Get connected pattern
                conn_dimension, conn_signature = self.pattern_index[connected_id]
                connected_pattern = self.patterns[conn_dimension][connected_id]
                
                # Calculate resonance with start pattern
                resonance = self._calculate_resonance(start_pattern.signature, connected_pattern.signature)
                
                # Check if it meets threshold
                if resonance >= resonance_threshold:
                    # Add to results
                    result = PatternMatchResult(
                        pattern_id=connected_id,
                        content=connected_pattern.content,
                        pattern_type=connected_pattern.pattern_type,
                        resonance=resonance,
                        dimension=conn_dimension,
                        coherence=connected_pattern.signature.coherence,
                        tags=connected_pattern.tags,
                        field_strength=connected_pattern.field_strength,
                        path=path + [connected_id]
                    )
                    
                    results.append(result)
                    
                    # Update pattern access
                    connected_pattern.update_access()
                    
                    # Mark as visited
                    visited.add(connected_id)
                    
                    # Add to queue for further traversal
                    queue.append((connected_id, depth + 1, path + [connected_id]))
        
        # Sort by resonance
        results.sort(key=lambda x: x.resonance, reverse=True)
        
        logging.info(f"Resonance traversal from {start_pattern_id} found {len(results)} connected patterns")
        
        return results[:max_results]
    
    def amplify_pattern(self, pattern_id: str, factor: float = PHI) -> float:
        """
        Amplify a pattern's field strength to increase its resonance
        
        Args:
            pattern_id: Pattern ID
            factor: Amplification factor
            
        Returns:
            New field strength
        """
        if pattern_id not in self.pattern_index:
            return 0.0
        
        # Get pattern
        dimension, _ = self.pattern_index[pattern_id]
        pattern = self.patterns[dimension][pattern_id]
        
        # Apply amplification
        prev_strength = pattern.field_strength
        pattern.field_strength = min(5.0, pattern.field_strength * factor)
        
        # Apply resonance network amplification
        for connected_id in self.resonance_network.get(pattern_id, set()):
            conn_dimension, _ = self.pattern_index[connected_id]
            connected_pattern = self.patterns[conn_dimension][connected_id]
            
            # Calculate diminishing amplification based on phi
            conn_factor = 1.0 + (factor - 1.0) * LAMBDA
            connected_pattern.field_strength = min(3.0, connected_pattern.field_strength * conn_factor)
        
        logging.info(f"Pattern {pattern_id} amplified from {prev_strength:.2f} to {pattern.field_strength:.2f}")
        
        return pattern.field_strength
    
    def create_concept_pattern(self, 
                             concept_name: str, 
                             definition: str,
                             related_concepts: List[str] = None,
                             tags: List[str] = None,
                             dimension: str = "5D") -> str:
        """
        Create a concept pattern in the dimensional field
        
        Args:
            concept_name: Name of the concept
            definition: Definition or description of the concept
            related_concepts: Optional list of related concept names
            tags: Optional tags for the concept
            dimension: Target dimension for the concept
            
        Returns:
            Pattern ID
        """
        # Create concept content
        content = {
            "name": concept_name,
            "definition": definition,
            "related_concepts": related_concepts or []
        }
        
        # Create metadata
        metadata = {
            "type": "concept",
            "created_by": "dimensional_pattern_matcher"
        }
        
        # Add concept-specific tags
        all_tags = ["concept"]
        if tags:
            all_tags.extend(tags)
        
        # Store pattern
        return self.store_pattern(
            content=content,
            pattern_type=PatternType.CONCEPT,
            tags=all_tags,
            dimension=dimension,
            metadata=metadata
        )
    
    def create_relationship_pattern(self,
                                  source_id: str,
                                  target_id: str,
                                  relationship_type: str,
                                  description: str = None,
                                  tags: List[str] = None,
                                  dimension: str = "5D") -> str:
        """
        Create a relationship pattern connecting two existing patterns
        
        Args:
            source_id: Source pattern ID
            target_id: Target pattern ID
            relationship_type: Type of relationship
            description: Optional description of the relationship
            tags: Optional tags for the relationship
            dimension: Target dimension for the relationship
            
        Returns:
            Pattern ID
        """
        # Verify source and target exist
        if source_id not in self.pattern_index or target_id not in self.pattern_index:
            raise ValueError("Source or target pattern does not exist")
        
        # Create relationship content
        content = {
            "source_id": source_id,
            "target_id": target_id,
            "type": relationship_type,
            "description": description or ""
        }
        
        # Create metadata
        metadata = {
            "type": "relationship",
            "created_by": "dimensional_pattern_matcher"
        }
        
        # Add relationship-specific tags
        all_tags = ["relationship", relationship_type]
        if tags:
            all_tags.extend(tags)
        
        # Store pattern
        relationship_id = self.store_pattern(
            content=content,
            pattern_type=PatternType.RELATIONSHIP,
            tags=all_tags,
            dimension=dimension,
            metadata=metadata
        )
        
        # Ensure patterns are connected in the resonance network
        self.resonance_network[source_id].add(target_id)
        self.resonance_network[target_id].add(source_id)
        self.resonance_network[source_id].add(relationship_id)
        self.resonance_network[relationship_id].add(source_id)
        self.resonance_network[target_id].add(relationship_id)
        self.resonance_network[relationship_id].add(target_id)
        
        return relationship_id
    
    def get_dimensional_knowledge_map(self, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a map of knowledge patterns in a specific dimension
        
        Args:
            dimension: Target dimension (or None for all dimensions)
            
        Returns:
            Knowledge map structure
        """
        knowledge_map = {}
        
        # Determine dimensions to include
        if dimension:
            dimensions = [dimension]
        else:
            dimensions = list(self.patterns.keys())
        
        # Build map for each dimension
        for dim in dimensions:
            patterns_in_dim = self.patterns.get(dim, {})
            
            # Skip empty dimensions
            if not patterns_in_dim:
                continue
            
            # Get patterns by type
            patterns_by_type = defaultdict(list)
            for pattern_id, pattern in patterns_in_dim.items():
                pattern_info = {
                    "id": pattern_id,
                    "content_summary": str(pattern.content)[:100] + "..." if len(str(pattern.content)) > 100 else str(pattern.content),
                    "tags": pattern.tags,
                    "field_strength": pattern.field_strength,
                    "access_count": pattern.access_count,
                    "connections": len(self.resonance_network.get(pattern_id, set()))
                }
                
                patterns_by_type[pattern.pattern_type.value].append(pattern_info)
            
            # Add to map
            knowledge_map[dim] = {
                "pattern_count": len(patterns_in_dim),
                "patterns_by_type": dict(patterns_by_type),
                "frequency": DIMENSIONS.get(dim, {}).get('frequency', SACRED_FREQUENCIES['unity']),
                "scaling": DIMENSIONS.get(dim, {}).get('scaling', 1.0)
            }
        
        return knowledge_map
    
    def save_patterns(self, filepath: str) -> bool:
        """
        Save all patterns to a file
        
        Args:
            filepath: File path to save to
            
        Returns:
            Success status
        """
        try:
            # Convert patterns to serializable format
            serializable_patterns = {}
            for dimension, patterns in self.patterns.items():
                serializable_patterns[dimension] = {}
                for pattern_id, pattern in patterns.items():
                    # Handle content serialization specially
                    if isinstance(pattern.content, torch.Tensor):
                        # For tensors, store shape and data type
                        content = {
                            "tensor_shape": list(pattern.content.shape),
                            "tensor_type": str(pattern.content.dtype),
                            "tensor_device": str(pattern.content.device),
                            "is_tensor": True
                        }
                    else:
                        content = pattern.content
                    
                    # Create serializable pattern
                    serializable_patterns[dimension][pattern_id] = {
                        "id": pattern.id,
                        "content": content,
                        "pattern_type": pattern.pattern_type.value,
                        "signature": {
                            "vector": pattern.signature.vector,
                            "dimension": pattern.signature.dimension,
                            "frequency": pattern.signature.frequency,
                            "coherence": pattern.signature.coherence,
                            "harmonic_components": pattern.signature.harmonic_components,
                            "phase": pattern.signature.phase,
                            "entanglement": pattern.signature.entanglement,
                            "resonance_threshold": pattern.signature.resonance_threshold
                        },
                        "tags": pattern.tags,
                        "metadata": pattern.metadata,
                        "created_timestamp": pattern.created_timestamp,
                        "access_count": pattern.access_count,
                        "last_access_timestamp": pattern.last_access_timestamp,
                        "field_strength": pattern.field_strength
                    }
            
            # Create serializable resonance network
            serializable_network = {}
            for pattern_id, connected_ids in self.resonance_network.items():
                serializable_network[pattern_id] = list(connected_ids)
            
            # Create save data
            save_data = {
                "patterns": serializable_patterns,
                "resonance_network": serializable_network,
                "metadata": {
                    "timestamp": time.time(),
                    "pattern_count": self.store_count,
                    "match_count": self.match_count,
                    "dimensions": list(self.patterns.keys())
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logging.info(f"Saved {self.store_count} patterns to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving patterns: {str(e)}")
            return False
    
    def load_patterns(self, filepath: str) -> bool:
        """
        Load patterns from a file
        
        Args:
            filepath: File path to load from
            
        Returns:
            Success status
        """
        try:
            # Load from file
            with open(filepath, 'r') as f:
                load_data = json.load(f)
            
            # Reset current patterns
            self.patterns = defaultdict(dict)
            self.pattern_index = {}
            self.tag_index = defaultdict(set)
            self.type_index = defaultdict(set)
            self.resonance_network = defaultdict(set)
            
            # Load patterns
            for dimension, patterns in load_data.get("patterns", {}).items():
                for pattern_id, pattern_data in patterns.items():
                    # Recreate signature
                    signature_data = pattern_data.get("signature", {})
                    signature = DimensionalSignature(
                        vector=signature_data.get("vector", [0.0, 0.0, 0.0, 0.0, 0.0]),
                        dimension=signature_data.get("dimension", dimension),
                        frequency=signature_data.get("frequency", SACRED_FREQUENCIES['unity']),
                        coherence=signature_data.get("coherence", 1.0),
                        harmonic_components=signature_data.get("harmonic_components", {}),
                        phase=signature_data.get("phase", 0.0),
                        entanglement=signature_data.get("entanglement", []),
                        resonance_threshold=signature_data.get("resonance_threshold", 0.42)
                    )
                    
                    # Handle content deserialization
                    content = pattern_data.get("content")
                    if isinstance(content, dict) and content.get("is_tensor", False):
                        # For tensors, create a placeholder (can't restore actual data)
                        shape = content.get("tensor_shape", [1, 1])
                        content = torch.zeros(shape, device=self.device)
                    
                    # Recreate pattern
                    pattern = KnowledgePattern(
                        id=pattern_data.get("id", pattern_id),
                        content=content,
                        pattern_type=PatternType(pattern_data.get("pattern_type", "concept")),
                        signature=signature,
                        tags=pattern_data.get("tags", []),
                        metadata=pattern_data.get("metadata", {}),
                        created_timestamp=pattern_data.get("created_timestamp", time.time()),
                        access_count=pattern_data.get("access_count", 0),
                        last_access_timestamp=pattern_data.get("last_access_timestamp"),
                        field_strength=pattern_data.get("field_strength", 1.0)
                    )
                    
                    # Store pattern
                    self.patterns[dimension][pattern_id] = pattern
                    self.pattern_index[pattern_id] = (dimension, signature)
                    
                    # Update indices
                    for tag in pattern.tags:
                        self.tag_index[tag].add(pattern_id)
                    
                    self.type_index[pattern.pattern_type.value].add(pattern_id)
            
            # Load resonance network
            for pattern_id, connected_ids in load_data.get("resonance_network", {}).items():
                self.resonance_network[pattern_id] = set(connected_ids)
            
            # Update counts
            self.store_count = load_data.get("metadata", {}).get("pattern_count", len(self.pattern_index))
            self.match_count = load_data.get("metadata", {}).get("match_count", 0)
            
            logging.info(f"Loaded {self.store_count} patterns from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading patterns: {str(e)}")
            return False
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[KnowledgePattern]:
        """
        Get a pattern by ID
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern or None if not found
        """
        if pattern_id not in self.pattern_index:
            return None
        
        dimension, _ = self.pattern_index[pattern_id]
        return self.patterns[dimension].get(pattern_id)
    
    def find_patterns_by_tag(self, tag: str, max_results: int = 10) -> List[KnowledgePattern]:
        """
        Find patterns by tag
        
        Args:
            tag: Tag to search for
            max_results: Maximum number of results
            
        Returns:
            List of matching patterns
        """
        matching_ids = self.tag_index.get(tag, set())
        
        # Get patterns
        results = []
        for pattern_id in matching_ids:
            if pattern_id in self.pattern_index:
                dimension, _ = self.pattern_index[pattern_id]
                pattern = self.patterns[dimension].get(pattern_id)
                if pattern:
                    results.append(pattern)
                    
                    # Update access
                    pattern.update_access()
        
        # Sort by field strength
        results.sort(key=lambda p: p.field_strength, reverse=True)
        
        return results[:max_results]
    
    def search(self, query: str, max_results: int = 5) -> List[PatternMatchResult]:
        """
        Search for patterns matching a query string
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching patterns
        """
        return self.match_pattern(
            query=query,
            pattern_type=None,  # Match any type
            intent="search",
            max_results=max_results
        )
    
    def recommend_related_patterns(self, pattern_id: str, max_results: int = 5) -> List[PatternMatchResult]:
        """
        Recommend patterns related to a given pattern
        
        Args:
            pattern_id: Pattern ID
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended patterns
        """
        if pattern_id not in self.pattern_index:
            return []
        
        # Get pattern
        dimension, _ = self.pattern_index[pattern_id]
        pattern = self.patterns[dimension][pattern_id]
        
        # Use the pattern as a query
        if isinstance(pattern.content, str):
            query = pattern.content
        else:
            query = pattern.content
        
        # Match patterns
        matches = self.match_pattern(
            query=query,
            pattern_type=None,  # Match any type
            intent="discover",
            max_results=max_results + 1  # Add 1 to account for self-match
        )
        
        # Filter out self-match
        return [m for m in matches if m.pattern_id != pattern_id][:max_results]
    
    def get_pattern_history(self, pattern_id: str) -> Dict[str, Any]:
        """
        Get access history for a pattern
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            Pattern history information
        """
        if pattern_id not in self.pattern_index:
            return {}
        
        # Get pattern
        dimension, _ = self.pattern_index[pattern_id]
        pattern = self.patterns[dimension].get(pattern_id)
        
        if not pattern:
            return {}
        
        # Get connected patterns
        connected_patterns = []
        for connected_id in self.resonance_network.get(pattern_id, set()):
            if connected_id in self.pattern_index:
                conn_dimension, _ = self.pattern_index[connected_id]
                conn_pattern = self.patterns[conn_dimension].get(connected_id)
                
                if conn_pattern:
                    connected_patterns.append({
                        "id": connected_id,
                        "type": conn_pattern.pattern_type.value,
                        "dimension": conn_pattern.signature.dimension,
                        "field_strength": conn_pattern.field_strength
                    })
        
        # Create history
        history = {
            "id": pattern_id,
            "type": pattern.pattern_type.value,
            "dimension": dimension,
            "created": pattern.created_timestamp,
            "access_count": pattern.access_count,
            "last_access": pattern.last_access_timestamp,
            "field_strength": pattern.field_strength,
            "tags": pattern.tags,
            "connected_patterns": connected_patterns,
            "coherence_history": pattern.coherence_history
        }
        
        return history


def test_dimensional_pattern_matcher():
    """Test the dimensional pattern matcher"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pattern matcher
    matcher = DimensionalPatternMatcher()
    
    # Store some test patterns
    print("\nStoring test patterns...")
    
    # Store a concept in 5D
    concept_id = matcher.create_concept_pattern(
        concept_name="Quantum Computing",
        definition="Computing that uses quantum mechanical phenomena such as superposition and entanglement.",
        related_concepts=["Quantum Mechanics", "Superposition", "Entanglement"],
        tags=["computing", "quantum", "technology"],
        dimension="5D"
    )
    print(f"Stored concept with ID: {concept_id}")
    
    # Store a related concept in 5D
    concept2_id = matcher.create_concept_pattern(
        concept_name="Quantum Entanglement",
        definition="Quantum phenomenon where pairs of particles are generated in such a way that the quantum state of each particle cannot be described independently.",
        related_concepts=["Quantum Computing", "Quantum Mechanics", "Non-locality"],
        tags=["quantum", "physics", "entanglement"],
        dimension="5D"
    )
    print(f"Stored related concept with ID: {concept2_id}")
    
    # Create a relationship between the concepts
    relationship_id = matcher.create_relationship_pattern(
        source_id=concept_id,
        target_id=concept2_id,
        relationship_type="related_to",
        description="Quantum Computing relies on Quantum Entanglement as a key phenomenon",
        tags=["quantum", "relationship"],
        dimension="5D"
    )
    print(f"Created relationship with ID: {relationship_id}")
    
    # Store some text in 4D
    text_id = matcher.store_pattern(
        content="Quantum computing is a type of computation that harnesses the collective properties of quantum states to perform calculations.",
        pattern_type=PatternType.TEXT,
        tags=["quantum", "computing", "definition"],
        dimension="4D"
    )
    print(f"Stored text with ID: {text_id}")
    
    # Store a tensor in 3D
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tensor_id = matcher.store_pattern(
        content=tensor,
        pattern_type=PatternType.TENSOR,
        tags=["tensor", "numeric"],
        dimension="3D"
    )
    print(f"Stored tensor with ID: {tensor_id}")
    
    # Test pattern matching
    print("\nTesting pattern matching...")
    
    # Match with a query
    query = "What is quantum computing and how does it use entanglement?"
    matches = matcher.search(query)
    
    print(f"Found {len(matches)} matches for query: {query}")
    for i, match in enumerate(matches):
        print(f"  {i+1}. {match.pattern_id} (Resonance: {match.resonance:.4f})")
        print(f"     {match.content if isinstance(match.content, str) else type(match.content)}")
    
    # Test traversal
    print("\nTesting resonance network traversal...")
    related = matcher.traverse_resonance_network(concept_id)
    
    print(f"Found {len(related)} patterns related to {concept_id}")
    for i, rel in enumerate(related):
        print(f"  {i+1}. {rel.pattern_id} (Resonance: {rel.resonance:.4f})")
        if isinstance(rel.content, dict) and "name" in rel.content:
            print(f"     {rel.content['name']}")
        else:
            print(f"     {type(rel.content)}")
    
    # Test amplification
    print("\nTesting pattern amplification...")
    new_strength = matcher.amplify_pattern(concept_id)
    print(f"Amplified {concept_id} to field strength {new_strength:.2f}")
    
    # Test knowledge map
    print("\nGetting knowledge map...")
    knowledge_map = matcher.get_dimensional_knowledge_map()
    
    print("Knowledge Map Summary:")
    for dimension, dim_data in knowledge_map.items():
        print(f"  {dimension}: {dim_data['pattern_count']} patterns")
        for pattern_type, patterns in dim_data['patterns_by_type'].items():
            print(f"    {pattern_type}: {len(patterns)} patterns")
    
    # Save patterns
    print("\nSaving patterns...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    matcher.save_patterns(os.path.join(results_dir, "dimensional_patterns.json"))

    print("\nPattern matching test complete!")


if __name__ == "__main__":
    test_dimensional_pattern_matcher()