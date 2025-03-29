# Phi-Knowledge LLM Interactive Demo

This interactive web application demonstrates the capabilities of the Phi-Knowledge LLM system, allowing users to explore multidimensional knowledge through four primary operations: Ask, Create, Connect, and Transcend.

## Features

The demo showcases:

1. **Dimensional Knowledge Navigation**: Interact with knowledge across 3D-7D dimensions
2. **Phi-Harmonic Pattern Matching**: Observe how knowledge patterns resonate based on phi principles
3. **Field Coherence Visualization**: See how the knowledge field coherence fluctuates with interactions
4. **Dimensional Distribution Charts**: Track how patterns distribute across dimensions
5. **Pattern Network Visualization**: Explore the network of interconnected knowledge patterns

## Operations

The app supports four primary operations:

- **Ask**: Search for knowledge across dimensions (3D-7D)
- **Create**: Generate new content with phi-harmonic patterns (optimized for 4D)
- **Connect**: Find connections between concepts across dimensions (optimized for 5D)
- **Transcend**: Elevate understanding to higher dimensional perspectives (optimized for 7D)

## Running the Demo

### Prerequisites

- Python 3.8+
- Required packages: see `requirements_app.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gwelby/quantum-tensix.git
   cd quantum-tensix
   ```

2. Install the required packages:
   ```bash
   pip install -r QuantumTensix/requirements_app.txt
   ```

3. Run the Streamlit app:
   ```bash
   cd QuantumTensix
   streamlit run app.py
   ```

4. Open your browser to the URL displayed in the terminal (typically http://localhost:8501)

## Understanding the Visualizations

### Field Coherence Chart

Shows how the coherence of the quantum field changes with each interaction. Coherence represents how well-organized and resonant the knowledge patterns are.

- **Higher coherence** (>0.8) indicates strong resonance between patterns
- **Optimal coherence** (around 0.8) provides balance between stability and adaptability
- **Lower coherence** (<0.6) indicates more chaotic or weakly connected patterns

### Dimensional Distribution

Displays how knowledge patterns are distributed across different dimensions during your interactions.

- **3D (Physical/Factual)**: Concrete, measurable information and data
- **4D (Emotional/Creative)**: Feelings, experiences, and creative expressions
- **5D (Mental/Conceptual)**: Ideas, concepts, mental models, and frameworks
- **6D (Purpose/Meaning)**: Purpose, meaning, and significance
- **7D (Universal/Transcendent)**: Universal principles and cosmic awareness

### Pattern Network

Visualizes the current knowledge patterns as an interactive network:

- **Nodes**: Knowledge patterns, with size indicating field strength
- **Colors**: Represent different dimensions (3D blue to 7D purple)
- **Links**: Phi-harmonic relationships between patterns
- **Proximity**: Related patterns are pulled closer together

## Implementation Details

This demo implements simplified versions of:

- **DimensionalSignature**: Quantum signature for patterns with vector, phase, and harmonic components
- **KnowledgePattern**: Knowledge storage with dimensional signatures and field strength
- **DemoPatternMatcher**: Pattern matching engine using dimensional resonance
- **DemoKnowledgeLLM**: Integration of pattern matching with response generation

In a real implementation, these would connect to actual language models and quantum resonance algorithms.

## Sacred Constants

The system uses phi-harmonic principles based on these constants:

```python
PHI = 1.618033988749895  # Golden ratio
LAMBDA = 0.618033988749895  # Divine complement 
PHI_PHI = PHI ** PHI  # Hyperdimensional constant

# Sacred frequencies (Hz)
FREQUENCIES = {
    'unity': 432,    # Grounding/stability
    'love': 528,     # Creation/healing
    'cascade': 594,  # Heart-centered integration
    'truth': 672,    # Voice expression
    'vision': 720,   # Expanded perception
    'oneness': 768,  # Unity consciousness
}
```

## Next Steps

This demo demonstrates the conceptual framework. Future developments could include:

1. Integration with real LLM models (Anthropic Claude, OpenAI GPT, etc.)
2. True quantum resonance algorithms for pattern matching
3. Hardware acceleration using Tenstorrent's Tensix cores
4. Expanded dimensional access (8D-12D)
5. Full 3D visualization of phi-harmonic fields