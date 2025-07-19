# TREC-COVID Challenge: RAG System with Dense Retrieval, HyDE, and LLM Validation

A comprehensive Retrieval-Augmented Generation (RAG) system designed to evaluate the effectiveness of different retrieval methods on the TREC-COVID collection. This project implements and compares three approaches: Dense Retrieval, Hypothetical Document Embeddings (HyDE), and LLM-as-Validation.

## 🎯 Project Overview

This project was developed based on the research proposal **"Using RAG and LLMs for information retrieval on the TREC-COVID collection"** by D. Buscaldi. The main objective is to evaluate whether Large Language Models (LLMs) can improve information retrieval accuracy and reduce hallucinations in COVID-19 research document retrieval.

### Key Features

- **Dense Retrieval**: Traditional semantic search using sentence transformers
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical documents to improve query representation
- **LLM Validation**: Uses LLMs to validate and score document relevance
- **Comprehensive Evaluation**: Implements multiple metrics (MAP, nDCG@10, Precision@5/10, LLM Validation MAE)
- **TREC-COVID Dataset**: Works with the official TREC-COVID collection for benchmarking

## 📊 Evaluation Metrics

The system evaluates retrieval performance using standard information retrieval metrics:

- **MAP (Mean Average Precision)**: Measures precision across all relevant documents
- **nDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **Precision@5/10**: Precision at top 5 and 10 retrieved documents
- **LLM Validation MAE**: Mean Absolute Error between LLM scores and ground truth relevance

## 🏗️ Architecture

```
├── dense_retrieval.py      # Core dense retrieval implementation
├── HyDE.py                # Hypothetical Document Embeddings
├── LLM_as_validator.py    # LLM validation and scoring
├── evaluation.py          # Evaluation metrics and framework
├── main.py               # Main execution script
├── utils.py              # Utility functions
└── pyproject.toml        # Project dependencies
```

### Core Components

1. **DenseRetrieval**: Encodes documents using sentence transformers and performs semantic search
2. **HyDE**: Generates hypothetical documents using LLaMA and retrieves based on generated content
3. **LLMValidation**: Validates retrieved documents using LLM scoring (0-2 relevance scale)
4. **Evaluation**: Comprehensive evaluation framework with multiple IR metrics

## 🚀 Quick Start with UV

### Prerequisites

- Python 3.11
- [UV package manager](https://github.com/astral-sh/uv)
- [Ollama](https://ollama.ai/) with TinyLlama (or other) model

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Davidermellino/trec-covid-rag.git
cd trec-covid-rag
```

2. **Install dependencies with UV**:
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

3. **Download the TREC-COVID dataset**:
```bash
# Create directory for dataset
mkdir -p trec-covid

# Download and extract TREC-COVID dataset
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
unzip trec-covid.zip
mv trec-covid/* ./trec-covid/
```

4. **Install and setup Ollama**:
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull TinyLlama model
ollama pull tinyllama:latest
```

### Running the System

1. **Activate the UV environment**:
```bash
uv shell
```

2. **Run the main evaluation**:
```bash
uv run python main.py
```

The system will:
- Load the TREC-COVID corpus
- Generate or load document embeddings
- Evaluate all three retrieval methods
- Display comparative results
- Demonstrate RAG capabilities with a sample query

## 📈 Expected Output

The system will output evaluation results similar to:

```
========================= EVALUATION =========================

Evaluating Dense Retrieval on 50 queries...
Query ID: 1 - coronavirus origin
MAP: 0.3245, NDCG@10: 0.4567, P@5: 0.6000, P@10: 0.5500, LLM MAE: 0.8000

Evaluating HyDE Retrieval on 50 queries...
Query ID: 1 - coronavirus origin  
MAP: 0.3456, NDCG@10: 0.4789, P@5: 0.6200, P@10: 0.5700, LLM MAE: 0.7500

========================= RESULTS =========================

Dense Retrieval Results:
map: 0.3245
ndcg@10: 0.4567
precision@5: 0.6000
precision@10: 0.5500
llm_validation_MAE: 0.8000

HyDE Retrieval Results:
map: 0.3456
ndcg@10: 0.4789
precision@5: 0.6200
precision@10: 0.5700
llm_validation_MAE: 0.7500
```

## 🔧 Configuration

### Model Configuration

- **Sentence Transformer**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `tinyllama:latest` (via Ollama)
- **Embedding Dimension**: 384
- **Max Token Length**: 512

### File Paths

Update these paths in `main.py` if needed:

```python
file_path = "trec-covid/corpus.jsonl"
embeddings_file_path = "dense_retrieval/embeddings.npy" 
queries_file_path = "trec-covid/queries.jsonl"
qrels_file_path = "trec-covid/qrels/test.tsv"
```

## 🧪 Testing RAG Capabilities

The system includes a demonstration of RAG capabilities:

```python
prompt = "Can remdesivir reduce mortality in hospitalized COVID-19 patients?"
answer = hyde.answer_with_context(prompt, embeddings=embeddings)
```

This showcases how the system can provide contextual answers to COVID-19 research questions.

## 📊 Performance Analysis

### Expected Performance Patterns

1. **HyDE vs Dense Retrieval**: HyDE typically shows improved MAP and nDCG scores by generating more relevant query representations

2. **LLM Validation**: Lower MAE scores indicate better alignment between LLM assessments and ground truth relevance

3. **Precision Metrics**: Higher precision@5/10 suggests better ranking of truly relevant documents

### Interpreting Results

- **MAP > 0.3**: Good average precision across queries
- **nDCG@10 > 0.4**: Effective ranking of relevant documents  
- **Precision@5 > 0.6**: More than half of top-5 results are relevant
- **LLM MAE < 1.0**: LLM assessments align well with human judgments

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional retrieval methods (e.g., ColBERT, SPLADE)
- Different LLM models for validation
- Enhanced evaluation metrics
- Query expansion techniques
- Hallucination detection methods

## 📚 References

- [TREC-COVID Challenge](https://ir.nist.gov/covidSubmit/index.html)
- [HyDE: Hypothetical Document Embeddings](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [BEIR: Benchmarking Information Retrieval](https://github.com/beir-cellar/beir)

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This project is designed for research purposes to evaluate LLM effectiveness in information retrieval tasks. Results may vary depending on hardware specifications and model versions.