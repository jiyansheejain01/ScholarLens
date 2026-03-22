# ScholarLens 🔬

**Automated Scientific Quality Assessment & Missing Citation Detection**

ScholarLens is a college AI project that takes a research paper (PDF), scores its scientific quality, and suggests recent papers from arXiv that the author may have missed citing — a task that is genuinely difficult for human reviewers to keep up with.

---

## What It Does

| Feature | Description |
|---|---|
| Quality Assessment | Scores coherence, completeness, and novelty of a paper's abstract |
| Gap Analysis | Finds recent arXiv papers (last 6 months) semantically similar to your paper but not cited |
| PDF Upload | Upload any research PDF — no copy-pasting required |
| Web UI | Clean dark-themed browser interface built with Gradio |

---

## Project Structure

```
ScholarLens/
├── config/
│   └── config.yaml          # All settings (model, thresholds, paths)
├── data/
│   ├── raw/                 # Original OpenReview dataset files
│   ├── processed/           # Cleaned paper-review pairs
│   └── embeddings/          # Cached SciBERT embeddings
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Loads & preprocesses OpenReview dataset
│   ├── encoder.py           # SciBERT encoder (mean pooling)
│   ├── arxiv_search.py      # Queries arXiv API for recent papers
│   ├── citation_ranker.py   # Ranks candidates by cosine similarity
│   ├── quality_assessor.py  # Scores coherence, completeness, novelty
│   └── pipeline.py          # Ties all modules together
├── tests/
│   ├── test_encoder.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
├── .env                     # API keys (not committed to git)
├── .gitignore
├── app.py                   # Gradio web UI — run this
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/ScholarLens.git
cd ScholarLens
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the root:

```
SEMANTIC_SCHOLAR_API_KEY=your_key_here
ARXIV_EMAIL=your_email@example.com
```

Get a free Semantic Scholar key at: https://www.semanticscholar.org/product/api

> arXiv does not require an API key. The `arxiv` Python library works out of the box.

---

## Running the App

### Web UI (recommended)

```bash
python app.py
```

Then open your browser at: **http://localhost:7860**

Upload any research paper PDF and click **Analyse Paper**.

### Command Line

```bash
python main.py
```

Edit the `title` and `abstract` variables in `main.py` to test with your own paper.

---

## How It Works

```
PDF Upload
    ↓
T5 / SciBERT Encoder  →  Paper Embedding
    ↓                          ↓
Quality Assessor         arXiv Semantic Search
    ↓                          ↓
Quality Score          Missing Citation Ranking
    └──────────┬───────────────┘
               ↓
         Review Report
```

1. **PDF Parsing** — `pypdf` extracts the title and abstract from the uploaded paper
2. **Encoding** — SciBERT (`allenai/scibert_scivocab_uncased`) encodes the paper into a semantic vector
3. **arXiv Search** — keywords are extracted and used to query arXiv for papers published in the last 6 months
4. **Citation Ranking** — cosine similarity between the paper embedding and each arXiv result ranks missing citations
5. **Quality Scoring** — rule-based scoring across coherence, completeness, and novelty produces a final grade

---

## Configuration

All tunable parameters are in `config/config.yaml`:

```yaml
model:
  name: "allenai/scibert_scivocab_uncased"
  max_length: 512
  batch_size: 8
  device: "cpu"          # change to "cuda" for GPU

arxiv:
  months_back: 6         # how far back to search
  max_results: 50        # arXiv papers fetched per query

ranking:
  top_k: 10              # max missing citations to return
  similarity_threshold: 0.75   # minimum cosine similarity

quality:
  weights:
    coherence: 0.4
    completeness: 0.3
    novelty: 0.3
```

---

## Validation Metric

**Citation Recall** — out of all the missing citations a human reviewer suggested, how many did ScholarLens correctly identify?

$$\text{Citation Recall} = \frac{|\text{predicted} \cap \text{human-suggested}|}{|\text{human-suggested}|}$$

Ground truth labels come from the **OpenReview dataset** (ICLR / NeurIPS / ICML paper-review pairs).

---

## Dependencies

| Package | Purpose |
|---|---|
| `transformers` | SciBERT model loading |
| `torch` | Model inference |
| `arxiv` | arXiv API client |
| `scikit-learn` | Cosine similarity |
| `gradio` | Web UI |
| `pypdf` | PDF text extraction |
| `datasets` | OpenReview dataset |
| `pyyaml` | Config loading |
| `python-dotenv` | `.env` file loading |

---

## References

- Wahle et al. (2022). *Identifying Relevant Literature for Scholarly Papers*
- Lo et al. (2020). *S2ORC: The Semantic Scholar Open Research Corpus*
- Analysis of LLM use across the peer review pipeline (arXiv, Jan 2026)
- OpenReview Dataset: https://huggingface.co/datasets/aclanthology/openreview

---