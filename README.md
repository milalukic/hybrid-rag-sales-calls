# Basic RAG (Retrieval-Augmented Generation) 

## What each file does

- **`chunking.py`** — splits each document into fixed-size, non-overlapping word chunks.
- **`retrieval.py`** — embeds every chunk with a small pretrained model, and finds the chunks most similar in meaning to a question via cosine similarity.
- **`generate.py`** — sends the retrieved chunks + the question to Claude, instructed to answer only from that context. Falls back to returning the raw top chunk if no API key is set.
- **`main.py`** — a simple CLI tying the three steps together.

## Setup

```bash
cd path/to/rag_project

# Recommended - virtual environment
python3 -m venv venv
source venv/bin/activate 

pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional; omit to use offline fallback
```

## Usage

```bash
python3 main.py "why did halberg retail group churn?"
python3 main.py "discount negotiation" --stage in_negotiation
python3 main.py "onboarding issues" --customer Nordwind
python3 main.py "pricing pushback" --stage in_negotiation --customer Birkenstadt --top_k 5
```
