# Basic RAG (Retrieval-Augmented Generation) 

## What each file does

- **`chunking.py`** — splits each document into fixed-size, non-overlapping word chunks.
- **`retrieval.py`** — embeds every chunk with a small pretrained model, and finds the chunks most similar in meaning to a question via cosine similarity.
- **`generate.py`** — sends the retrieved chunks + the question to Claude, instructed to answer only from that context. Falls back to returning the raw top chunk if no API key is set.
- **`main.py`** — a simple CLI tying the three steps together.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here   # optional; omit to use offline fallback
```

## Usage

```bash
python main.py "why did halberg retail group churn?"
```
