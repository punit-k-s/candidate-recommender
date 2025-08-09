# Candidate Recommender - Resume Ranking + On-Demand Fit Summaries

A lightweight resume-screening app that:

- embeds a Job Description (JD) and a pile of resumes,
- ranks candidates by semantic similarity,
- shows the top-K with relevance scores,
- and generates a short AI *"why this person fits"* summary for the #1 ranked resume.

Built with **Python**, **Streamlit**, **FastAPI utilities**, **OpenAI `text-embedding-3-small`** for embeddings, and an OpenAI chat model for the summary.

---

## Why this approach?

We follow the mainstream research pattern for automated resume screening: convert both the JD and resumes to **dense vector embeddings** and rank with **cosine similarity**. Transformer models and sentence embeddings (e.g., SBERT/BERT families) are a widely used foundation for this problem, consistently outperforming keyword rules and being robust to format/noise. 

Recent work shows embedding-based matching aligns better with human judgments than classic ATS keyword filters; cosine similarity over resume/JD vectors is a standard scoring choice. 

We specifically cite **"Resume Screening Using Large Language Models"** (ICAST 2023), which argues for **vector embeddings + LLMs** for ranking and semantic similarity search in screening pipelines - exactly what we implement here (but using OpenAI's `text-embedding-3-small` for an inexpensive, fast baseline).


---

## Features

- **Drop in a JD + upload multiple resumes (PDF/DOCX/TXT)**  
- **Top-K ranking** using cosine similarity over `text-embedding-3-small` vectors
- **Evidence-based fit summary** for the top candidate (3-5 bullets + a short risk note)
- **Zip download** of the original top-K files
- **Token-aware truncation** for long JDs/resumes to keep requests cheap/reliable

---

## Architecture (high level)

1. **Parse** resumes -> plain text.  
2. **Embed** JD + each resume with `text-embedding-3-small`.  
3. **Score** cosine similarity JD<->resume -> rank.  
4. **Display** a results table (candidate, score, key metadata).  
5. **Explain** the #1 match via a small chat-completion prompt.  

This mirrors "shortlist -> rank with cosine similarity -> summarize/justify" described in the literature.

---

## Repo layout

```plaintext
CANDIDATE-RECOMMENDER/
├── app/
│   └── app.py                  # Streamlit entry point
│
├── src/
│   ├── embeddings/
│   │   ├── base.py
│   │   └── openai_client.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── fit_summary.py      # Generates AI fit summaries
│   │   ├── io_utils.py
│   │   ├── parsing.py
│   │   ├── ranking.py
│   │   └── state.py
│
├── .env.example                # Example environment variables
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Getting started

### 1) Clone

```bash
git clone https://github.com/<you>/candidate-recommender.git
cd candidate-recommender
```

### 2) Python env

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Set secrets

Copy `.env.example` → `.env` and set your key:

```env
OPENAI_API_KEY=sk-...           # required
# optional overrides
# LLM_MODEL=gpt-4o-mini
# EMBED_MODEL=text-embedding-3-small
```

### 4) Run

```bash
streamlit run app.py
```

Open the local URL Streamlit prints. Paste your **JD**, upload a folder or multiple **resumes**, hit **Compute similarity**.

---

## How ranking works (in this repo)

- We embed JD and resumes with **`text-embedding-3-small`** (cheap/fast).  
- We compute **cosine similarity** JD→resume and sort descending.  
- We display **Top-K** with scores.  
- For the highest-scored candidate, the engine automatically generates an AI fit summary after the ranked results are displayed. It uses the truncated job description and the candidate’s full parsed resume to produce a 120–180 word verdict with bullet points and a brief risk/mitigation note.

---

## Example prompt (fit summary)

We keep it short, structured, and evidence-driven:

- one-line verdict,  
- 3–5 bullets mapping **JD requirements → concrete resume evidence**,  
- close with one risk + mitigation.

This is implemented in `src/fit_summary.py` and called from `app.py` for the #1 candidate.

---

## Reproducibility & limits

- **Determinism:** embeddings are deterministic; the summary uses a low temperature.  
- **Token limits:** we truncate JD/resume text before sending to the LLM.  
- **Bias & fairness:** any automated screening can surface bias; embeddings reduce keyword brittleness but you should still review shortlists with human oversight.

---

## Running on your own data

1) Put your JD in the left panel.  
2) Drag/drop a batch of resumes (PDF/DOCX/TXT).  
3) Choose K (top 5/10).  
4) Click **Compute similarity**.  
5) After the ranked results appear, the app automatically generates an AI fit summary for the #1 candidate.  
6) Click **Download top-K** to export the original files as a zip.

---

## Configuration

Set these via `.env`:

- `OPENAI_API_KEY` — required  
- `EMBED_MODEL` — default `text-embedding-3-small` (recommended baseline)  
- `LLM_MODEL` — default `gpt-4o-mini` (concise + inexpensive)

---

## Benchmarks / why `text-embedding-3-small`?

The project targets *practical* throughput with strong semantic matching. Embedding-based ranking is well-documented as effective for resume↔JD matching and scales better than keyword rules; cosine similarity is the standard comparison.

If you want even higher recall/precision, swap to a larger embedding model (cost↑), or add **re-rank** with a cross-encoder / LLM judging pass.

---

## Tests

- Unit tests for text cleaning & cosine ranking (`src/preprocess.py`, `src/ranking.py`).  
- Smoke test: tiny sample JD + 3 resumes → stable ordering, non-empty top-1 summary.

Run:

```bash
pytest -q
```

---

## Troubleshooting

- **403 pushing to GitHub** → ensure HTTPS Remote uses your GitHub username + a **fine-grained PAT** if 2FA is on; or use SSH.  
- **OpenAI errors** → check `OPENAI_API_KEY`, and that you didn’t exceed rate/size limits.  
- **Blank summary** → resume/JD were too short after truncation; try a smaller K or fewer files.

---

## Roadmap

- Re-rank top-N with a cross-encoder  
- Optional skill extraction and facet filters  
- Batch caching of embeddings (sqlite/vector DB) for speed  
- Multi-JD compare / panels

---


## References (selected)

- **Resume Shortlisting and Ranking with Transformers (SBERT)** — sentence embeddings + cosine similarity for JD/candidate ranking.  
- **Resume Screening Using Large Language Models (ICAST 2023)** — advocates vector embeddings + LLMs; discusses similarity calculations and vector search.  
- **Resume2Vec (Electronics 2025)** — embedding-based ATS improvements; shows benefits over keyword-based filters.

---

### How to cite this project in your write-up

> “We follow the embedding-based resume screening pipeline described in the literature—encode both resumes and the job description with transformer embeddings and rank by cosine similarity—using OpenAI’s `text-embedding-3-small` for efficiency, then optionally generate a short LLM summary for the top candidate.”
