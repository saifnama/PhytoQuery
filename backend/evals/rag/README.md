# RAG Evaluation Guide

End-to-end guide for evaluating PhytoQuery's RAG chatbot.

## Prerequisites

1. Qdrant run:

2. `.env` configured with your LLM provider (Ollama, llama.cpp, or OpenRouter).

3. Python venv with dependencies:
   ```bash
   uv pip install -r backend/requirements.txt --python pq\Scripts\python.exe
   ```

## Step 1: Ingest PDFs

Place PDF files in `backend/knowledge_base/papers/`, then run:

```bash
python scripts/ingest.py
```

This parses the PDFs, chunks them, embeds them, and stores them in the `kb_papers` Qdrant collection.

To check what's indexed:
```bash
python scripts/ingest.py --status
```

## Step 2: Prepare Questions

Questions live in `backend/evals/rag/data/`. Two formats:

**MCQ** (`mcq.csv`):
```csv
id,question,option_a,option_b,option_c,option_d,correct_option,difficulty
1,"What species was studied?","Rosmarinus","Lippia alba","Mentha","Origanum",B,easy
```

**Open-ended** (`open_ended.csv`):
```csv
id,question,reference,difficulty
1,"What are the main bioactive compounds?","The main compounds are neral and geranial.",easy
```

If your questions are in JSON, convert them:

```bash
python scripts/json_to_csv.py your_questions.json
```

## Step 3: Generate RAG Answers

Run the appropriate script to query the RAG and save answers:

**MCQ:**
```bash
python backend/evals/rag/rag_mcq.py
```

**Open-ended:**
```bash
python backend/evals/rag/rag_open_ended.py
```

Output goes to `backend/evals/rag/results/`.

Checkpoint/resume is automatic — if interrupted, re-running continues from where it left off.

## Step 4: Score the Answers

**MCQ (regex accuracy):**
```bash
python backend/evals/rag/eval_mcq.py
```

Prints overall accuracy and accuracy by difficulty. Output: `mcq_eval_results.csv`.

**Open-ended (5 Ragas metrics):**
```bash
python backend/evals/rag/eval_open_ended.py
```

Prints averages for Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall. Output: `open_ended_eval_results.csv`.

## Configuration

All settings are env-var driven. Defaults work out of the box if `.env` is configured for the RAG engine.

| Env Var | Default | Purpose |
|---------|---------|---------|
| `EVAL_OUTPUT_DIR` | `backend/evals/rag/results` | Where CSVs go |
| `EVAL_MCQ_CSV` | `backend/evals/rag/data/mcq.csv` | MCQ questions input |
| `EVAL_OPEN_ENDED_CSV` | `backend/evals/rag/data/open_ended.csv` | Open-ended questions input |
| `RAGAS_LLM_MODEL` | same as RAG engine | Judge LLM for Ragas |
| `RAGAS_LLM_BASE_URL` | same as RAG engine | Judge LLM endpoint |
| `RAGAS_LLM_API_KEY` | same as RAG engine | Judge LLM API key |
| `RAGAS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embeddings for Ragas |

The evaluator LLM automatically inherits whatever provider is configured in `.env` (llamacpp > openrouter > ollama). Override with `RAGAS_LLM_*` vars if you want a different model as judge.

## File Structure

```
backend/evals/rag/
  rag_mcq.py              Run RAG on MCQ questions
  rag_open_ended.py       Run RAG on open-ended questions
  eval_mcq.py             Score MCQ answers (regex)
  eval_open_ended.py      Score open-ended answers (Ragas)
  data/
    mcq.csv               MCQ question bank
    open_ended.csv         Open-ended question bank
  results/
    rag_outputs_mcq.csv   RAG answers for MCQ
    rag_outputs_open_ended.csv  RAG answers for open-ended
    mcq_eval_results.csv  MCQ scores
    open_ended_eval_results.csv  Open-ended scores
scripts/
  json_to_csv.py          Convert JSON QA datasets to CSV
  ingest.py               Ingest PDFs into Qdrant
```
