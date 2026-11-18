# AI-Powered CHRIST University Student Helpdesk: LLM vs RAG vs LLMOps

LCA Activity: Experiential and Problem-Based Learning (Explore LLM / RAG / LLMOps).
Everything runs **locally** on your laptop with [Ollama](https://ollama.com). You don't need an API key, and no data leaves your machine.

| Model | What it is | File |
|---|---|---|
| **Model 1: LLM only** | A local LLM (`llama3.2:3b`) answers from what it learned in training | `src/helpdesk/bots.py` → `BaselineLLMBot` |
| **Model 2: RAG** | The same LLM answers from CHRIST documents retrieved through hybrid search (ChromaDB vectors + BM25, fused with RRF) | `RAGBot`, `knowledge_base.py` |
| **Model 3: LLMOps** | The RAG pipeline plus guardrails, PII masking, a versioned prompt registry, semantic cache, calibrated confidence gate with human escalation, groundedness check, SQLite tracing, feedback, MLflow tracking and a monitoring dashboard | `LLMOpsBot`, `guardrails.py`, `observability.py`, `prompts/registry.yaml` |

---

## 1. Setup (one time, about 10 minutes)

**Requirements:** Python 3.10 to 3.12, 8 GB RAM, about 4 GB of free disk.

1. **Install Ollama** from https://ollama.com/download (Windows/macOS/Linux). Then, in a terminal:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
   On a low-RAM laptop, use `qwen2.5:1.5b` instead and set `chat_model` in `config.yaml` to match.

2. **Install the Python packages** from the project folder:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux
   pip install -r requirements.txt
   ```

3. **Put your details in `config.yaml`** under `report:`. Fill in the course name/code and faculty name. These go on the report's cover page.

## 2. Run it

```bash
python scripts/ingest.py                 # builds the vector + BM25 index from data/kb/
streamlit run app.py                     # chat UI: pick a model, compare all 3, monitoring tab
python scripts/chat_cli.py --model rag   # or chat in the terminal (llm | rag | llmops)
```

## 3. Run the experiments and generate the report

```bash
python scripts/run_experiments.py            # full run: about 20-40 min on a laptop CPU
python scripts/run_experiments.py --quick    # faster subset: about 8-12 min
```

This runs:

* **E0**: calibrates the escalation threshold on a separate calibration set
* **E1**: compares Model 1, 2 and 3 on the golden set (accuracy, LLM-judge, groundedness, hallucinated numbers, out-of-scope refusals, prompt-injection safety, PII protection, retrieval Hit@1/MRR, latency, tokens)
* **E2**: retrieval ablation (chunk size 300/700/1200 × BM25/dense/hybrid)
* **E3**: prompt A/B test (prompt v1 vs v2 inside Model 3)
* **E4**: semantic-cache test on paraphrased questions, plus the monitoring summary

Outputs go to `results/`:

* **`Helpdesk_Chatbot_LLM_RAG_LLMOps_Report.docx`**: the complete report, with your real numbers, charts, sample answers, interpretations and recommendations. **This is the file you submit.**
* `results.json`, `responses.csv`, `fig_*.png`

To see the MLflow experiment tracking:
```bash
mlflow ui --backend-store-uri sqlite:///storage/mlflow.db     # open http://127.0.0.1:5000
```

To rebuild the report without re-running anything: `python scripts/build_report.py`

## 4. Project structure

```
config.yaml                 models, retrieval, LLMOps settings, report cover details
prompts/registry.yaml       versioned prompts (baseline, v1 naive RAG, v2 grounded RAG)
data/kb/*.md                knowledge base (front-matter holds the source URL and verified flag)
data/eval/golden_qa.json    31 evaluation questions with reference answers
data/eval/calibration.json  separate questions used only to calibrate the escalation threshold
src/helpdesk/
  llm_client.py             Ollama chat + embeddings client
  knowledge_base.py         chunking, ChromaDB + BM25 indexing, hybrid RRF retrieval
  bots.py                   the three models
  guardrails.py             injection/harm detection, PII masking
  metrics.py                accuracy, groundedness, hallucinated numbers, retrieval metrics, LLM judge
  observability.py          SQLite tracing, feedback, monitoring summary
scripts/                    ingest, CLI chat, experiments, charts, report builder
app.py                      Streamlit UI
tests/mock_ollama.py        TEST-ONLY fake Ollama server, for checking the code without models
```

## 5. Notes

* **Knowledge base:** compiled from public CHRIST pages (general regulations, admissions contacts, exam notifications, CCHS). `07_student_services_illustrative.md` is placeholder text and is marked `verified: false`. To extend the bot, add `.md` files to `data/kb/` and re-run `ingest.py`.
* **Mock server:** `tests/mock_ollama.py` is only for checking that the code runs. Reports produced with it are stamped **MOCK TEST** and their numbers mean nothing. Run with real Ollama before you submit.
* **Troubleshooting:**
  * `Cannot reach Ollama`: start the Ollama app or run `ollama serve`.
  * `Missing models`: run the `ollama pull` commands above.
  * Very slow runs: use `--quick`, `--no-judge`, or a smaller chat model.
