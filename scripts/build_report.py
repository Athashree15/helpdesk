"""Generate the assignment report (Word .docx) from the experiment results.

    python scripts/build_report.py            # uses results/results.json from the last run

All numbers, tables, charts and sample answers in the report come from YOUR local run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BLUE = RGBColor(0x1C, 0x3F, 0x77)
GREY = RGBColor(0x55, 0x55, 0x55)


# ------------------------------------------------------------------ helpers
def pct(v, d=0):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def pp(a, b):
    """Percentage-point difference b - a, as text."""
    if a is None or b is None:
        return "n/a"
    d = (b - a) * 100
    return f"{'+' if d >= 0 else ''}{d:.0f} pp"


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


class Report:
    def __init__(self):
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.3))
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        st = self.doc.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(11)
        st["Normal"].paragraph_format.space_after = Pt(6)
        st["Normal"].paragraph_format.line_spacing = 1.15
        for lvl, size in ((1, 15), (2, 12.5), (3, 11.5)):
            h = st[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = BLUE
            h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
            h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
            rpr = h.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(a), "Calibri")
        self.fig_no = 0
        self.tab_no = 0

    # basic blocks
    def h(self, text, lvl=1):
        self.doc.add_heading(text, level=lvl)

    def p(self, text="", bold_lead: str | None = None, italic=False, size=None, align=None, color=None):
        para = self.doc.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        r = para.add_run(text)
        r.italic = italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
        if align:
            para.alignment = align
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            para = self.doc.add_paragraph(style=style)
            if isinstance(it, tuple):
                para.add_run(it[0]).bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def table(self, header, rows, widths=None, caption=None, font=9.5):
        if caption:
            self.tab_no += 1
            c = self.doc.add_paragraph()
            r = c.add_run(f"Table {self.tab_no}: {caption}")
            r.bold, r.font.size = True, Pt(9.5)
            c.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(htxt))
            run.bold, run.font.size = True, Pt(font)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            shade(cell, "1C3F77")
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(val))
                run.font.size = Pt(font)
                if ri % 2 == 1:
                    shade(cells[i], "F2F5FA")
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def figure(self, path, caption, width_cm=16):
        if not path or not Path(path).exists():
            return
        self.fig_no += 1
        self.doc.add_picture(str(path), width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.p(f"Figure {self.fig_no}: {caption}", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER,
                   color=GREY)
        c.paragraph_format.space_after = Pt(10)

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        self.doc.save(str(path))


def add_page_numbers(doc):
    for section in doc.sections:
        para = section.footer.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)
        run.font.size = Pt(9)


def short(text, n=260):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "..."


# ------------------------------------------------------------------ report
def build_report(cfg, results, out_dir) -> Path:
    from make_charts import make_all

    out_dir = Path(out_dir)
    figs = make_all(results, out_dir)
    meta, S, names = results["meta"], results["summary"], results["model_names"]
    m1, m2, m3 = S["llm"], S["rag"], S["llmops"]
    rep_cfg = cfg["report"]
    R = Report()
    doc = R.doc

    # ============================================================ cover
    if meta.get("mock"):
        R.p("TEST RUN WITH THE MOCK BACKEND - NUMBERS ARE NOT REAL. Re-run with Ollama before submitting.",
            bold_lead="WARNING: ", color=RGBColor(0xC0, 0x00, 0x00))
    for _ in range(3):
        doc.add_paragraph()
    R.p("CHRIST (Deemed to be University)", size=20, align=WD_ALIGN_PARAGRAPH.CENTER, color=BLUE).runs[0].bold = True
    R.p("Bengaluru, Karnataka", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    R.p(rep_cfg["department"], size=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=GREY)
    doc.add_paragraph()
    t = R.p(rep_cfg["title"], size=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    t.runs[0].bold = True
    R.p(rep_cfg["activity"], size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, color=GREY)
    for _ in range(3):
        doc.add_paragraph()
    info = [("Submitted by", rep_cfg["student_name"]), ("Register Number", rep_cfg["register_no"]),
            ("Class", rep_cfg["class"]), ("Course", rep_cfg["course"]), ("Submitted to", rep_cfg["faculty"]),
            ("Date", time.strftime("%d %B %Y"))]
    tbl = doc.add_table(rows=0, cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in info:
        cells = tbl.add_row().cells
        cells[0].text, cells[1].text = k, v
        cells[0].paragraphs[0].runs[0].bold = True
        cells[0].width, cells[1].width = Cm(5), Cm(8)
    R.page_break()

    # ============================================================ abstract
    R.h("Abstract")
    R.p(f"Students at CHRIST (Deemed to be University) repeatedly ask the same administrative questions about "
        f"attendance, leave, examinations, campuses and contacts. This activity explores how Generative AI can "
        f"answer such questions reliably. Three chatbot models were implemented and run entirely on a local "
        f"machine using Ollama: Model 1, a plain Large Language Model ({meta['chat_model']}); Model 2, a "
        f"Retrieval-Augmented Generation (RAG) system that grounds the same LLM in a curated knowledge base "
        f"stored in ChromaDB; and Model 3, an LLMOps-enabled RAG system that adds guardrails, versioned prompts, "
        f"a semantic cache, confidence-based escalation, groundedness checks, tracing, user feedback and MLflow "
        f"experiment tracking. On a golden set of {m1['n_in_scope']} in-scope questions plus out-of-scope, "
        f"adversarial and privacy test cases, answer accuracy was {pct(m1['keyword_accuracy'])} for the plain LLM, "
        f"{pct(m2['keyword_accuracy'])} for RAG and {pct(m3['keyword_accuracy'])} for the LLMOps system. "
        f"The LLMOps system declined {pct(m3['out_of_scope_decline_rate'])} of out-of-scope questions and handled "
        f"{pct(m3['adversarial_safe_rate'])} of prompt-injection attempts safely. The results show that the LLM "
        f"supplies language ability, RAG supplies institution-specific truth, and LLMOps supplies the reliability, "
        f"safety and measurability needed before a chatbot can be trusted by real students.")

    # ============================================================ 1 intro
    R.h("1. Introduction and Problem Statement")
    R.p("Every semester the class teachers, the Office of Examinations, the admissions offices and the Office of "
        "Student Affairs receive a large number of repetitive queries: \"What is the minimum attendance?\", "
        "\"I missed an exam, what can I do?\", \"What is the address of the Yeshwanthpur campus?\". The answers "
        "exist in regulations, handbooks and web pages, but they are scattered and students do not always know "
        "where to look. A helpdesk chatbot that answers instantly, cites the source and knows when to hand over to "
        "a human could reduce this load.")
    R.p("General-purpose LLMs are fluent, but they do not know CHRIST-specific rules (for example, the 85% "
        "attendance requirement differs from the common 75% UGC figure) and they tend to answer confidently even "
        "when they are wrong. The problem addressed in this Problem-Based Learning activity is therefore:")
    R.p("How can a Generative-AI student helpdesk for CHRIST give accurate, source-backed, safe answers, and what "
        "does each layer - the LLM, retrieval augmentation and LLMOps - contribute?", italic=True)

    R.h("1.1 Objectives", 2)
    R.bullets(["Implement three models locally: a plain LLM chatbot, a RAG chatbot and an LLMOps-managed RAG chatbot.",
               "Build a small, source-annotated knowledge base of CHRIST student regulations and services.",
               "Design a repeatable evaluation covering accuracy, faithfulness, hallucination, safety, privacy, "
               "latency and cost.",
               "Run controlled experiments on retrieval design (chunk size, dense vs keyword vs hybrid search) and "
               "prompt versions.",
               "Interpret the results and recommend how a university should deploy such a system."])

    # ============================================================ 2 background
    R.h("2. Background: LLM, RAG and LLMOps")
    R.h("2.1 Large Language Models (Model 1)", 2)
    R.p("An LLM is a transformer network trained to predict the next token over very large text corpora. Its "
        "knowledge is frozen in its parameters at training time. This makes it excellent at understanding questions "
        "and producing fluent language, but it has three limits for a helpdesk: it lacks private or local knowledge, "
        "its knowledge goes out of date, and it hallucinates - it produces plausible but unsupported statements. "
        f"This activity uses {meta['chat_model']}, a small open-weight model that runs on a laptop CPU through "
        "Ollama, so no student data leaves the machine.")
    R.h("2.2 Retrieval-Augmented Generation (Model 2)", 2)
    R.p("RAG (Lewis et al., 2020) separates knowledge from language ability. Documents are split into chunks and "
        "converted into embedding vectors stored in a vector database. At question time the most relevant chunks "
        "are retrieved and inserted into the prompt, so the LLM answers from evidence rather than memory. Updating "
        "the chatbot then means updating documents, not retraining a model, and every answer can cite its source.")
    R.h("2.3 LLMOps (Model 3)", 2)
    R.p("LLMOps is the set of engineering practices for running LLM applications reliably in production - the "
        "LLM-era extension of MLOps. It covers evaluation datasets and metrics, experiment tracking, prompt "
        "versioning, guardrails for safety and privacy, caching for cost and speed, observability (tracing every "
        "request), human feedback loops and fallback or escalation policies. LLMOps does not make the model "
        "smarter; it makes the system measurable, safe and maintainable.")

    # ============================================================ 3 architecture
    R.h("3. System Design and Implementation")
    R.figure(figs["architecture"], "Architecture of the helpdesk: offline ingestion and the online Model 3 pipeline.")
    R.table(["Component", "Technology", "Purpose"], [
        ["LLM runtime", f"Ollama, {meta['chat_model']} (temperature {meta['temperature']})", "Local text generation"],
        ["Embeddings", f"Ollama, {meta['embed_model']}", "Semantic vectors for chunks and questions"],
        ["Vector store", "ChromaDB (persistent, cosine)", "Dense similarity search"],
        ["Keyword search", "BM25 (rank-bm25)", "Exact-term matching (names, emails, codes)"],
        ["Fusion", "Reciprocal Rank Fusion (k = 60)", "Combine dense and BM25 rankings"],
        ["Experiment tracking", "MLflow", "Parameters, metrics and artifacts of each run"],
        ["Tracing / feedback", "SQLite", "Per-request logs, flags, latency per stage, ratings"],
        ["User interface", "Streamlit", "Chat, side-by-side comparison, monitoring dashboard"],
        ["Language", "Python 3", "Implementation of all three models and the evaluation"],
    ], widths=[3.6, 6.2, 6.6], caption="Technology stack (all components run locally and are free/open-source)")

    R.h("3.1 Knowledge Base", 2)
    R.p(f"The knowledge base contains seven markdown documents compiled from public CHRIST web pages (general "
        f"regulations, admissions contacts, examination notifications, counselling centre pages and campus "
        f"information). Each document carries front-matter with its source URL and a verification flag. Six documents "
        f"are marked verified; one (library, IT, hostel, fees, scholarships) is explicitly marked as illustrative "
        f"placeholder content, because no authoritative public text was available - this is a deliberate design "
        f"choice so that the system never presents unverified text as official. The documents were split into "
        f"{meta['n_chunks']} chunks of about {meta['chunk_size']} characters using heading-aware chunking, and every "
        f"chunk is prefixed with its document title and section heading (a contextual chunk header), which helps "
        f"short chunks get retrieved.")

    R.h("3.2 Model 1 - LLM only", 2)
    R.p("The question is sent to the local LLM with a short system prompt describing it as a CHRIST helpdesk "
        "assistant. No documents are provided, so the answer reflects only the model's pre-trained knowledge. "
        "This is the baseline that shows what an LLM can and cannot do on its own.")
    R.h("3.3 Model 2 - RAG", 2)
    R.p(f"The question is used to retrieve the top {meta['top_k']} chunks with hybrid search (dense vectors from "
        "ChromaDB fused with BM25 keyword scores via Reciprocal Rank Fusion). The retrieved chunks are numbered and "
        "placed in a simple prompt (prompt v1) before the question. This is a standard \"naive RAG\" "
        "implementation with no extra safety or operations features.")
    R.h("3.4 Model 3 - LLMOps-managed RAG", 2)
    R.p("Model 3 uses the same retriever and LLM, but wraps them in an operational pipeline:")
    R.bullets([
        ("Input guardrails: ", "regular-expression detectors block prompt-injection and jailbreak attempts and "
         "harmful requests (e.g. hacking the portal); phone numbers, e-mail addresses, Aadhaar-like numbers and "
         "register numbers are masked before the text reaches the LLM or the logs."),
        ("Semantic cache: ", "if a new question's embedding has cosine similarity of at least "
         f"{cfg['llmops']['cache_similarity']} with a previously answered question, the stored answer is returned "
         "without calling the LLM."),
        ("Confidence gate and escalation: ", "if the best retrieval similarity is below a threshold, the bot does "
         "not guess; it escalates to the relevant human office. The threshold "
         f"({num(results['calibration']['threshold'], 3)}) was calibrated automatically on a separate calibration "
         "set of 10 in-scope and 10 out-of-scope questions (balanced accuracy "
         f"{pct(results['calibration']['balanced_accuracy'])})."),
        ("Versioned prompt registry: ", f"prompts live in prompts/registry.yaml; version "
         f"{meta['prompt_version']} enforces strict grounding, numbered citations, a refusal policy and a length limit."),
        ("Output checks: ", "a groundedness score (share of answer sentences supported by the retrieved passages) "
         "is computed for every answer; weakly grounded answers receive a verification disclaimer, and any personal "
         "data in the output is scrubbed."),
        ("Observability and feedback: ", "every request is traced in SQLite with latency per stage, token counts, "
         "retrieved sources and guardrail/cache/escalation flags; users can rate answers, and the Streamlit "
         "dashboard summarises the traces."),
        ("Experiment tracking: ", "every evaluation run is logged to MLflow with its parameters, metrics and artifacts, "
         "so changes to the model, prompt or retriever can be compared over time."),
    ])

    # ============================================================ 4 setup
    R.h("4. Experimental Setup")
    cats = {}
    for r in results["records"]:
        if r["model_key"] == "llm":
            cats[r["category"]] = cats.get(r["category"], 0) + 1
    R.p("A golden evaluation set was written with reference answers and required facts for each question:")
    R.table(["Category", "Count", "What it tests"], [
        ["In-scope", cats.get("in_scope", 0), "Factual questions answerable from the knowledge base"],
        ["Out-of-scope", cats.get("out_of_scope", 0), "Questions the bot should decline or redirect (weather, IPL, "
                                                      "hostel fee not in KB, private phone number)"],
        ["Adversarial", cats.get("adversarial", 0), "Prompt injection and jailbreak attempts"],
        ["Privacy (PII)", cats.get("pii", 0), "Question containing a phone number and e-mail address"],
        ["Paraphrases", len([r for r in results["cache_test"]["records"] if r["model_key"] == "llmops"]),
         "Re-worded repeats of earlier questions (semantic-cache test)"],
    ], widths=[3.2, 1.8, 11.4], caption="Evaluation dataset")
    R.p("Metrics:", bold_lead="")
    R.bullets([
        ("Answer accuracy: ", "share of required facts (e.g. \"85\", \"Nagasandra\") present in the answer."),
        ("LLM-judge score: ", "the local LLM grades each answer against the reference on a 1-5 scale "
         "(LLM-as-a-judge)." if meta["judge"] else "not used in this run."),
        ("Groundedness: ", "share of answer sentences whose content words are mostly found in the knowledge base - "
         "a deterministic proxy for faithfulness."),
        ("Hallucinated-number rate: ", "share of answers containing a number that appears nowhere in the knowledge base."),
        ("Decline / safety / privacy rates: ", "correct refusals for out-of-scope questions, safe handling of attacks "
         "(no compliance, no system-prompt leak), and whether personal data was masked before reaching the LLM."),
        ("Retrieval Hit@1, Hit@k and MRR: ", "whether the correct source document is ranked first / in the top k, and "
         "the mean reciprocal rank."),
        ("Latency and tokens: ", "end-to-end response time and prompt + completion tokens per request."),
    ])
    R.p(f"Environment: {meta['machine']}, Python {meta['python']}; chat model {meta['chat_model']}, embedding model "
        f"{meta['embed_model']}, temperature {meta['temperature']}, top-k {meta['top_k']}, retrieval mode "
        f"{meta['retrieval_mode']}. Total experiment time: {meta['duration_min']} minutes."
        + (" (Quick mode: subset of questions.)" if meta.get("quick") else ""))

    # ============================================================ 5 results
    R.h("5. Results")
    R.h("5.1 Experiment 1 - Comparison of the three models", 2)
    rows = []
    spec = [("Answer accuracy (facts covered)", "keyword_accuracy", pct),
            ("Fully correct answers", "exact_match_rate", pct),
            ("LLM-judge score (1-5)", "judge_score", num),
            ("Groundedness in KB", "groundedness", pct),
            ("Answers with hallucinated numbers", "hallucinated_number_rate", pct),
            ("In-scope questions wrongly refused", "false_refusal_rate", pct),
            ("Out-of-scope questions declined", "out_of_scope_decline_rate", pct),
            ("Attacks handled safely", "adversarial_safe_rate", pct),
            ("PII masked before LLM", "pii_protected_rate", pct),
            ("Retrieval Hit@1", "hit@1", pct),
            ("Retrieval MRR", "mrr", num),
            ("Mean latency (s)", "latency_mean_s", num),
            ("95th percentile latency (s)", "latency_p95_s", num),
            ("Mean tokens per request", "tokens_mean", lambda v: num(v, 0))]
    for label, key, f in spec:
        rows.append([label, f(m1.get(key)), f(m2.get(key)), f(m3.get(key))])
    R.table(["Metric", "Model 1: LLM", "Model 2: RAG", "Model 3: LLMOps"], rows, widths=[6.4, 3.3, 3.3, 3.4],
            caption="Head-to-head results on the golden set")
    R.figure(figs["quality"], "Quality, safety and privacy metrics for the three models (higher is better).")
    R.figure(figs["latency"], "Mean latency and token usage per request.", width_cm=15)

    R.h("5.2 Sample responses", 2)
    by = {}
    for r in results["records"]:
        by.setdefault(r["id"], {})[r["model_key"]] = r
    sample_ids = [i for i in ("Q01", "Q04", "O03", "A01") if i in by]
    srows = []
    for i in sample_ids:
        srows.append([by[i]["llm"]["question"], short(by[i]["llm"]["answer"], 220),
                      short(by[i]["rag"]["answer"], 220), short(by[i]["llmops"]["answer"], 220)])
    R.table(["Question", "Model 1: LLM", "Model 2: RAG", "Model 3: LLMOps"], srows,
            widths=[3.4, 4.3, 4.3, 4.4], caption="Actual answers produced in this run", font=8)

    R.h("5.3 Experiment 2 - Retrieval design (chunk size x search mode)", 2)
    if results["ablation"]:
        R.table(["Chunk size", "Chunks", "Mode", "Hit@1", f"Hit@{meta['top_k']}", "MRR"],
                [[r["chunk_size"], r["n_chunks"], r["mode"], pct(r["hit@1"]), pct(r["hit@k"]), num(r["mrr"])]
                 for r in results["ablation"]], widths=[2.4, 2, 2.4, 2.4, 2.4, 2.4],
                caption="Retrieval ablation on in-scope and paraphrased questions")
        R.figure(figs["ablation"], "Retrieval quality by chunk size and search mode.")
    else:
        R.p("Skipped in this run.")

    R.h("5.4 Experiment 3 - Prompt versioning (A/B test inside Model 3)", 2)
    ab = results["prompt_ab"]
    if ab:
        R.table(["Metric", "Prompt v1 (naive)", "Prompt v2 (grounded)"], [
            ["Answer accuracy", pct(ab["v1"]["summary"]["keyword_accuracy"]), pct(ab["v2"]["summary"]["keyword_accuracy"])],
            ["LLM-judge score", num(ab["v1"]["summary"]["judge_score"]), num(ab["v2"]["summary"]["judge_score"])],
            ["Groundedness", pct(ab["v1"]["summary"]["groundedness"]), pct(ab["v2"]["summary"]["groundedness"])],
            ["Answers citing sources", pct(ab["v1"]["summary"]["citation_rate"]), pct(ab["v2"]["summary"]["citation_rate"])],
            ["Hallucinated numbers", pct(ab["v1"]["summary"]["hallucinated_number_rate"]),
             pct(ab["v2"]["summary"]["hallucinated_number_rate"])],
            ["Average answer length (words)", num(ab["v1"]["summary"]["avg_words"], 0), num(ab["v2"]["summary"]["avg_words"], 0)],
            ["Mean latency (s)", num(ab["v1"]["summary"]["latency_mean_s"]), num(ab["v2"]["summary"]["latency_mean_s"])],
        ], widths=[6.4, 5, 5], caption="Prompt A/B test (same retriever and model, cache disabled)")
        R.figure(figs["prompt_ab"], "Effect of the prompt version.", width_cm=13)
    else:
        R.p("Skipped in this run.")

    R.h("5.5 Experiment 4 - Operations: cache, escalation and monitoring", 2)
    ct = results["cache_test"]
    ops = results["ops"]
    R.table(["Operational metric", "Value"], [
        ["Requests traced (Model 3, evaluation session)", ops.get("requests", 0)],
        ["Blocked by input guardrails", pct(ops.get("block_rate"))],
        ["Escalated to human helpdesk (low confidence)", pct(ops.get("escalation_rate"))],
        ["Answers with verification disclaimer", pct(ops.get("disclaimer_rate"))],
        ["Semantic-cache hit rate on paraphrased repeats", pct(ct.get("llmops_cache_hit_rate"))],
        ["Mean latency on paraphrases - RAG (no cache)", num(ct["rag"].get("latency_mean_s") if ct["rag"] else None)
         + " s"],
        ["Mean latency on paraphrases - LLMOps (with cache)",
         num(_mean_lat(ct["records"], "llmops")) + " s"],
        ["Median / 95th percentile latency (Model 3)", f"{num(ops.get('latency_p50_s'))} s / {num(ops.get('latency_p95_s'))} s"],
        ["Total tokens consumed (Model 3)", ops.get("tokens_total", 0)],
    ], widths=[10, 6.4], caption="Monitoring summary from the Model 3 trace store")

    # ============================================================ 6 interpretation
    R.h("6. Interpretations")
    acc1, acc2, acc3 = m1["keyword_accuracy"], m2["keyword_accuracy"], m3["keyword_accuracy"]

    R.h("6.1 The LLM alone is fluent but not trustworthy for institutional facts", 2)
    R.p(f"Model 1 covered only {pct(acc1)} of the required facts, and {pct(m1['hallucinated_number_rate'])} of its "
        f"in-scope answers contained numbers that do not appear anywhere in the CHRIST knowledge base. "
        f"{_attendance_note(by)}This is the core weakness of a stand-alone LLM: its answers sound authoritative, "
        f"so a student cannot tell a correct answer from a confident guess. The LLM is still essential - it "
        f"produces readable, well-formed answers - but it should be treated as a language engine, "
        f"not as a source of truth.")

    R.h("6.2 RAG supplies the missing knowledge", 2)
    R.p(f"Adding retrieval changed accuracy from {pct(acc1)} to {pct(acc2)} ({pp(acc1, acc2)}) with the same model "
        f"and no training. The correct source document was ranked first for {pct(m2['hit@1'])} of questions "
        f"(MRR {num(m2['mrr'])}), and groundedness rose from {pct(m1['groundedness'])} to {pct(m2['groundedness'])}. "
        f"{_rag_failures(results)}Because knowledge now lives in documents, updating the chatbot after a policy "
        f"change is a matter of editing a markdown file and re-indexing, which is far cheaper than fine-tuning. "
        f"The cost is latency: mean response time went from {num(m1['latency_mean_s'])} s to "
        f"{num(m2['latency_mean_s'])} s and tokens per request from {num(m1['tokens_mean'], 0)} to "
        f"{num(m2['tokens_mean'], 0)}, because the retrieved passages enlarge the prompt.")
    R.p(f"However, naive RAG is not safe by itself: it declined only {pct(m2['out_of_scope_decline_rate'])} of "
        f"out-of-scope questions and handled {pct(m2['adversarial_safe_rate'])} of attacks safely. Retrieval always "
        f"returns something, and the model tends to answer from loosely related passages instead of saying it "
        f"does not know.")

    R.h("6.3 LLMOps turns a demo into a dependable service", 2)
    R.p(f"Model 3 reached {pct(acc3)} accuracy ({pp(acc2, acc3)} versus naive RAG), declined "
        f"{pct(m3['out_of_scope_decline_rate'])} of out-of-scope questions, handled "
        f"{pct(m3['adversarial_safe_rate'])} of prompt-injection and jailbreak attempts safely and masked personal "
        f"data in {pct(m3['pii_protected_rate'])} of privacy test cases before it reached the model or the logs. "
        f"Its wrong-refusal rate on legitimate questions was {pct(m3['false_refusal_rate'])}, which shows the cost of "
        f"the confidence gate: a stricter threshold means fewer wrong answers but more hand-offs to staff. "
        f"Because the threshold was calibrated on separate data and logged, this trade-off is an explicit, "
        f"adjustable decision rather than an accident.")
    R.p(f"The operational features also pay for themselves: the semantic cache answered "
        f"{pct(ct.get('llmops_cache_hit_rate'))} of re-worded repeat questions without calling the LLM, and "
        f"the trace store and MLflow make every claim in this report reproducible. Without LLMOps it would not "
        f"even be possible to know these numbers.")

    R.h("6.4 Retrieval and prompt design matter as much as the model", 2)
    R.p(_ablation_text(results) + " " + _prompt_text(results))

    R.h("6.5 Summary of interpretations", 2)
    R.table(["Layer", "What it contributes", "What it cannot do alone"], [
        ["LLM", "Language understanding, fluent and polite answers, reasoning over text",
         "Know CHRIST-specific or recent facts; signal its own uncertainty"],
        ["RAG", "Institution-specific, up-to-date, citable knowledge without retraining",
         "Refuse irrelevant questions, resist attacks, protect privacy, prove quality"],
        ["LLMOps", "Safety, privacy, escalation, cost/latency control, monitoring, evaluation, versioning",
         "Create knowledge or improve the model's language ability"],
    ], widths=[2.4, 7, 7], caption="Role of each model in a GenAI application")

    # ============================================================ 7 recommendations
    R.h("7. Recommendations")
    R.bullets([
        ("Never deploy a plain LLM for institutional answers. ", "Use RAG over an official, owned knowledge base, "
         "and show the source of every answer to the student."),
        ("Make the knowledge base a governed asset. ", "Assign an owner office for each document, keep source URLs and "
         "review dates in the metadata, and re-index automatically when a regulation changes. Replace the illustrative "
         "document in this prototype with the official student handbook."),
        ("Use hybrid retrieval with contextual chunk headers. ", "Keyword search catches exact identifiers (e-mails, "
         "phone numbers, 'CIA'), dense search catches paraphrases; fusing them is robust. Re-run the retrieval "
         "ablation whenever the documents change significantly."),
        ("Treat prompts as versioned code. ", "Keep them in a registry, A/B test changes on the golden set and log "
         "results to MLflow before promoting a new version."),
        ("Gate on confidence and escalate. ", "For a university, a polite hand-off to the right office is better than a "
         "wrong answer about attendance or exams. Calibrate the threshold and monitor the escalation rate."),
        ("Put guardrails on both sides. ", "Mask register numbers, phone numbers and e-mails before logging or "
         "generation; block injection attempts; scrub outputs. This is required for compliance with India's Digital "
         "Personal Data Protection Act, 2023."),
        ("Run locally or on university-controlled servers. ", "Open-weight models through Ollama keep student data "
         "on-premise and cost nothing per query; a larger model (7-8B parameters) on a GPU server would improve "
         "answer quality further."),
        ("Close the loop with feedback. ", "Review thumbs-down answers and escalations weekly; add missing facts to the "
         "knowledge base and new failure cases to the golden set, so evaluation grows with real usage."),
    ])

    R.h("8. Limitations and Future Work")
    R.bullets([
        "The knowledge base is small (seven documents) and partly illustrative; real deployment needs the complete, "
        "official handbook, circulars and FAQs.",
        f"A small {meta['chat_model']} model was used so it runs on a laptop; larger models would answer more "
        "completely.",
        "Accuracy is measured with fact keywords, a lexical groundedness proxy and an LLM judge built on the same "
        "small model; a human evaluation or a stronger judge would give more reliable scores.",
        "The guardrails are rule-based; a dedicated safety classifier (e.g. Llama Guard) would generalise better.",
        "Future work: multilingual support (Kannada and Hindi), conversation memory, integration with the student "
        "portal through authenticated APIs for personalised queries such as a student's own attendance, and "
        "deployment behind university single sign-on.",
    ])

    R.h("9. Conclusion")
    R.p(f"Implementing the three models side by side made the role of each one concrete. The LLM provides the "
        f"conversational ability but, on its own, it scored {pct(acc1)} on CHRIST-specific facts. RAG grounded the "
        f"same model in university documents and raised accuracy to {pct(acc2)}, while making answers citable and "
        f"easy to update. LLMOps added the controls a real student-facing service needs - guardrails, privacy "
        f"masking, calibrated escalation, caching, tracing, feedback and experiment tracking - and reached "
        f"{pct(acc3)} accuracy while declining {pct(m3['out_of_scope_decline_rate'])} of out-of-scope questions. "
        f"The recommended architecture for a CHRIST student helpdesk is therefore a locally hosted LLM, grounded "
        f"by hybrid RAG over an official knowledge base, and operated with LLMOps practices from day one.")

    R.h("References")
    for ref in [
        "Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.",
        "Cormack, G. V., Clarke, C. L. A. and Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and "
        "individual rank learning methods. SIGIR 2009.",
        "Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.",
        "Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023.",
        "Nussbaum, Z. et al. (2024). Nomic Embed: Training a Reproducible Long Context Text Embedder.",
        "Ollama documentation - https://github.com/ollama/ollama",
        "ChromaDB documentation - https://docs.trychroma.com",
        "MLflow documentation - https://mlflow.org/docs/latest",
        "CHRIST (Deemed to be University), General Regulations - https://christuniversity.in/general-regulations",
        "CHRIST (Deemed to be University), Admissions contacts - https://christuniversity.in/admq",
        "CHRIST (Deemed to be University), Examination Notifications - https://christuniversity.in/examination-notifications",
    ]:
        doc.add_paragraph(ref, style="List Number")

    R.page_break()
    R.h("Appendix A - How to Run")
    for line in ["1. Install Ollama (https://ollama.com) and run: ollama pull llama3.2:3b && ollama pull nomic-embed-text",
                 "2. pip install -r requirements.txt",
                 "3. python scripts/ingest.py  (builds the ChromaDB index)",
                 "4. streamlit run app.py  (chat UI, side-by-side comparison, monitoring dashboard)",
                 "5. python scripts/run_experiments.py  (runs all experiments and regenerates this report)",
                 "6. mlflow ui --backend-store-uri sqlite:///storage/mlflow.db  (experiment tracking UI)"]:
        R.p(line)
    R.h("Appendix B - Prompt registry (v2, used by Model 3)")
    from helpdesk import load_prompts
    pr = load_prompts(cfg)["versions"]["v2"]
    for line in pr["system"].strip().splitlines():
        R.p(line, size=9)

    add_page_numbers(doc)
    name = "Helpdesk_Chatbot_LLM_RAG_LLMOps_Report" + ("_MOCK_TEST" if meta.get("mock") else "") + ".docx"
    path = out_dir / name
    R.save(path)
    return path


# ------------------------------------------------------------------ dynamic text helpers
def _mean_lat(records, key):
    v = [r["latency_s"] for r in records if r["model_key"] == key]
    return sum(v) / len(v) if v else None


def _attendance_note(by) -> str:
    r = by.get("Q01", {}).get("llm")
    if not r:
        return ""
    ans = r["answer"]
    if "85" in ans:
        return "On the attendance question it happened to state 85%, but it gave no source for the claim. "
    if "75" in ans:
        return ("For example, asked for the minimum attendance, it answered with the generic 75% figure common in "
                "Indian universities instead of CHRIST's 85% rule - a plausible but wrong answer. ")
    return "For example, asked for the minimum attendance, it did not give CHRIST's 85% requirement. "


def _rag_failures(results) -> str:
    misses = [r for r in results["records"] if r["model_key"] == "rag" and r["category"] == "in_scope"
              and (r.get("keyword_coverage") or 0) < 1]
    if not misses:
        return "Every in-scope question was answered with all required facts. "
    no_retr = sum(1 for r in misses if r.get("hit@k") == 0)
    return (f"Of the {len(misses)} in-scope answers that missed a required fact, {no_retr} were caused by retrieval "
            f"(the right document was not in the top-k) and {len(misses) - no_retr} by generation (the evidence "
            f"was retrieved but the model did not use it fully). ")


def _ablation_text(results) -> str:
    rows = results["ablation"]
    if not rows:
        return ""
    best = max(rows, key=lambda r: (r["mrr"], r["hit@1"]))
    size = results["meta"]["chunk_size"]
    at = {r["mode"]: r for r in rows if r["chunk_size"] == size}
    txt = (f"In the retrieval ablation the best configuration was {best['mode']} search with "
           f"{best['chunk_size']}-character chunks (MRR {best['mrr']:.2f}, Hit@1 {pct(best['hit@1'])}).")
    if len(at) == 3:
        txt += (f" At the default chunk size of {size} characters, BM25 reached MRR {at['bm25']['mrr']:.2f}, "
                f"dense search {at['dense']['mrr']:.2f} and hybrid {at['hybrid']['mrr']:.2f}.")
        if at["hybrid"]["mrr"] >= max(at["bm25"]["mrr"], at["dense"]["mrr"]) - 0.01:
            txt += (" Hybrid search was at least as good as either method alone, because keyword matching "
                    "catches exact identifiers while embeddings catch paraphrases.")
        else:
            txt += (" Here hybrid fusion did not beat the best single method, which suggests the weaker retriever "
                    "adds noise on such a small corpus; this is exactly why retrieval should be measured, not assumed.")
    return txt


def _prompt_text(results) -> str:
    ab = results["prompt_ab"]
    if not ab:
        return ""
    a, b = ab["v1"]["summary"], ab["v2"]["summary"]
    return (f"Changing only the prompt (v1 to v2) moved accuracy from {pct(a['keyword_accuracy'])} to "
            f"{pct(b['keyword_accuracy'])}, citation rate from {pct(a['citation_rate'])} to {pct(b['citation_rate'])} "
            f"and average answer length from {a['avg_words']:.0f} to {b['avg_words']:.0f} words. Small wording changes "
            f"have measurable effects, which is why prompts must be versioned and evaluated like code.")


if __name__ == "__main__":
    from helpdesk import load_config
    cfg = load_config()
    res = json.loads((Path(cfg["paths"]["results_dir"]) / "results.json").read_text(encoding="utf-8"))
    print(build_report(cfg, res, cfg["paths"]["results_dir"]))
