# 🔬 Pharmaceutical RAG Assistant - Pfizer Externship 2026
### by Rizwana · Data Science & AI in Pharma

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

# 💊 Pharmaceutical RAG Assistant

> An OCR-aware Retrieval-Augmented Generation (RAG) pipeline built to answer questions over pharmaceutical supply-chain documents... with receipts. 📄🔍

Built as part of the **Pfizer Advanced Cohort Data Science Externship**, this project takes messy, scanned pharma documents and turns them into a searchable, citation-backed Q&A system.

---

## ✨ What it does

- 🧠 **Hybrid retrieval** — combines semantic + keyword search for sharper results
- 🔁 **Reranking** — pulls the *actually* relevant chunks to the top
- 📑 **Citation-scored answers** — no hallucinated confidence, every answer points back to its source
- 🖥️ **Gradio interface** — clean, interactive UI, no command-line wrangling required
- 🔎 **OCR-aware ingestion** — reads scanned/multi-format documents, not just clean text

## 🧩 Bonus features built for the Pfizer pitch

| Feature | What it catches |
|---|---|
| 📅 Compliance date flagging | Automatically flags upcoming regulatory/compliance deadlines |
| ⚠️ Supplier risk scorecard | Surfaces risk signals across suppliers |
| 🕸️ Traceability gap detection | DSCSA-style entity graph (via NetworkX) linking lot/part numbers across documents to catch traceability gaps |

## 📊 Performance

Tested on a 5-query benchmark set:

| Metric | Result |
|---|---|
| Recall@3 | 100% |
| MRR | 0.767 |
| Avg. retrieval latency | ~143ms |
| Avg. full-pipeline latency | ~1.8s |

## 🛠️ Built with

- **Language/Env:** Python, Google Colab (T4 GPU)
- **Retrieval/RAG:** LangChain-style patterns, BGE-small-en-v1.5 embeddings
- **OCR:** PaddleOCR (benchmarked vs. Tesseract & EasyOCR — PaddleOCR won at ~98% confidence)
- **LLMs:** Groq / OpenAI-compatible endpoints, local Mistral 7B
- **Graph analysis:** NetworkX
- **Data wrangling:** Pandas, python-dateutil, PyMuPDF (fitz)
- **Interface:** Gradio

## 🚀 Why this exists

Pharma supply chains generate mountains of dense, semi-structured documents. This project shows how a well-built RAG pipeline can turn that mess into fast, trustworthy, cited answers — while adding intelligence layers (compliance, risk, traceability) that go beyond a plain Q&A bot.

## 📌 Status

Built and benchmarked as a working prototype during the externship — nominated to present live to Pfizer stakeholders as part of a pitch deck. 🎤
Repo may be updated spontaneously!

---

*Part of an ongoing journey from an English Literature degree ➡️ Data Science & AI. Follow the build log on www.linkedin.com/c/rizwanasalad/.*
