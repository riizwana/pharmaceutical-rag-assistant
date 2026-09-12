# -*- coding: utf-8 -*-
!pip install -q gradio

!pip install -q pymupdf

!pip install -q pytesseract

!pip install -q pillow

!apt-get install -y tesseract-ocr --quiet

!pip install -q llama-index-retrievers-bm25 sentence-transformers

!pip install -q llama-index-llms-groq

!pip install -q llama-index-embeddings-huggingface

import os
from google.colab import userdata

GROQ_API_KEY = userdata.get('Groq')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from llama_index.llms.groq import Groq
llm = Groq(model="openai/gpt-oss-20b", api_key=GROQ_API_KEY)

print("Groq ready!")

import fitz
import pytesseract
from PIL import Image
import numpy as np
import io

def extract_text_with_ocr_fallback(pdf_path):
  doc = fitz.open(pdf_path)
  pages_data = []

  for i, page in enumerate(doc):
    text = page.get_text()

    if text.strip():
      pages_data.append({"page": i + 1, "text": text, "method": "digital"})
    else:
      pix = page.get_pixmap(dpi=200)
      img_bytes = pix.tobytes("png")
      img = Image.open(io.BytesIO(img_bytes))
      ocr_text = pytesseract.image_to_string(img)
      pages_data.append({"page": i + 1, "text": ocr_text, "method": "ocr"})

  doc.close()
  return pages_data

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

def classify_page_type(page_text):
  text_lower = page_text[:300].lower()
  if "certificate of quality" in text_lower or "quality certificate" in text_lower:
    return "Certificate of Quality"
  elif "packaging" in text_lower and "specification" in text_lower:
    return "Packaging Specification"
  elif "bse" in text_lower or "tse" in text_lower or "spongiform" in text_lower:
    return "BSE/TSE Declaration"
  elif "material description" in text_lower:
    return "Material Description"
  elif "supplier qualification" in text_lower or "vendor qualification" in text_lower:
    return "Supplier Qualification"
  elif "chain of custody" in text_lower:
    return "Chain of Custody"
  elif "from:" in text_lower and "to:" in text_lower:
    return "Cover Letter"
  elif "sterilization" in text_lower or "certificate of processing" in text_lower:
    return "Processing Certificate"
  else:
    return "Unclassified"

def process_document(pdf_path, source_filename):
  pages_data = extract_text_with_ocr_fallback(pdf_path)

  all_documents = []
  splitter = SentenceSplitter(chunk_size=400, chunk_overlap=80)

  for page_info in pages_data:
    page_num = page_info["page"]
    page_text = page_info["text"]

    if not page_text.strip():
      continue

    doc_type = classify_page_type(page_text)

    temp_doc = Document(text=page_text)
    chunks = splitter.get_nodes_from_documents([temp_doc])

    for chunk_idx, chunk in enumerate(chunks):
      all_documents.append(
          Document(
              text=chunk.text,
              metadata={
                  "source_file": source_filename,
                  "doc_type": doc_type,
                  "page": page_num,
                  "chunk_index": chunk_idx,
                  "extraction_method": page_info["method"]
              }
          )
      )

  return all_documents

print("Document processing pipeline ready!")

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.llm = llm

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3
)

class HybridRetriever(BaseRetriever):
  def __init__(self, vector_retriever, keyword_retriever, top_k=5):
    self.vector_retriever = vector_retriever
    self.keyword_retriever = keyword_retriever
    self.top_k = top_k
    super().__init__()

  def _retrieve(self, query_bundle, **kwargs):
    vector_nodes = self.vector_retriever.retrieve(query_bundle)
    keyword_nodes = self.keyword_retriever.retrieve(query_bundle)

    all_nodes = list(vector_nodes) + list(keyword_nodes)
    unique_nodes = {}
    for node in all_nodes:
      if node.node_id not in unique_nodes:
        unique_nodes[node.node_id] = node

    sorted_nodes = sorted(
        unique_nodes.values(),
        key=lambda x: x.score if hasattr(x, 'score') else 0.0,
        reverse=True
    )

    return sorted_nodes[:self.top_k]


def build_index_and_retriever(documents):
  index = VectorStoreIndex.from_documents(documents)
  nodes = list(index.docstore.docs.values())

  vector_retriever = index.as_retriever(similarity_top_k=5)
  bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

  hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever, top_k=5)

  return index, hybrid_retriever


print("Retrieval pipeline ready.")

!pip install -q networkx

import networkx as nx
import re

def extract_entities(text):
  entities = []
  lot_matches = re.findall(r'\b(?:Lot Number|Lot)[\s:]*(\d{6,})\b', text, re.IGNORECASE)
  for lot in lot_matches:
    entities.append(("lot_number", lot))
  part_matches = re.findall(r'\b\d{2}[\s-]?\d{4}[\s-]?\d{2}\b', text)
  for part in part_matches:
    entities.append(("part_number", part.strip()))
  return entities

def build_entity_graph(documents):
  G = nx.Graph()
  for doc in documents:
    doc_node = f"{doc.metadata['source_file']}_p{doc.metadata['page']}"
    G.add_node(doc_node, type="document", doc_type=doc.metadata['doc_type'])
    for entity_type, value in extract_entities(doc.text):
      entity_node = f"{entity_type}:{value}"
      G.add_node(entity_node, type=entity_type)
      G.add_edge(doc_node, entity_node)
  return G

def find_related_documents(entity_value, graph):
  matches = [n for n in graph.nodes if entity_value in n]
  related_docs = set()
  for m in matches:
    related_docs.update(graph.neighbors(m))
  return list(related_docs)

print("Entity graph building pipeline ready.")

!pip install -q python-dateutil

from dateutil import parser as dateparser
from datetime import datetime

COMPLIANCE_KEYWORDS = [
    "expiration date", "valid until", "assessment valid until",
    "next scheduled audit", "last on-site audit"
]

def find_compliance_dates(text, source_file, page):
  findings = []
  lines = [l for l in text.split("\n") if l.strip()]
  for i, line in enumerate(lines):
    line_lower = line.lower()
    for kw in COMPLIANCE_KEYWORDS:
      if kw in line_lower:
        candidate = line.split(":", 1)[1].strip() if ":" in line else ""
        if not candidate and i + 1 < len(lines):
          candidate = lines[i + 1].strip()

        try:
          parsed = dateparser.parse(candidate, fuzzy=True, default=datetime(2000, 1, 1))
          if parsed.year > 2000:
            days_until = (parsed.date() - datetime.now().date()).days
            findings.append({
                "source_file": source_file,
                "page": page,
                "label": kw.title(),
                "date": parsed.date().isoformat(),
                "days_until": days_until
            })
        except Exception:
          pass
  return findings

def check_compliance_dates(documents, warning_days=90):
  all_findings = []
  for doc in documents:
    all_findings.extend(find_compliance_dates(doc.text, doc.metadata["source_file"], doc.metadata["page"]))


    for f in all_findings:
      if f["days_until"] < 0:
        f["status"] = "🔴 EXPIRED"
      elif f[f"days_until"] <= warning_days:
        f["status"] = "🟡 EXPIRING SOON"
      else:
        f["status"] = "🟢 COMPLIES"

  return all_findings

print("Compliance date pipeline ready.")

def extract_supplier_metrics(text, source_file):
    metrics = {}
    patterns = {
        "supplier_name": r'Supplier Name[:\s]*\n?\s*([^\n]+)',
        "qualification_status": r'Qualification Status[:\s]*\n?\s*([^\n]+)',
        "on_time_delivery": r'On-Time Delivery[:\s]*([\d.]+)%',
        "incoming_quality": r'Incoming Quality[:\s]*([\d.]+)%',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics[key] = match.group(1).strip()
    if metrics.get("supplier_name"):
        metrics["source_file"] = source_file
        return metrics
    return None

def build_supplier_scorecard(documents):
  by_file = {}
  for doc in documents:
    source = doc.metadata["source_file"]
    if source not in by_file:
      by_file[source] = ""
    by_file[source] += "\n" + doc.text

  scorecards = []
  for source, combined_text in by_file.items():
    metrics = extract_supplier_metrics(combined_text, source)
    if metrics:
      scorecards.append(metrics)

  return scorecards

print("Supplier scorecard pipeline ready!")

def check_traceability_gaps(graph):
  gaps = []
  lot_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "lot_number"]
  for lot_node in lot_nodes:
    neighbors = list(graph.neighbors(lot_node))
    doc_types = set()
    for n in neighbors:
      node_data = graph.nodes[n]
      if node_data.get("type") == "document":
        doc_types.add(node_data.get("doc_type"))
    if "Chain of Custody" not in doc_types:
      gaps.append({
          "lot_number": lot_node.replace("lot_number:", ""),
          "found_in": ", ".join(doc_types) if doc_types else "unknown",
          "missing": "Chain of Custody"
      })
  return gaps

print("Traceability gap pipeline ready!")

def generate_answer_with_sources(query, hybrid_retriever, reranker):
    retrieved_nodes = hybrid_retriever.retrieve(query)

    if not retrieved_nodes:
        return {
            "answer": "I couldn't find relevant information in the document to answer that question.",
            "sources": [],
            "confidence": "Low",
            "chunk_count": 0
        }

    reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=query)

    context_parts = []
    sources = []
    for i, node in enumerate(reranked_nodes):
        meta = node.metadata
        context_parts.append(f"[Source {i+1}] (Page {meta.get('page')}, {meta.get('doc_type')}): {node.get_text()}")
        sources.append({
            "source_file": meta.get("source_file"),
            "doc_type": meta.get("doc_type"),
            "page": meta.get("page"),
            "extraction_method": meta.get("extraction_method"),
            "relevance_score": round(node.score, 3) if node.score is not None else None
        })

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant answering questions about pharmaceutical documents. Use ONLY the context below to answer the question. If the answer isn't in the context, say so clearly.
When you use information from a source, cite it using [Source N] notation matching the sources below.

CONTEXT:
{context}

QUESTION: {query}

Answer with citations:"""

    response = llm.complete(prompt)
    answer = str(response).strip()

    top_score = sources[0]["relevance_score"] if sources and sources[0]["relevance_score"] is not None else 0

    if top_score > -3:
        confidence = "High"
    elif top_score > -8:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "chunk_count": len(reranked_nodes)
    }

print("Answer generation pipeline ready.")

from google.colab import files

uploaded = files.upload()
pdf_filename = list(uploaded.keys())[0]
pdf_path = f"/content/{pdf_filename}"

with open(pdf_path, 'wb') as f:
  f.write(uploaded[pdf_filename])

test_documents = process_document(pdf_path, pdf_filename)
print(f"Processed into {len(test_documents)} chunks.")

test_index, test_retriever = build_index_and_retriever(test_documents)
print("Index and retriever built.")

entity_graph = build_entity_graph(test_documents)
print(f"Graph: {entity_graph.number_of_nodes()} nodes, {entity_graph.number_of_edges()} edges")

# Testing

related = find_related_documents("17242818", entity_graph)
print(f"Document referencing lot 17242818: {related}")

result = generate_answer_with_sources(
    "What are the storage conditions for this product?",
    test_retriever,
    reranker
)

print("Answer:")
print(result["answer"])
print(f"\nConfidence: {result['confidence']}")
print(f"Chunk count: {result['chunk_count']}")
print("\nSources:")
for s in result["sources"]:
  print(f" - {s['source_file']}, Page {s['page']}, {s['doc_type']} (score: {s['relevance_score']}, method: {s['extraction_method']})")

def classify_page_type(page_text):
    text_lower = page_text[:300].lower()
    if "certificate of quality" in text_lower or "quality certificate" in text_lower:
        return "Certificate of Quality"
    elif "packaging" in text_lower and "specification" in text_lower:
        return "Packaging Specification"
    elif "bse" in text_lower or "tse" in text_lower or "spongiform" in text_lower:
        return "BSE/TSE Declaration"
    elif "material description" in text_lower:
        return "Material Description"
    elif "supplier qualification" in text_lower or "vendor qualification" in text_lower:
        return "Supplier Qualification"
    elif "chain of custody" in text_lower:
        return "Chain of Custody"
    elif "storage conditions" in text_lower or "from:" in text_lower and "to:" in text_lower:
        return "Cover Letter"
    elif "sterilization" in text_lower or "certificate of processing" in text_lower:
        return "Processing Certificate"
    else:
        return "Unclassified"

print("Fixed!")


def generate_answer_with_sources(query, hybrid_retriever, reranker):
  retrieved_nodes = hybrid_retriever.retrieve(query)

  if not retrieved_nodes:
        return {
            "answer": "I couldn't find relevant information in the document to answer that question.",
            "sources": [],
            "confidence": "Low",
            "chunk_count": 0
        }

  reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=query)

  context_parts = []
  sources = []
  for i, node in enumerate(reranked_nodes):
      meta = node.metadata
      context_parts.append(f"[Source {i+1}] (Page {meta.get('page')}, {meta.get('doc_type')}): {node.get_text()}")
      sources.append({
          "source_file": meta.get("source_file"),
          "doc_type": meta.get("doc_type"),
          "page": meta.get("page"),
          "extraction_method": meta.get("extraction_method"),
          "relevance_score": round(float(node.score), 3) if node.score is not None else None
      })

  context = "\n\n".join(context_parts)

  prompt = f"""You are a helpful assistant answering questions about pharmaceutical documents. Use ONLY the context below to answer the question. If the answer isn't in the context, say so clearly.
When you use information from a source, cite it using [Source N] notation matching the sources below.

CONTEXT:
{context}

QUESTION: {query}

Answer with citations:"""

  response = llm.complete(prompt)
  answer = str(response).strip()

  top_score = sources[0]["relevance_score"] if sources and sources[0]["relevance_score"] is not None else 0

  if top_score > -3:
      confidence = "High"
  elif top_score > -8:
      confidence = "Medium"
  else:
      confidence = "Low"

  return {
      "answer": answer,
      "sources": sources,
      "confidence": confidence,
      "chunk_count": len(reranked_nodes)
  }


print("Functions updated!")

result = generate_answer_with_sources(
    "What are the storage conditions for this product?",
    test_retriever,
    reranker
)

print("ANSWER:")
print(result["answer"])
print(f"\nConfidence: {result['confidence']}")
print(f"Chunk count: {result['chunk_count']}")
print("\nSOURCES:")
for s in result["sources"]:
    print(f"  - {s['source_file']}, Page {s['page']}, {s['doc_type']} (score: {s['relevance_score']}, method: {s['extraction_method']})")

import sqlite3
from datetime import datetime
import json

conn = sqlite3.connect('/content/audit_trail.db', check_same_thread=False)
conn.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    query TEXT,
    answer TEXT,
    confidence TEXT,
    chunk_count INTEGER,
    sources_json TEXT,
    model_version TEXT
  )
''')
conn.commit()

def generate_answer_with_audit(query, hybrid_retriever, reranker, model_version="llama-3.1-8b-instant"):
  result = generate_answer_with_sources(query, hybrid_retriever, reranker)

  conn.execute('''
    INSERT INTO audit_log (timestamp, query, answer, confidence, chunk_count, sources_json, model_version)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  ''', (
      datetime.now().isoformat(),
      query,
      result["answer"],
      result["confidence"],
      result["chunk_count"],
      json.dumps(result["sources"]),
      model_version
  ))
  conn.commit()

  return result

def get_audit_history(limit=20):
  cursor = conn.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT ?', (limit,))
  return cursor.fetchall()

print("Audit trail ready.")

print(repr(test_documents[0].text[:300]))

test_documents = process_document(pdf_path, pdf_filename)
print(f"Reprocessed into {len(test_documents)} chunks")

test_index, test_retriever = build_index_and_retriever(test_documents)
print("Index rebuilt!")

result = generate_answer_with_sources(
    "What are the storage conditions for this product?",
    test_retriever,
    reranker
)

print("\nANSWER:")
print(result["answer"])
print(f"\nConfidence: {result['confidence']}")
print("\nSOURCES:")
for s in result["sources"]:
    print(f"  - {s['source_file']}, Page {s['page']}, {s['doc_type']} (score: {s['relevance_score']})")

import gradio as gr
print(gr.__version__)

chatbot_ui = gr.Chatbot(label="Answer Questions", height=450)

import gradio as gr

current_retriever = None
current_index = None
current_graph = None
chat_log = []

def process_upload(pdf_files, progress=gr.Progress()):
    global current_retriever, current_index, current_graph, chat_log
    chat_log = []

    if not pdf_files:
        return "Please upload at least one PDF file.", [], [], []

    try:
        all_documents = []
        filenames = []

        for i, pdf_file in enumerate(pdf_files):
            progress((i + 0.5) / len(pdf_files), desc=f"Processing file {i+1} of {len(pdf_files)}...")
            docs = process_document(pdf_file.name, os.path.basename(pdf_file.name))
            all_documents.extend(docs)
            filenames.append(os.path.basename(pdf_file.name))

        progress(0.8, desc="Building combined search index...")
        current_index, current_retriever = build_index_and_retriever(all_documents)
        current_graph = build_entity_graph(all_documents)

        progress(0.9, desc="Checking compliance dates and traceability...")
        compliance_findings = check_compliance_dates(all_documents)
        flagged = [f for f in compliance_findings if f["status"] != "🟢 OK"]
        compliance_rows = [[f["status"], f["source_file"], f["page"], f["label"], f["date"]] for f in flagged]

        supplier_scores = build_supplier_scorecard(all_documents)
        supplier_rows = [[s.get("supplier_name", ""), s.get("qualification_status", ""), s.get("on_time_delivery", ""), s.get("incoming_quality", "")] for s in supplier_scores]

        gaps = check_traceability_gaps(current_graph)
        gap_rows = [[g["lot_number"], g["found_in"], g["missing"]] for g in gaps]

        progress(1.0, desc="Done!")
        doc_types = set(d.metadata["doc_type"] for d in all_documents)
        status = f"✅ Processed {len(pdf_files)} file(s) ({', '.join(filenames)}) into {len(all_documents)} chunks across {len(doc_types)} document type(s): {', '.join(doc_types)}. Entity graph: {current_graph.number_of_nodes()} nodes."

        return status, compliance_rows, supplier_rows, gap_rows
    except Exception as e:
        return f"❌ Error processing files: {e}", [], [], []

def lookup_entity(entity_value):
    global current_graph
    if current_graph is None:
        return "Please upload a document first before proceeding!"
    related = find_related_documents(entity_value, current_graph)
    if not related:
        return f"No documents found referencing '{entity_value}'."
    return "\n".join([f"📄 {r}" for r in related])

def chat_respond(message, history):
    global current_retriever, chat_log

    if current_retriever is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please upload a document first."}
        ]

    try:
        result = generate_answer_with_audit(message, current_retriever, reranker)

        sources_text = "\n".join([
            f"📄 {s['source_file']} — Page {s['page']} ({s['doc_type']}) [{s['extraction_method']}]"
            for s in result["sources"]
        ])

        confidence_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(result["confidence"], "⚪")

        full_response = f"{result['answer']}\n\n---\n**{confidence_emoji} Confidence: {result['confidence']}** | **Chunks used: {result['chunk_count']}**\n\n**Sources:**\n{sources_text}"

        chat_log.append({"question": message, "answer": full_response})

        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response}
        ]
    except Exception as e:
        import traceback
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"}
        ]

def save_history():
    if not chat_log:
        return None
    filename = "/content/rag_chat_history.txt"
    with open(filename, "w") as f:
        for entry in chat_log:
            f.write(f"Q: {entry['question']}\n\nA: {entry['answer']}\n\n{'='*60}\n\n")
    return filename

def show_audit():
    rows = get_audit_history()
    display_rows = [[r[0], r[1][:19], r[2][:50], r[4], r[7]] for r in rows]
    return gr.update(value=display_rows, visible=True)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Pharma RAG Assistant") as demo:
    gr.Markdown("# 🔬 Pharmaceutical Document RAG Assistant")
    gr.Markdown("Upload any pharmaceutical document (digital or scanned) and ask questions — answers include sources, confidence, citations, compliance flags, and supply-chain traceability.")

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📄 Upload PDF(s)", file_types=[".pdf"], file_count="multiple")
            upload_output = gr.Textbox(label="Processing Status", interactive=False, lines=3)

            gr.Markdown("### ⚠️ Compliance Flags")
            compliance_table = gr.Dataframe(headers=["Status", "File", "Page", "Field", "Date"], label="Expiring/Expired Items")

            gr.Markdown("### 🏭 Supplier Scorecard")
            supplier_table = gr.Dataframe(headers=["Supplier", "Status", "On-Time %", "Quality %"], label="Supplier Metrics")

            gr.Markdown("### 🔗 Traceability Gaps")
            gap_table = gr.Dataframe(headers=["Lot Number", "Found In", "Missing"], label="DSCSA-style Gaps")

            download_btn = gr.Button("💾 Export Chat History")
            download_output = gr.File(label="Downloaded File")
            audit_btn = gr.Button("📋 View Audit Trail")
            audit_output = gr.Dataframe(headers=["ID", "Timestamp", "Query", "Confidence", "Model"], visible=False)
            entity_input = gr.Textbox(label="🔗 Lookup Lot/Part Number", placeholder="e.g. 17242818")
            entity_output = gr.Textbox(label="Related Documents", interactive=False)

        with gr.Column(scale=2):
            chatbot_ui = gr.Chatbot(label="Ask Questions", height=450)
            query_box = gr.Textbox(label="Your Question", placeholder="e.g. What test methods were used for quality control?")
            clear_btn = gr.Button("🗑️ Clear Chat")

    pdf_input.upload(
        fn=process_upload,
        inputs=pdf_input,
        outputs=[upload_output, compliance_table, supplier_table, gap_table],
        show_progress="full"
    )
    query_box.submit(fn=chat_respond, inputs=[query_box, chatbot_ui], outputs=chatbot_ui, show_progress="full")
    query_box.submit(fn=lambda: "", outputs=query_box)
    clear_btn.click(lambda: None, None, chatbot_ui, queue=False)
    download_btn.click(fn=save_history, outputs=download_output)
    audit_btn.click(fn=show_audit, outputs=audit_output)
    entity_input.submit(fn=lookup_entity, inputs=entity_input, outputs=entity_output)

demo.launch(share=True)

import inspect
print(inspect.getsource(generate_answer_with_sources))

pdf_path_eval = pdf_path
eval_documents = process_document(pdf_path_eval, os.path.basename(pdf_path_eval))
print(f"Processed {len(eval_documents)} chunks")

eval_index, eval_retriever = build_index_and_retriever(eval_documents)
print("Eval index built)")

for d in eval_documents:
  print(f"Page {d.metadata['page']}: {d.metadata['doc_type']}")

from google.colab import files

uplaoded = files.upload()
pdf_filename_eval = list(uploaded.keys())[0]
pdf_path_eval = f"/content/{pdf_filename_eval}"

with open(pdf_path_eval, 'wb') as f:
  f.write(uploaded[pdf_filename_eval])

eval_documents = process_document(pdf_path_eval, pdf_filename_eval)
print(f"Processed {len(eval_documents)} chunks")

eval_index, eval_retriever = build_index_and_retriever(eval_documents)
print("Eval index built!")

for d in eval_documents:
  print(f"Page {d.metadata['page']}: {d.metadata['doc_type']}")

import inspect
print(inspect.getsource(classify_page_type))

test_queries = [
    {"query": "What are the storage conditions for this product?", "correct_page": 1, "correct_doc_type": "Cover Letter"},
    {"query": "What is the lot number for the Low Flow Kit?", "correct_page": 2, "correct_doc_type": "Certificate of Quality"},
    {"query": "What test methods were used for quality control?", "correct_page": 3, "correct_doc_type": "Certificate of Quality"},
    {"query": "What is the expiration date for the High Flow Kit?", "correct_page": 3, "correct_doc_type": "Certificate of Quality"},
    {"query": "What is the operating temperature range?", "correct_page": 1, "correct_doc_type": "Cover Letter"},
]

print(f"Test set: {len(test_queries)} queries")

def evaluate_retrieval(test_queries, retriever, k=3):
  results = []

  for tq in test_queries:
    retrieved_nodes = retriever.retrieve(tq["query"])
    retrieved_pages = [node.metadata.get("page") for node in retrieved_nodes[:k]]

    hit = tq["correct_page"] in retrieved_pages
    rank = None
    if hit:
      rank = retrieved_pages.index(tq["correct_page"]) + 1

    results.append({
            "query": tq["query"],
            "correct_page": tq["correct_page"],
            "retrieved_pages": retrieved_pages,
            "hit": hit,
            "rank": rank
        })

    recall_at_k = sum(1 for r in results if r["hit"]) / len(results)
    reciprocal_ranks = [1/r["rank"] if r["hit"] else 0 for r in results]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    precision_at_k = sum(1 for r in results if r["hit"]) / (len(results) * k)

  return results, recall_at_k, mrr, precision_at_k

eval_results, recall, mrr, precision = evaluate_retrieval(test_queries, eval_retriever, k=3)

print("=== EVALUATION RESULTS ===\n")
for r in eval_results:
  status = "✅ HIT" if r["hit"] else "❌ MISS"
  print(f"{status} | Query: {r['query']}")
  print(f"  Expected page: {r['correct_page']} | Retrieved pages: {r['retrieved_pages']} | Rank: {r['rank']}")

print(f"Recall@3: {recall:.2%}")
print(f"Precision@3: {precision:.2%}")
print(f"MRR: {mrr:.3f}")

import time

timing_results = []

for tq in test_queries[:3]:
  start_retrieve = time.time()
  nodes = eval_retriever.retrieve(tq["query"])
  retrieve_time = time.time() - start_retrieve

  start_rerank = time.time()
  reranked = reranker.postprocess_nodes(nodes, query_str=tq["query"])
  rerank_time = time.time() - start_rerank

  start_gen = time.time()
  result = generate_answer_with_sources(tq["query"], eval_retriever, reranker)
  gen_time = time.time() - start_gen

  timing_results.append({
      "query": tq["query"],
      "retrieve_ms": round(retrieve_time * 1000, 1),
      "rerank_ms": round(rerank_time * 1000, 1),
      "full_pipeline_ms": round(gen_time * 1000, 1)
  })

  print("=== TIMING RESULTS ===")
  for t in timing_results:
    print(f"Query: {t['query']}")
    print(f"  Retrieval: {t['retrieve_ms']}ms | Reranking: {t['rerank_ms']}ms | Full pipeline (including LLM): {t['full_pipeline_ms']}ms\n")

avg_retrieve = sum(t["retrieve_ms"] for t in timing_results) / len(timing_results)
avg_rerank = sum(t["rerank_ms"] for t in timing_results) / len(timing_results)
avg_full = sum(t["full_pipeline_ms"] for t in timing_results) / len(timing_results)

print(f"AVERAGE — Retrieval: {avg_retrieve:.1f}ms | Reranking: {avg_rerank:.1f}ms | Full pipeline: {avg_full:.1f}ms")