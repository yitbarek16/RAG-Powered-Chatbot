import os
import gradio as gr
import pandas as pd
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# =====================
# Load Models and Data
# =====================

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load LLM model (text2text generation)
llm_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    max_new_tokens=256
)

# Load FAISS index and metadata
VECTOR_DIR = "vector_store"
index = faiss.read_index(os.path.join(VECTOR_DIR, "complaint_index.faiss"))
metadata = pd.read_csv(os.path.join(VECTOR_DIR, "metadata.csv"))

# Ensure metadata has required columns
if "chunk" not in metadata.columns:
    raise KeyError("Metadata CSV must contain a 'chunk' column")

# =====================
# Core Functions
# =====================

# Format prompt for the LLM
def format_prompt(context: str, question: str) -> str:
    return f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:
""".strip()

# Retrieve top-K most relevant chunks
def retrieve_chunks(question, k=5):
    query_vec = embed_model.encode([question])
    D, I = index.search(np.array(query_vec).astype("float32"), k)

    retrieved = []
    for i in I[0]:
        row = metadata.iloc[i]
        retrieved.append({
            "chunk": row["chunk"],
            "complaint_id": row.get("complaint_id", "N/A"),
            "product": row.get("product", "N/A")
        })
    return retrieved

# Generate answer from retrieved chunks
def generate_answer(question):
    chunks = retrieve_chunks(question)
    context = "\n---\n".join([chunk["chunk"] for chunk in chunks])
    prompt = format_prompt(context, question)
    response = llm_model(prompt)[0]["generated_text"]

    # Show top-2 sources
    top_sources = "\n\n".join([f"• {c['chunk'][:300]}..." for c in chunks[:2]])
    return response.strip(), top_sources

# RAG interface logic
def rag_interface(question):
    if not question.strip():
        return "Please enter a question.", ""
    answer, sources = generate_answer(question)
    return answer, sources

# =====================
# Gradio UI
# =====================

with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("## CrediTrust Customer Complaint Explorer")
    gr.Markdown("Ask a question about customer complaints and get AI-generated answers with real complaint excerpts.")

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question", placeholder="e.g. What are common issues with credit cards?", lines=1)
        submit_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")

    answer_output = gr.Textbox(label="AI-Generated Answer", lines=3)
    sources_output = gr.Textbox(label="Source Chunks (from complaints)", lines=6)

    submit_btn.click(rag_interface, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(lambda: ("", ""), inputs=None, outputs=[answer_output, sources_output])

demo.launch()
