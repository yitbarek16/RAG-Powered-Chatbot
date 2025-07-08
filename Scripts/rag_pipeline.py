import os
import faiss
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =====================
# Load Vector Store
# =====================
def load_vector_store(index_path='vector_store/complaint_index.faiss', metadata_path='vector_store/metadata.csv'):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_path}")
    
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)

    if 'chunk' not in metadata.columns:
        raise KeyError("Metadata CSV must contain a 'chunk' column")
    
    return index, metadata

# =====================
# Load Embedding Model
# =====================
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# =====================
# Load LLM Model (T5-style requires text2text-generation)
# =====================
def load_llm_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_new_tokens=256
    )

# =====================
# Format Prompt
# =====================
def format_prompt(context: str, question: str) -> str:
    return f"""
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:
""".strip()

# =====================
# Retrieve Top-K Chunks
# =====================
def retrieve_chunks(question: str, index, metadata_df, model, k=5):
    query_vec = model.encode([question])
    D, I = index.search(np.array(query_vec).astype("float32"), k)

    retrieved = []
    for i in I[0]:
        row = metadata_df.iloc[i]
        retrieved.append({
            "chunk": row["chunk"],
            "complaint_id": row["complaint_id"] if "complaint_id" in row else None,
            "product": row["product"] if "product" in row else None
        })
    return retrieved

# =====================
# Run Full RAG Pipeline
# =====================
def answer_question(question: str, index, metadata, embed_model, llm_model, k=5):
    retrieved = retrieve_chunks(question, index, metadata, embed_model, k=k)
    context = "\n---\n".join([r['chunk'] for r in retrieved])

    prompt = format_prompt(context, question)
    response = llm_model(prompt)[0]['generated_text']

    return {
        "question": question,
        "answer": response.strip(),
        "sources": retrieved[:2]  # Top 2 chunks shown as sources
    }

# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    print("Device set to use CPU")

    embed_model = load_embedding_model()
    llm_model = load_llm_model()
    index, metadata = load_vector_store()

    example_questions = [
        "What are common issues with credit card billing?",
        "Do customers complain about fraud on savings accounts?",
        "How often do people report missing refunds?",
        "Are late fees mentioned in complaints?",
        "What complaints are made about personal loans?"
    ]

    results = []
    for q in tqdm(example_questions):
        out = answer_question(q, index, metadata, embed_model, llm_model)
        results.append(out)
        print("\n===============================")
        print(f"Q: {out['question']}")
        print(f"A: {out['answer']}")
        print("Top Sources:")
        for src in out['sources']:
            print(" -", src['chunk'][:100], "...")

    # Sample of Q and A
    pd.DataFrame(results).to_csv("rag_eval_results.csv", index=False)
