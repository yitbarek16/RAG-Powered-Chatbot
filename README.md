#  CrediTrust Complaint Explorer

An end-to-end Retrieval-Augmented Generation (RAG) system for analyzing and answering questions on customer complaints filed with CrediTrust. This project covers the full pipeline—from exploratory data analysis to an interactive question-answering interface—leveraging modern NLP tools.

---


## Exploratory Data Analysis (EDA)

- Cleaned and parsed key fields (e.g., complaint text, issue types, products).
- Visualized Distribution of Compliant Narrative Lengths

---

## Chunking & Embedding

- Sampled 5% of the dataset for prototyping.
- Split long complaint texts into overlapping chunks.
- Embedded chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Stored embeddings in a FAISS index for fast semantic search.
- Saved metadata (e.g., product, complaint ID) for traceability.

---

## Retrieval-Augmented Generation (RAG)

- Built a pipeline to:
  - Accept user questions.
  - Retrieve top-k relevant complaint chunks via vector similarity.
  - Format context and query an LLM for answers.

**Model Used:** `google/flan-t5-base` via HuggingFace's `text2text-generation` pipeline.

**Output Includes:**
- A concise, generated answer.
- Top source chunks used (for transparency and verification).

**Evaluation Topics:**
- Credit card issues, fraud, fees, refunds, and more.

---

## Gradio Interface

- Developed an interactive web app using Gradio.

**Features:**
- Input box for user questions.
- Displays AI-generated answers.
- Shows top retrieved source chunks.
- "Clear" button to reset the interface.

Ideal for non-technical stakeholders to explore complaint trends.

---

## How to Run

 **Install Requirements**
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
 **Run app.py**
   ```bash
   python app.py
   ```
## Sample Questions Tested

- What are common issues with credit card billing?
- Do customers complain about fraud on savings accounts?
- How often do people report missing refunds?
- Are late fees mentioned in complaints?
- What complaints are made about personal loans?

---

## Future Enhancements

- Add support for streaming LLM responses (token-by-token display).
- Use more of the dataset (full or larger percentage) for improved coverage.
- Utilize GPU for faster inference and embedding generation.
- Experiment with multilingual complaints or cross-lingual retrieval.
- Deploy as a web service for internal customer support analysts.



