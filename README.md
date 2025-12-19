# 🧠 Intelligent Complaint Analysis for Financial Services

## 📘 Overview

Fast-growing financial companies receive **thousands of customer complaints** every month, but these complaints are often stored as long, unstructured text that is difficult to analyze at scale. As a result, teams spend hours manually reading feedback, delaying critical decisions and making it hard to detect emerging issues early.

This project builds an **AI-powered complaint analysis system** for **CrediTrust Financial** that transforms raw customer complaints into **clear, evidence-backed insights**. Using **Retrieval-Augmented Generation (RAG)**, the system allows internal teams to ask plain-English questions and instantly receive grounded answers based on real customer narratives.

The project was completed as part of the **10 Academy – Artificial Intelligence Mastery Program**.


## 🎯 Business Problem

CrediTrust Financial serves over **500,000 customers** across multiple financial products, including:

* Credit Cards
* Personal Loans
* Buy Now, Pay Later (BNPL)
* Savings Accounts
* Money Transfers

Internal teams face several challenges:

* **Product Managers** struggle to identify recurring issues quickly
* **Customer Support** is overwhelmed by complaint volume
* **Compliance & Risk teams** react too late to repeated violations
* **Executives** lack visibility into emerging customer pain points

The business needs a tool that can **surface trends in minutes instead of days** and make complaint data accessible to **non-technical users**.


## 📂 Data

The project uses real consumer complaint data from the **Consumer Financial Protection Bureau (CFPB)**. Each record includes:

* Product category
* Issue type
* Free-text customer narrative
* Submission date and metadata

The **complaint narrative** serves as the core input for semantic search and answer generation.


## 🔍 What This Project Does

### 1️⃣ Exploratory Data Analysis (EDA)

* Analyzed complaint distribution across financial products
* Identified dominant issue types such as billing disputes, fraud, and service quality
* Examined narrative length and structure to guide text chunking strategy


### 2️⃣ Chunking, Embedding & Vector Indexing

* Split long complaint narratives into overlapping text chunks
* Embedded chunks using **sentence-transformers/all-MiniLM-L6-v2**
* Indexed embeddings with **FAISS** for fast semantic retrieval
* Stored metadata to maintain traceability between answers and source complaints
* Used **5% of the dataset** for prototyping


### 3️⃣ Retrieval-Augmented Generation (RAG)

* Accepts natural-language questions from users
* Retrieves the most relevant complaint chunks via vector similarity search
* Feeds retrieved context into **google/flan-t5-base** to generate concise, grounded answers
* Returns both the **answer** and **supporting complaint excerpts** for transparency


### 4️⃣ Interactive Analyst Interface

* Built a lightweight **Gradio web app** for internal stakeholders
* Enables users to:

  * Ask questions in plain English
  * View AI-generated answers
  * Inspect source complaint evidence
* Designed for **non-technical teams** such as Product, Support, and Compliance


## 📊 Key Outcomes

* Reduced time to identify complaint trends from **days to minutes**
* Enabled self-service analytics for non-technical teams
* Improved transparency through evidence-backed answers
* Demonstrated how RAG can support **proactive issue detection** in financial services


## 🧰 Tech Stack

* **Python**
* **Hugging Face Transformers**
* **Sentence Transformers**
* **FAISS**
* **Gradio**
* **Pandas, NumPy**
* **Jupyter Notebook**

## 🛡️ License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and share this project with proper attribution.

---
Let's stay in touch! Feel free to connect with me on LinkedIn:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yitbarektesfaye)
