# **RAG-based QA System with Pinecone and Google Generative AI**

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** system for answering questions based on uploaded documents. The system combines document retrieval using **Pinecone** with a generative model (Google Generative AI) to provide contextually relevant answers. Additionally, a **Streamlit** interface is provided for user interaction, where users can upload documents, ask questions, and receive responses.

## **Table of Contents**

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Technologies Used](#technologies-used)
6. [Contributing](#contributing)
7. [License](#license)

## **Features**

- **Document Upload:** Users can upload PDFs, and the system processes the document into vector embeddings.
- **Semantic Search:** Queries are embedded and matched against the document embeddings stored in Pinecone to retrieve relevant sections.
- **Generative Answers:** A generative model synthesizes coherent answers based on the retrieved document sections.
- **Real-Time Interface:** The system uses **Streamlit** for a user-friendly web interface, allowing users to interact with the QA bot.
- **Efficient Query Processing:** Handles large documents efficiently with fast retrieval times using Pinecone's vector database.
  
## **Project Structure**


## **Installation**

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system
conda create -m venv python=3.12 -y
conda activate venv/  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
export PINECONE_API_KEY=your_pinecone_api_key
export GOOGLE_API_KEY=your_google_api_key
streamlit run app.py
```
