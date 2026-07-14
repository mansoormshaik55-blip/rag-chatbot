# 📄 RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **Pinecone**, **Hugging Face Embeddings**, and **Groq LLM**. The chatbot retrieves relevant information from an uploaded PDF and answers user questions based only on the document's content.

## 🚀 Features

* Upload and process PDF documents
* Split documents into text chunks
* Generate embeddings using Hugging Face (`all-MiniLM-L6-v2`)
* Store embeddings in Pinecone Vector Database
* Retrieve the most relevant document chunks
* Generate grounded answers using Groq LLM
* Simple and interactive Streamlit interface

## 🛠️ Technologies Used

* Python
* Streamlit
* LangChain
* Pinecone
* Hugging Face Embeddings
* Groq API
* PyPDF
* python-dotenv

## 📂 Project Structure

```
rag-chatbot/
│── app.py                 # Streamlit application
│── ingest.py              # PDF ingestion and indexing
│── rag_logic.py           # Retrieval and generation logic
│── verify_grounding.py    # Grounding verification
│── requirements.txt
│── .gitignore
│── tests/
```

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/mansoormshaik55-blip/rag-chatbot.git
cd rag-chatbot
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Create a `.env` file and add your API keys:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
GROQ_API_KEY=your_groq_api_key
```

## ▶️ Running the Project

First, index your PDF:

```bash
python ingest.py
```

Then launch the chatbot:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`) in your browser.

## 📸 Demo

You can add screenshots of the chatbot interface here to showcase the project.

## 📌 Future Improvements

* Support multiple PDF uploads
* Conversation memory
* Chat history
* Source citations for each answer
* Deploy online using Streamlit Community Cloud

## 👨‍💻 Author

**Shaik Mansoor**

GitHub: https://github.com/mansoormshaik55-blip
