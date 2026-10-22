# 🧠 RAG Knowledge Assistant

### AI-Powered PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application developed using **Python, Streamlit, TF-IDF, and GroqCloud**.

Users can upload a PDF document, ask questions in natural language, and receive AI-generated answers grounded in relevant document passages, along with source page references.

## 🚀 Project Overview

The RAG Knowledge Assistant demonstrates how document retrieval can be combined with a Large Language Model (LLM) to provide context-aware answers.

Instead of sending an entire document to an AI model, the application extracts the document's text, divides it into smaller chunks, identifies relevant sections using TF-IDF and cosine similarity, and sends the retrieved context to GroqCloud for answer generation.

The application is designed as an AI Developer portfolio project.

**GitHub Repository:** https://github.com/gajan-spec/RAG-Knowledge-Assistant

**Project Status:** Working prototype — local execution

**AI Model:** Groq-hosted GPT-OSS 20B


## ✨ Key Features

- **PDF Upload:** Upload and process text-based PDF documents.
- **Text Extraction:** Extract document content using PyPDF.
- **Document Chunking:** Divide extracted text into overlapping chunks.
- **TF-IDF Retrieval:** Identify passages relevant to the user's question.
- **Cosine Similarity:** Rank document sections by relevance.
- **Groq AI Integration:** Generate natural-language answers using a cloud-based LLM.
- **Page References:** Display source page citations and retrieved page numbers.
- **Hallucination Reduction:** Instruct the AI to answer using retrieved document information.
- **Modern User Interface:** Dark-themed Streamlit application with PDF upload and chat functionality.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend logic and application development |
| Streamlit | Interactive web interface |
| GroqCloud API | LLM-based answer generation |
| PyPDF | PDF text extraction |
| Scikit-learn | TF-IDF vectorization and cosine similarity |
| Python-dotenv | Environment variable management |

## ⚙️ How It Works

1. The user uploads a PDF document.
2. PyPDF extracts readable text from each page.
3. The extracted content is split into overlapping chunks.
4. The user submits a question.
5. TF-IDF and cosine similarity identify relevant chunks.
6. The retrieved passages are sent to a Groq-hosted AI model.
7. The AI generates an answer based on the supplied context.
8. The application displays the answer and retrieved source pages.

### RAG Architecture

```text
PDF Upload
    |
    v
Text Extraction (PyPDF)
    |
    v
Document Chunking
    |
    v
TF-IDF Vectorization
    |
    v
Cosine Similarity Search
    |
    v
Relevant PDF Passages
    |
    v
GroqCloud LLM
    |
    v
AI Answer + Page References
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/gajan-spec/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
```

Replace `gajan-spec` with your GitHub username after creating the repository.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows CMD:

```cmd
venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root.

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your GroqCloud API key.

**Security:** Never upload your real `.env` file or API key to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the application at:

http://localhost:8501

## 🧪 Testing and Evaluation

The application was tested using a sample PDF containing information about Project Atlas.

| Test Scenario | Expected Behavior | Result |
|---|---|---|
| Ask for the project coordinator | Identify Mira Sen from Page 1 | Passed |
| Ask for an undocumented salary | State that the information is unavailable | Passed |
| Ask for a project milestone | Return 1 June 2026 from Page 3 | Passed |
| Retrieve information from multiple pages | Find relevant document sections | Passed |

These tests demonstrate successful document retrieval and grounded response generation on the sample document. They do not guarantee accuracy for every possible PDF or question.

## 🔐 Security

- API keys are stored in environment variables.
- `.env` and virtual environment folders are excluded from Git.
- The Groq API key is not hardcoded into the application.
- Uploaded document passages are sent to GroqCloud when generating AI answers.

Use non-sensitive documents unless you have reviewed the applicable provider privacy and data-handling terms.

## ⚠️ Current Limitations

- Supports one PDF at a time.
- Works best with PDFs containing selectable text.
- Scanned or image-only PDFs may require OCR.
- TF-IDF retrieval is keyword-based and may miss semantically related passages.
- AI-generated answers and citations should be verified against the original PDF.
- Requires an internet connection for GroqCloud inference.
- GroqCloud Free Plan usage limits apply.
- Currently designed for local execution; public hosting is not yet configured.

## 🔮 Possible Future Improvements

- Semantic retrieval using embeddings and a vector database.
- Support for multiple PDF documents.
- Improved citation validation.
- Enhanced error handling and document privacy controls.
- Public deployment with appropriate API security and rate limiting.

## 👨‍💻 Developer

**G.T.Gajanesh**

Aspiring AI Developer

Pearson BTEC HND in Digital Technologies — Artificial Intelligence

GitHub: https://github.com/gajan-spec

---

*Developed as part of an AI Developer portfolio to demonstrate Python development, information retrieval, and LLM integration.*