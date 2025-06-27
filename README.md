# 🇬🇧 UK Sponsor Licence Cover Letter Generator

A Streamlit web app that generates UKVI-compliant sponsor licence cover letters using a Retrieval-Augmented Generation (RAG) pipeline powered by:

- 🧠 **OpenAI Embeddings + GPT**
- 📚 **LangChain-style document parsing**
- 📦 **Zilliz Cloud (Milvus) vector database**

---

## 🚀 Features

- Generates formal cover letters for UK Skilled Worker sponsor licence applications
- Retrieves context-aware examples from embedded guidance documents and templates
- Automatically includes placeholders for missing information
- Supports PDF and DOCX embedding

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/uk-legaltech-rag-model.git
cd uk-legaltech-rag-model
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # on Linux/macOS
venv\Scripts\activate      # on Windows
```

### 3. Install dependencies

```bash
pip install pymilvus --no-deps
pip install -r requirements.txt
```

> ✅ Make sure `pymilvus`, `openai`, `python-dotenv`, and `streamlit` are included.

---

## 🔐 .env File Setup

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
ZILLIZ_ADDRESS=https://xxxxx.api.gcp.zillizcloud.com
ZILLIZ_TOKEN=xxxxxxxxxxxxxxxxxxxx
COLLECTION_NAME=coverletter_guidelines
```

> You can get Zilliz Cloud URI and Token from your [Zilliz Cloud dashboard](https://cloud.zilliz.com/).

---

## 📥 Embed Documents into VectorDB (Optional)

> This script loads all documents in `./docs/`, splits them, embeds them via OpenAI, and upserts into the Milvus collection.

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🧠 Project Structure

```
uk-legaltech-rag-model/
│
├── app.py                    # Streamlit frontend
├── rag_model.py              # Core RAG logic (retriever + LLM)
├── embed_documents.py        # Document loading & embedding
├── .env                      # Secrets and config
├── requirements.txt          # Python dependencies
└── docs/                     # Folder to store PDF/DOCX guidance/templates
```

---

## 📋 Example Prompt

Enter into the text area:

> _"We are a software firm in Manchester with 30 staff. We need to sponsor a data scientist under SOC code 2136 due to ongoing project expansion."_

---

## ⚠️ Known Issues

- If collection is not loaded, run `create_collection.py` or load it manually via the Zilliz UI.
- Large text responses need escaping via `json.dumps()` when injecting into HTML/JS.

---

## ⚙️ Manual Installation (Optional)

To install `pymilvus` manually **without dependencies**:

```bash
pip install pymilvus --no-deps
```

---

## 📘 Credits

Built by **UK LegalTech** with 💼 using:
- [OpenAI](https://platform.openai.com/)
- [Zilliz Cloud](https://cloud.zilliz.com/)
- [Streamlit](https://streamlit.io/)
- [LangChain-style patterns](https://www.langchain.com/)

---

## 📄 License

MIT License — feel free to use, fork, and contribute!