## 📚 Hogwarts Telegram QA Bot

A Telegram bot that uses FAISS and SentenceTransformer for semantic search, and Cohere for context-aware question answering.

### ⚙️ Features

* Hierarchical section-based navigation via inline menus.
* Local embedding generation (`all-MiniLM-L6-v2`) and FAISS vector similarity search over preprocessed text chunks.
* Cohere API integration for retrieving contextual answers using top-k relevant chunks.
* Markdown-based formatting for responses and additional information display.

### 📁 Files & Structure

* `data.csv` – Hierarchical content source (`Section`, `Sub-section`, `Content`, `Additional`).
* `chunks.json` – Preprocessed semantic chunks of the content.
* `embs.npy` – FAISS-ready embeddings of the chunks.
* `bot.py` – Main bot logic, handlers, FAISS index, and Cohere integration.

### 🧠 Requirements

* Python 3.8+
* Telegram Bot Token
* Cohere API Key

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🚀 Run the Bot

```bash
python bot.py
```

