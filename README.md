# Document Q&A Bot

A terminal-based AI assistant that reads any PDF and answers questions strictly from its contents. Built with Groq's inference API and LLaMA 3.3 70B.

---

## What it does

Drop any PDF into the `docs/` folder, run the script, and ask questions in plain English. The model is constrained to answer only from the document — it won't pull from its general training knowledge. Every response tells you which page the information came from and shows live token usage so you can see exactly what's being sent to the model.

---

## How it works

This project uses a technique called **context stuffing** — the entire PDF is extracted, converted to plain text, and injected directly into the system prompt before the conversation starts. Every message you send includes the full document text + the conversation history, so the model always has complete context.
The model is stateless — it has no memory of its own. The full conversation history is rebuilt and sent with every single API call. This is how all LLM applications work under the hood.

---

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Provider | [Groq](https://groq.com) (free tier) |
| Model | LLaMA 3.3 70B Versatile |
| PDF Extraction | PyMuPDF (fitz) |
| Config | python-dotenv |

---

## Setup

1. Clone the repo
2. Create and activate a virtual environment
3. Install dependencies: pip install groq python-dotenv pymupdf
4. Get a Groq API key
5. Create your `.env` file, add the key
6. Add a PDF to the docs folder

---

## Usage

```bash
# pass the PDF directly
python main.py docs/yourfile.pdf

# or let it prompt you
python main.py
```

## Key concepts demonstrated

**Grounding**:  The system prompt explicitly instructs the model to only use the provided document. This prevents hallucination by constraining the knowledge source. If you ask something not in the document, it says so instead of making something up.

**Context window**: Every API call sends the full document + full conversation history. Watch the `prompt_tokens` counter grow with each message. This is why the technique breaks down with large documents — you're resending hundreds of thousands of tokens on every turn.

**Temperature**: Set to `0.2` (vs `0.7` in a general chatbot). Lower temperature makes the model more deterministic and factual, which is what you want for document Q&A. Higher temperature introduces creativity — useful for writing, not for reading contracts.

**Stateless API**: The Groq API (like all LLM APIs) has zero memory between calls. The conversation history is a plain Python list that grows each turn and gets unpacked into every request. The "memory" lives entirely in your code, not in the model.

---

## Limitations

This approach works well for documents up to ~50 pages. Beyond that:

- **Token cost** grows fast — you're resending the full document on every message
- **Context window limits** — LLaMA 3.3 70B supports ~128k tokens, but large documents + long conversations will approach this ceiling
- **Scanned PDFs** — PyMuPDF extracts text layer only. Image-based/scanned PDFs return empty text and require OCR preprocessing

These limitations are exactly what **RAG (Retrieval Augmented Generation)** solves — instead of stuffing the whole document, you embed it into chunks and only retrieve the relevant sections per query.
