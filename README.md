# NVIDIA SEC Filing Assistant

An English-only Chainlit chatbot that answers questions using locally indexed NVIDIA SEC filings and financial reports. It uses Google Gemini embeddings, Groq for question answering and speech transcription, and a local Qdrant vector store.

## Features

- Grounded financial answers with clickable filing citations, PDF page links, and supporting excerpts.
- Financial-unit safeguards that distinguish share counts from monetary amounts and preserve table headings.
- Charts for consistent financial tables, CSV table exports, and Markdown answer downloads.
- Helpful / not helpful feedback buttons. Ratings are stored locally without saving answer text.
- Voice input through Groq Whisper and optional spoken answers through ElevenLabs.
- NVIDIA-themed Chainlit UI with English interface text and English-only responses.
- A visible status and one retry when Groq rate-limits a request.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Google AI API key and Groq API key
- ElevenLabs API key (optional; required only for spoken answers)

## Setup

From the project root:

```bash
uv sync
cp .env.example .env
```

Add your API keys to `.env`. Never commit `.env` or real API keys.

Put supported source files (`.pdf`, `.docx`, or `.txt`) directly in `data/`. The ingestion script reads files from that directory (not nested folders) and creates a local Qdrant index:

```bash
mkdir -p data
uv run python src/ingest.py
```

Start the chat application:

```bash
uv run chainlit run src/app.py
```

Open [http://localhost:8000](http://localhost:8000).

## Example questions

- What was NVIDIA's Data Center revenue for the period covered by the filing?
- What were the balances as of July 27, 2025? Keep share counts separate from dollar amounts.
- Compare NVIDIA's revenue across the periods shown in the filing.
- What risks does the filing describe regarding supply constraints or export controls?

## Project layout

```text
.chainlit/config.toml       Chainlit configuration, including English UI language
chainlit.md                 Chainlit welcome-page content
prompts/bot.prompt.md       Financial-answer and English-only instructions
public/                     NVIDIA branding and UI styling
src/app.py                  Chainlit application and chat handlers
src/callbacks.py            LLM callback logging
src/financial_results.py    Financial-table charts and exports
src/ingest.py               Filing ingestion into local Qdrant
src/rag.py                  Retrieval, answer context, and citations
src/text_cleaner.py         Speech-output text cleanup
tests/                      Focused financial citation and export tests
.env.example                Placeholder environment-variable template
pyproject.toml / uv.lock    Project dependencies and lockfile
```

PDFs, the local Qdrant database, feedback storage, virtual environments, and `.env` are local data and should not be committed. The ingestion script rebuilds the NVIDIA collection from the source files in `data/`.

Run the focused tests with:

```bash
uv run --with pytest pytest -q
```
