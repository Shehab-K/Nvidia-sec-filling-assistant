"""
Purpose: Core RAG engine definition establishing AsyncQdrantClient retrieval tools
and setting up the LlamaIndex FunctionAgent for financial QA answering.
"""

import os
import sys
import re
import logging
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

import qdrant_client
from qdrant_client.http.models import ScoredPoint
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.callbacks import CallbackManager

from callbacks import SimplePromptLogger

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
)
QUERY_STOPWORDS = {"a", "an", "as", "for", "in", "of", "on", "the", "what", "was"}
MAX_CONTEXT_CHARS = 1200
MAX_DATE_CONTEXT_CHARS = 2200
MAX_CITATION_EXCERPT_CHARS = 700
MAX_CONTEXT_CHUNKS = 3


def _query_date_pattern(query: str) -> re.Pattern[str] | None:
    match = re.search(
        rf"\b({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    month, day, year = match.groups()
    return re.compile(
        rf"\b{re.escape(month[:3])}[a-z]*\s+0?{int(day)},?\s+{year}\b",
        flags=re.IGNORECASE,
    )


def _normalized_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _retrieval_rank(
    point: ScoredPoint,
    query: str,
    query_date: re.Pattern[str] | None,
) -> tuple[bool, bool, int, float]:
    text = " ".join((point.payload or {}).get("text", "").split())
    normalized_text = _normalized_text(text)
    normalized_query = _normalized_text(query)
    query_terms = set(normalized_query.split()) - QUERY_STOPWORDS
    overlap = sum(term in normalized_text.split() for term in query_terms)
    return (
        bool(normalized_query and normalized_query in normalized_text),
        bool(query_date and query_date.search(text)),
        overlap,
        float(point.score or 0),
    )


def _context_excerpt(
    text: str,
    query: str,
    query_date: re.Pattern[str] | None,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    text = " ".join(text.split())
    match = query_date.search(text) if query_date else None
    if match:
        start = max(0, match.start() - max_chars * 3 // 4)
    else:
        terms = set(_normalized_text(query).split()) - QUERY_STOPWORDS
        positions = [
            found.start()
            for term in terms
            if (found := re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))
        ]
        start = max(0, min(positions, default=0) - max_chars // 4)

    end = min(len(text), start + max_chars)
    if end == len(text):
        start = max(0, end - max_chars)

    excerpt = text[start:end]
    return f"{'… ' if start else ''}{excerpt}{' …' if end < len(text) else ''}"


def _find_page_number(
    text: str,
    pages: tuple[str, ...],
    query: str = "",
    query_date: re.Pattern[str] | None = None,
) -> int | None:
    """Find a one-based PDF page containing a retrieved chunk or its unique prefix."""
    normalized_chunk = _normalized_text(text)
    if not normalized_chunk:
        return None
    normalized_pages = tuple(_normalized_text(page_text) for page_text in pages)

    exact_matches = [
        page_number
        for page_number, page_text in enumerate(normalized_pages, start=1)
        if normalized_chunk in page_text
    ]
    if exact_matches:
        return exact_matches[0]

    embedded_pages = [
        (normalized_chunk.find(page_text), page_number, page_text)
        for page_number, page_text in enumerate(normalized_pages, start=1)
        if page_text and page_text in normalized_chunk
    ]
    if embedded_pages:
        if query_date:
            row_match = next(
                (
                    match
                    for match in query_date.finditer(normalized_chunk)
                    if re.search(
                        r"\bbalances\s+as\s+of\s+$",
                        normalized_chunk[:match.start()],
                    )
                ),
                None,
            )
            if row_match:
                target_position = row_match.start()
                for start, page_number, page_text in embedded_pages:
                    if start <= target_position < start + len(page_text):
                        return page_number

        query_terms = set(_normalized_text(query).split()) - QUERY_STOPWORDS
        if query_terms:
            _, page_number, _ = max(
                embedded_pages,
                key=lambda page: sum(term in page[2].split() for term in query_terms),
            )
            return page_number
        return embedded_pages[0][1]

    for anchor_length in (160, 120, 80, 40):
        anchor = normalized_chunk[:anchor_length]
        if len(anchor) < 40:
            continue
        matches = [
            page_number
            for page_number, page_text in enumerate(normalized_pages, start=1)
            if anchor in page_text
        ]
        if len(matches) == 1:
            return matches[0]
    return None


@lru_cache(maxsize=32)
def _pdf_page_text(source: str) -> tuple[str, ...]:
    """Load normalized PDF pages for citation fallback when old indexes lack page metadata."""
    data_dir = (Path(__file__).resolve().parent.parent / "data").resolve()
    if Path(source).name != source or Path(source).suffix.lower() != ".pdf":
        return ()

    pdf_path = (data_dir / source).resolve()
    if pdf_path.parent != data_dir or not pdf_path.is_file():
        return ()

    try:
        return tuple(
            _normalized_text(page.extract_text() or "")
            for page in PdfReader(pdf_path).pages
        )
    except (OSError, PdfReadError):
        logger.warning("Unable to read PDF pages for citation source %s", source, exc_info=True)
        return ()


def _resolve_page_number(
    source: str,
    text: str,
    query: str,
    query_date: re.Pattern[str] | None,
) -> int | None:
    return _find_page_number(text, _pdf_page_text(source), query, query_date)


def _financial_unit_note(text: str) -> str:
    if (
        re.search(r"\bcommon\s+stock\s+outstanding\b", text, re.IGNORECASE)
        and re.search(r"shares\s+amount", text, re.IGNORECASE)
    ):
        return (
            "Unit note: The Common Stock Outstanding table separates Shares from Amount. "
            "Shares are counts in millions, not dollars; monetary columns are in millions "
            "of dollars. Preserve this distinction.\n"
        )
    return ""


def _shareholders_equity_row(
    text: str,
    query_date: re.Pattern[str] | None,
) -> str:
    if (
        not query_date
        or not re.search(r"\bcommon\s+stock\s+outstanding\b", text, re.IGNORECASE)
        or not re.search(r"shares\s+amount", text, re.IGNORECASE)
    ):
        return ""

    date_match = next(
        (
            match
            for match in query_date.finditer(text)
            if re.search(r"\bbalances\s+as\s+of\s+$", text[:match.start()], re.IGNORECASE)
        ),
        None,
    )
    if not date_match:
        return ""

    value_pattern = re.compile(
        r"(?<![\w])(?:\$[\s\u00a0]*)?(?:\([\d,]+(?:\.\d+)?\)|[\d,]+(?:\.\d+)?|[—–-])"
    )
    values = [
        match.group(0).replace("$", "").strip()
        for match in value_pattern.finditer(text, date_match.end())
    ][:6]
    if len(values) != 6:
        return ""

    return (
        f"Column-mapped row for Balances as of {date_match.group(0)} "
        f"(all values in millions, except per-share data): "
        f"Common stock outstanding: {values[0]} million shares; "
        f"Common stock amount: ${values[1]} million; "
        f"Additional paid-in capital: ${values[2]} million; "
        f"Accumulated other comprehensive income (loss): ${values[3]} million; "
        f"Retained earnings: ${values[4]} million; "
        f"Total shareholders’ equity: ${values[5]} million.\n"
    )


# -----------------------
# API KEYS
# -----------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys (GOOGLE_API_KEY / GROQ_API_KEY) in .env")

# -----------------------
# CALLBACK LOGGING
# -----------------------
prompt_logger = SimplePromptLogger()
Settings.callback_manager = CallbackManager([prompt_logger])

# -----------------------
# MODELS
# -----------------------
Settings.embed_model = GoogleGenAIEmbedding(
    model="models/embedding-001",
    api_key=API_KEY,
)

Settings.llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    max_tokens=1024,
    max_retries=0,
    context_window=8192,
    reasoning_effort="low",
)

# -----------------------
# CONFIG & PATHS
# -----------------------
COLLECTION_NAME = "nvidia_sec_collection"
BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = BASE_DIR / "data" / "qdrant_data"


# -----------------------
# QDRANT CLIENT
# -----------------------
def get_aclient():
    if not hasattr(sys, "_rag_qdrant_client"):
        logger.info(f"Initializing shared AsyncQdrantClient at: {QDRANT_PATH}")
        sys._rag_qdrant_client = qdrant_client.AsyncQdrantClient(
            path=str(QDRANT_PATH)
        )
    return sys._rag_qdrant_client


# -----------------------
# CHAT ENGINE SETUP
# -----------------------
async def get_chat_engine():
    logger.info("Initializing Financial RAG ChatEngine...")

    aclient = get_aclient()
    citation_registry: dict[str, dict[str, str | int | None]] = {}

    if not await aclient.collection_exists(COLLECTION_NAME):
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. Run ingestion (ingest.py) first."
        )

    # -----------------------
    # TOOL FUNCTION
    # -----------------------
    async def search_sec_filings(query: str) -> str:
        """
        Search Qdrant for relevant text chunks inside NVIDIA SEC filings.
        """
        query_embedding = Settings.embed_model.get_text_embedding(query)

        results = await aclient.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=8,
        )
        results = results.points

        if not results:
            return "No relevant information found in the financial database."

        query_date = _query_date_pattern(query)
        ranked_results = sorted(
            results,
            key=lambda point: _retrieval_rank(point, query, query_date),
            reverse=True,
        )

        def format_result(result: ScoredPoint, max_chars: int) -> str:
            payload = result.payload or {}
            text = payload.get("text", "")
            source = str(payload.get("source", "Document"))
            page_value = payload.get("page_number")
            try:
                page_number = int(page_value) if page_value is not None else None
            except (TypeError, ValueError):
                logger.warning("Invalid PDF page metadata for source %s: %r", source, page_value)
                page_number = None
            if page_number is not None and page_number < 1:
                page_number = None
            if page_number is None:
                page_number = _resolve_page_number(source, text, query, query_date)

            citation_id = next(
                (
                    existing_id
                    for existing_id, citation in citation_registry.items()
                    if citation["source"] == source
                    and citation["page_number"] == page_number
                ),
                None,
            )
            passage = _context_excerpt(
                text,
                query,
                query_date,
                max_chars=MAX_CITATION_EXCERPT_CHARS,
            )
            if citation_id is None:
                citation_id = f"S{len(citation_registry) + 1}"
                citation_registry[citation_id] = {
                    "source": source,
                    "page_number": page_number,
                    "excerpt": passage,
                }

            page_label = f", PDF page {page_number}" if page_number else ""
            unit_note = _financial_unit_note(text)
            mapped_row = _shareholders_equity_row(text, query_date)
            excerpt = _context_excerpt(text, query, query_date, max_chars=max_chars)
            return (
                f"[{citation_id}] Filing: {source}{page_label}\n"
                f"{unit_note}{mapped_row}{excerpt}"
            )

        if query_date:
            for result in ranked_results:
                text = (result.payload or {}).get("text", "")
                if query_date.search(text):
                    return format_result(result, MAX_DATE_CONTEXT_CHARS)

        context_chunks = []
        for result in ranked_results:
            text = (result.payload or {}).get("text", "")
            if text:
                context_chunks.append(format_result(result, MAX_CONTEXT_CHARS))
            if len(context_chunks) >= MAX_CONTEXT_CHUNKS:
                break

        context = "\n\n".join(context_chunks)

        # Remove potential leftover formatting noise
        context = re.sub(r'---\s*🤖.*', '', context, flags=re.DOTALL)
        return context

    # -----------------------
    # TOOL CREATION
    # -----------------------
    tool = FunctionTool.from_defaults(
        async_fn=search_sec_filings,
        name="search_sec_filings",
        description=(
            "Search for relevant textual and financial information inside "
            "NVIDIA SEC filings (Form 10-Q/10-K), financial statements, and notes. "
            "Preserve exact dates from the user's question and include matching "
            "table column headings. Cite retrieved passages using their [S#] source IDs."
        ),
    )

    # -----------------------
    # SYSTEM PROMPT
    # -----------------------
    prompt_path = BASE_DIR / "prompts" / "bot.prompt.md"
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "You are a helpful AI assistant specializing in analyzing SEC filings "
            "and financial reports. Answer strictly using retrieved context."
        )

    # -----------------------
    # AGENT INITIALIZATION
    # -----------------------
    agent = FunctionAgent(
        name="FinancialRAGBot",
        tools=[tool],
        llm=Settings.llm,
        system_prompt=system_prompt,
        max_iterations=3,
    )

    return agent, citation_registry