"""
Purpose: Ingestion pipeline to process financial documents (PDF/DOCX/TXT) into local Qdrant vector database for RAG.
Process: Load Document -> Clean Text -> Chunk -> Deduplicate -> Generate Embeddings -> Store in Qdrant
"""

import os
import time
import uuid
import hashlib
import logging
from pathlib import Path
from dotenv import load_dotenv

import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct

from docx import Document as DocxDocument
from pypdf import PdfReader

from llama_index.core import Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------
# API KEYS & SETTINGS
# -----------------------
API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys (GOOGLE_API_KEY / GROQ_API_KEY) in .env file.")

Settings.embed_model = GoogleGenAIEmbedding(
    model="models/embedding-001",
    api_key=API_KEY,
)

Settings.llm = Groq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
)

# -----------------------
# CONFIG & PATHS
# -----------------------
COLLECTION_NAME = "nvidia_sec_collection"
BASE_DIR = Path(__file__).resolve().parent.parent

# Input directory for documents (PDFs, DOCX, TXT)
DOCS_DIR = BASE_DIR / "data"

# Relative storage path for Qdrant database
QDRANT_PATH = BASE_DIR / "data" / "qdrant_data"


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def clean_text(text: str) -> str:
    """Normalize whitespace and strip text."""
    return " ".join(text.split()).strip()


def hash_text(text: str) -> str:
    """Generate MD5 hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def chunk_paragraphs(paragraphs: list[str], chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """Group paragraphs into smaller overlapping chunks for targeted context."""
    chunks = []
    i = 0
    while i < len(paragraphs):
        chunk = " ".join(paragraphs[i : i + chunk_size])
        if len(chunk) > 30:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# -----------------------
# DOCUMENT LOADERS
# -----------------------
def load_pdf(file_path: Path) -> list[tuple[int, list[str]]]:
    """Extract paragraphs grouped by their one-based PDF page number."""
    reader = PdfReader(file_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            paragraphs = [
                clean_text(paragraph)
                for paragraph in text.split("\n\n")
                if clean_text(paragraph)
            ]
            pages.append((page_number, paragraphs))
    return pages


def load_docx(file_path: Path) -> list[str]:
    """Extract paragraphs from DOCX files."""
    docx = DocxDocument(file_path)
    return [clean_text(p.text) for p in docx.paragraphs if p.text and p.text.strip()]


def load_txt(file_path: Path) -> list[str]:
    """Extract paragraphs from plain text files."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [clean_text(p) for p in text.split("\n\n") if clean_text(p)]


def load_documents_from_dir(docs_dir: Path) -> list[Document]:
    """Recursively scan docs directory and ingest supported formats."""
    if not docs_dir.exists():
        logger.warning(f"Documents directory '{docs_dir}' does not exist.")
        return []

    documents = []
    seen_hashes: set[tuple[str, int | None, str]] = set()

    for file_path in docs_dir.iterdir():
        if file_path.is_dir() or file_path.name.startswith("."):
            continue

        page_chunks: list[tuple[str, int | None]] = []
        ext = file_path.suffix.lower()

        logger.info(f"Processing document: {file_path.name}")
        if ext == ".pdf":
            page_chunks = [
                (chunk, page_number)
                for page_number, paragraphs in load_pdf(file_path)
                for chunk in chunk_paragraphs(paragraphs)
            ]
        elif ext == ".docx":
            page_chunks = [
                (chunk, None)
                for chunk in chunk_paragraphs(load_docx(file_path))
            ]
        elif ext == ".txt":
            page_chunks = [
                (chunk, None)
                for chunk in chunk_paragraphs(load_txt(file_path))
            ]
        else:
            logger.info(f"Skipping unsupported file extension: {file_path.name}")
            continue

        for chunk, page_number in page_chunks:
            chunk_hash = hash_text(chunk)
            deduplication_key = (file_path.name, page_number, chunk_hash)
            if deduplication_key in seen_hashes:
                continue
            seen_hashes.add(deduplication_key)

            metadata: dict[str, str | int] = {
                "source": file_path.name,
                "file_type": ext.replace(".", ""),
                "category": "sec_filing",
            }
            if page_number is not None:
                metadata["page_number"] = page_number

            documents.append(
                Document(
                    text=chunk,
                    metadata=metadata,
                )
            )

    return documents


# -----------------------
# VECTOR STORE EMBEDDING
# -----------------------
def embed_and_store(documents: list[Document], client: qdrant_client.QdrantClient):
    """Embed document chunks and store in Qdrant collection."""
    if not documents:
        logger.warning("No documents to embed. Exiting storage phase.")
        return

    # Delete existing collection if present for a clean re-index
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        logger.info("Deleting existing vector collection...")
        client.delete_collection(COLLECTION_NAME)

    # Get vector dimensions
    logger.info("Determining embedding dimension...")
    test_emb = Settings.embed_model.get_text_embedding("NVIDIA 10-Q Financial Report")
    vector_size = len(test_emb)
    logger.info(f"Embedding dimension size: {vector_size}")

    # Create new Qdrant collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Generate embeddings and build PointStructs
    points = []
    for i, doc in enumerate(documents):
        logger.info(f"Embedding chunk {i+1}/{len(documents)}: {doc.text[:60]}...")
        try:
            embedding = Settings.embed_model.get_text_embedding(doc.text)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": doc.text,
                        **doc.metadata,
                    },
                )
            )
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"Error embedding document chunk {i+1}: {e}")
            continue

    # Batch upload points into Qdrant
    logger.info(f"Uploading {len(points)} points to Qdrant...")
    batch_size = 10
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}")

    logger.info("Vector ingestion completed successfully.")


# -----------------------
# MAIN EXECUTION
# -----------------------
def main():
    logger.info(f"Initializing Qdrant client at path: {QDRANT_PATH}")
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = qdrant_client.QdrantClient(path=str(QDRANT_PATH))

    logger.info(f"Scanning for documents in: {DOCS_DIR}")
    documents = load_documents_from_dir(DOCS_DIR)
    logger.info(f"Total document chunks extracted: {len(documents)}")

    embed_and_store(documents, client)

    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()