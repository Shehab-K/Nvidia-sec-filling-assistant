"""
Purpose: Implements the main Chainlit chat interface, handling audio transcription, TTS generation, and user interactions.
Chainlit application with audio transcription and TTS support.
Handles voice input/output with Groq Whisper and ElevenLabs TTS for NVIDIA SEC filing QA.
"""
import asyncio
import chainlit as cl
import sys
import os
import rag
import logging
import io
import math
import sqlite3
import wave
import re
import html
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from dotenv import load_dotenv
from groq import AsyncGroq
from openai import RateLimitError
from elevenlabs.client import ElevenLabs
from fastapi import HTTPException
from fastapi.responses import FileResponse
from chainlit.server import app as chainlit_app

from llama_index.core.agent.workflow import AgentStream
from llama_index.core.memory import ChatMemoryBuffer
from financial_results import chart_spec, extract_markdown_table, table_to_csv
from text_cleaner import clean_text_for_tts

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))

# Constants
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2
CHANNELS = 1
DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_LIMIT = 25
FEEDBACK_DB_PATH = DOCUMENTS_DIR / "financial_feedback.sqlite3"
CITATION_MARKER_PATTERN = re.compile(r"\[(S\d+)\]|【(S\d+)】")
DEFAULT_RATE_LIMIT_RETRY_SECONDS = 36


@chainlit_app.get("/documents/{filename}", include_in_schema=False)
async def serve_filing_pdf(filename: str):
    """Serve an indexed filing PDF so citations can open at a specific page."""
    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Document not found")

    documents_dir = DOCUMENTS_DIR.resolve()
    pdf_path = (documents_dir / candidate).resolve()
    if pdf_path.parent != documents_dir or not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )


# Keep this route ahead of Chainlit's SPA catch-all route.
document_routes = [
    route
    for route in chainlit_app.router.routes
    if getattr(route, "path", None) == "/documents/{filename}"
]
document_route = document_routes[-1]
for route in document_routes:
    chainlit_app.router.routes.remove(route)
catch_all_index = next(
    (
        index
        for index, route in enumerate(chainlit_app.router.routes)
        if getattr(route, "path", None) == "/{full_path:path}"
    ),
    len(chainlit_app.router.routes),
)
chainlit_app.router.routes.insert(catch_all_index, document_route)


def _format_citations(
    content: str,
    citations: dict[str, dict[str, str | int | None]],
    include_sources: bool = True,
) -> str:
    source_lines = _citation_source_lines(content, citations)
    content = CITATION_MARKER_PATTERN.sub(
        lambda match: _citation_link(match, citations),
        content,
    )
    if source_lines and include_sources:
        content += "\n\n**Sources**\n\n" + "\n\n".join(source_lines)
    return content


def _citation_source_lines(
    content: str,
    citations: dict[str, dict[str, str | int | None]],
) -> list[str]:
    cited_ids = list(
        dict.fromkeys(
            match.group(1) or match.group(2)
            for match in CITATION_MARKER_PATTERN.finditer(content)
        )
    )
    source_ids = [citation_id for citation_id in cited_ids if citation_id in citations]
    source_ids = source_ids or list(citations)
    source_lines = []
    for citation_id in source_ids:
        citation = citations.get(citation_id)
        if not citation:
            continue

        source = str(citation["source"])
        page_number = citation["page_number"]
        if source.lower().endswith(".pdf"):
            label = (
                f"{source}, p. {page_number}"
                if page_number
                else f"{source} (page unavailable)"
            )
            url = f"/documents/{quote(source, safe='')}"
            if page_number:
                url += f"#page={page_number}"
            label = f"[{label}]({url})"
        else:
            label = f"{source}, p. {page_number}" if page_number else source

        excerpt = html.escape(str(citation["excerpt"]), quote=False)
        source_lines.append(f"- **[{citation_id}]** {label}\n  > {excerpt}")

    return source_lines


def _citation_link(
    match: re.Match[str],
    citations: dict[str, dict[str, str | int | None]],
) -> str:
    citation_id = match.group(1) or match.group(2)
    citation = citations.get(citation_id)
    if not citation:
        return match.group(0)

    source = str(citation["source"])
    page_number = citation["page_number"]
    if not source.lower().endswith(".pdf"):
        return match.group(0)

    label = (
        f"{source}, p. {page_number}"
        if page_number
        else f"{source} (page unavailable)"
    )
    url = f"/documents/{quote(source, safe='')}"
    if page_number:
        url += f"#page={page_number}"
    return f"[{label}]({url})"


def _rate_limit_retry_delay(error: RateLimitError) -> float:
    headers = getattr(getattr(error, "response", None), "headers", {})
    retry_after_ms = headers.get("retry-after-ms")
    try:
        delay = float(retry_after_ms) / 1000
        if delay > 0:
            return delay
    except (TypeError, ValueError):
        pass

    retry_after = headers.get("retry-after")
    try:
        delay = float(retry_after)
        if delay > 0:
            return delay
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            if delay > 0:
                return delay
        except (TypeError, ValueError, OverflowError):
            pass

    return float(DEFAULT_RATE_LIMIT_RETRY_SECONDS)


def _save_feedback(message_id: str, rating: str) -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_feedback (
                message_id TEXT PRIMARY KEY,
                rating TEXT NOT NULL CHECK (rating IN ('helpful', 'not_helpful')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO answer_feedback (message_id, rating)
            VALUES (?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                rating = excluded.rating,
                created_at = CURRENT_TIMESTAMP
            """,
            (message_id, rating),
        )


def _action_result(action: cl.Action) -> dict | None:
    message_id = action.payload.get("message_id")
    if not isinstance(message_id, str):
        return None
    result_cache = cl.user_session.get("financial_results", {})
    return result_cache.get(message_id)


@cl.action_callback("export_markdown")
async def on_export_markdown(action: cl.Action):
    result = _action_result(action)
    if not result or not result.get("content"):
        await cl.ErrorMessage(content="This answer is no longer available to export.").send()
        return

    answer_file = cl.File(
        name="financial-answer.md",
        content=result["content"].encode("utf-8"),
        mime="text/markdown",
    )
    await cl.Message(content="Download the answer with its citations:", elements=[answer_file]).send()
    await action.remove()


@cl.action_callback("export_csv")
async def on_export_csv(action: cl.Action):
    result = _action_result(action)
    if not result or not result.get("csv"):
        await cl.ErrorMessage(content="This answer does not contain an exportable table.").send()
        return

    csv_file = cl.File(
        name="financial-results.csv",
        content=result["csv"].encode("utf-8"),
        mime="text/csv",
    )
    await cl.Message(content="Download the financial table:", elements=[csv_file]).send()
    await action.remove()


@cl.action_callback("show_chart")
async def on_show_chart(action: cl.Action):
    result = _action_result(action)
    chart = result.get("chart") if result else None
    if not isinstance(chart, dict):
        await cl.ErrorMessage(content="This answer does not contain chartable data.").send()
        return

    import plotly.graph_objects as go

    if chart["kind"] == "line":
        trace = go.Scatter(
            x=chart["labels"],
            y=chart["values"],
            mode="lines+markers",
            name=chart["title"],
            line={"color": "#76b900"},
        )
    else:
        trace = go.Bar(
            x=chart["labels"],
            y=chart["values"],
            name=chart["title"],
            marker={"color": "#76b900"},
        )

    figure = go.Figure(data=[trace])
    figure.update_layout(
        title=chart["title"],
        xaxis_title=chart["x_label"],
        yaxis_title=chart["y_label"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#808080"},
        xaxis={"gridcolor": "rgba(128,128,128,0.2)"},
        yaxis={"gridcolor": "rgba(128,128,128,0.2)"},
        margin={"l": 60, "r": 24, "t": 56, "b": 60},
    )
    plot = cl.Plotly(name="Financial chart", figure=figure, display="inline")
    await cl.Message(content=f"**{chart['title']}**", elements=[plot]).send()
    await action.remove()


@cl.action_callback("answer_feedback")
async def on_answer_feedback(action: cl.Action):
    rating = action.payload.get("rating")
    message_id = action.payload.get("message_id")
    if rating not in {"helpful", "not_helpful"} or not isinstance(message_id, str):
        await cl.ErrorMessage(content="Unable to save feedback for this answer.").send()
        return

    try:
        _save_feedback(message_id, rating)
    except (OSError, sqlite3.Error):
        logger.exception("Could not save answer feedback")
        await cl.ErrorMessage(content="Could not save your feedback. Please try again.").send()
        return

    await cl.Message(content="Thanks for your feedback.").send()
    await action.remove()


async def transcribe_audio(client: AsyncGroq, wav_buffer: io.BytesIO) -> str:
    """Transcribe audio using Groq Whisper."""
    transcription = await client.audio.transcriptions.create(
        file=("audio.wav", wav_buffer),
        model="whisper-large-v3",
        response_format="json",
        temperature=0.0,
        prompt="This is an audio query regarding NVIDIA SEC filings and financial reports."
    )
    return transcription.text.strip()


def generate_tts(el_client: ElevenLabs, text: str) -> bytes:
    """Generate TTS audio from text using ElevenLabs."""
    cleaned_text = clean_text_for_tts(text)
    cleaned_text = re.sub(r'\*+', '', cleaned_text)
    cleaned_text = re.sub(r'#+', '', cleaned_text)
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'---', '', cleaned_text)
    cleaned_text = re.sub(r'🤖.*', '', cleaned_text)
    cleaned_text = re.sub(r'<br\s*/?>', ' ', cleaned_text)
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())
    cleaned_text = cleaned_text[:200]

    logger.info(f"TTS input ({len(cleaned_text)} chars): '{cleaned_text[:80]}...'")

    response = el_client.text_to_speech.convert(
        text=cleaned_text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_v3",
        output_format="mp3_44100",
    )
    return b"".join(response)


@cl.action_callback("audio_play")
async def on_action_audio_play(action: cl.Action):
    """Handle audio play action."""
    text = action.payload.get("value")
    if not text:
        return

    el_client = cl.user_session.get("elevenlabs_client")
    if not el_client:
        await cl.ErrorMessage(content="ElevenLabs API Key is missing.").send()
        return

    try:
        audio_data = generate_tts(el_client, text)

        audio_element = cl.Audio(
            auto_play=True,
            mime="audio/mpeg",
            content=audio_data,
        )

        await cl.Message(
            content=text,
            id=action.forId,
            elements=[audio_element],
        ).update()
        await action.remove()

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        await cl.ErrorMessage(content="Failed to generate TTS audio.").send()


@cl.set_starters
async def set_starters(user: cl.User | None, language: str | None):
    return [
        cl.Starter(
            label="Chart revenue",
            message=(
                "Compare NVIDIA total revenue for the three months ended July 26, 2026 "
                "and July 27, 2025. Return a chart-ready table with Period, Revenue, "
                "and Unit (millions of dollars)."
            ),
        ),
        cl.Starter(
            label="Check shareholder equity",
            message=(
                "What were NVIDIA's balances as of July 27, 2025? Keep share counts "
                "separate from dollar amounts and include citations."
            ),
        ),
        cl.Starter(
            label="Compare segments",
            message=(
                "Compare Compute & Networking and Graphics revenue for the three months "
                "ended July 26, 2026 and July 27, 2025, with units and citations."
            ),
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize chat session with clients and RAG engine."""
    logger.info("Starting new chat session")

    # Initialize Groq client for Whisper STT
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = AsyncGroq(api_key=groq_api_key)
        cl.user_session.set("groq_client", groq_client)
        logger.info("Groq client initialized")
    else:
        logger.warning("GROQ_API_KEY not found")

    # Initialize ElevenLabs client for TTS
    el_api_key = os.getenv("ELEVENLABS_API_KEY")
    if el_api_key:
        el_client = ElevenLabs(api_key=el_api_key)
        cl.user_session.set("elevenlabs_client", el_client)
        logger.info("ElevenLabs client initialized")
    else:
        logger.warning("ELEVENLABS_API_KEY not found")

    # Initialize RAG chat engine
    logger.info("Initializing RAG chat engine")
    chat_engine, citations = await rag.get_chat_engine()
    cl.user_session.set("chat_engine", chat_engine)
    cl.user_session.set("citations", citations)
    cl.user_session.set("financial_results", {})

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    cl.user_session.set("memory", memory)
    logger.info("Chat engine initialized")


@cl.on_audio_start
async def on_audio_start():
    """Initialize audio buffer for recording."""
    try:
        logger.info("Audio recording started")
        cl.user_session.set("audio_chunks", [])
        return True
    except Exception as e:
        logger.error(f"Error in on_audio_start: {e}")
        return False


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Append audio chunks to buffer."""
    if chunk.data:
        new_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks = cl.user_session.get("audio_chunks")
        if audio_chunks is not None:
            audio_chunks.append(new_chunk)


@cl.on_audio_end
async def on_audio_end():
    """Process recorded audio and transcribe using Groq Whisper."""
    logger.info("Audio recording ended")

    try:
        audio_chunks = cl.user_session.get("audio_chunks")

        if not audio_chunks:
            await cl.Message(content="No audio detected.").send()
            return

        audio_data = np.concatenate(audio_chunks)

        # Ignore very short recordings (less than 2 seconds)
        min_samples = SAMPLE_RATE * 2
        if len(audio_data) < min_samples:
            logger.info("Recording too short, ignoring")
            return

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())

        wav_buffer.seek(0)

        groq_client = cl.user_session.get("groq_client")
        if not groq_client:
            logger.error("Groq client not found")
            await cl.Message(content="Groq API Key is missing.").send()
            return

        logger.info("Sending audio to Groq Whisper")
        text = await transcribe_audio(groq_client, wav_buffer)
        logger.info(f"Transcription: '{text}'")

        # Filter Whisper hallucinations
        hallucinations = [
            "thank you for watching",
            "thank you for watching!",
            "thanks for watching",
            "please subscribe",
            "like and subscribe",
            ".",
            "...",
        ]
        if not text or text.lower().strip() in hallucinations:
            logger.info(f"Ignoring hallucinated transcription: '{text}'")
            return

        if not text:
            logger.warning("Empty transcription received")
            await cl.Message(content="Could not understand audio. Please speak clearly and try again.").send()
            return

        hidden_msg = cl.Message(
            content=f"///TRANSCRIPTION/// {text}",
            author="System",
            type="system_message"
        )
        await hidden_msg.send()

    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
        await cl.ErrorMessage(content=f"Audio processing error: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and generate responses."""
    logger.info(f"Received message: {message.content[:100]}...")

    chat_engine = cl.user_session.get("chat_engine")
    memory = cl.user_session.get("memory")
    citations = cl.user_session.get("citations", {})
    citations.clear()

    if not chat_engine:
        logger.error("Chat engine not found")
        await cl.Message(content="System not ready. Please reload the page.").send()
        return

    msg = cl.Message(content="")
    await msg.send()

    full_response = ""

    try:
        logger.info("Querying RAG engine")
        memory_snapshot = memory.get_all() if memory else None
        for attempt in range(2):
            try:
                handler = chat_engine.run(
                    user_msg=message.content,
                    memory=memory,
                    streaming=True,
                )
                async for event in handler.stream_events():
                    if isinstance(event, AgentStream):
                        token = event.delta
                        token = token.replace("Task manually stopped", "Task stopped")
                        full_response += token
                        await msg.stream_token(token)
            except RateLimitError as error:
                if attempt:
                    raise

                retry_delay = _rate_limit_retry_delay(error)
                retry_seconds = max(1, math.ceil(retry_delay))
                logger.warning(
                    "Groq rate limited the request; retrying in %.2f seconds",
                    retry_delay,
                )
                msg.content = (
                    f"Groq is rate-limited; retrying in {retry_seconds} seconds..."
                )
                await msg.update()
                await asyncio.sleep(retry_delay)
                full_response = ""
                msg.content = ""
                citations.clear()
                if memory is not None and memory_snapshot is not None:
                    memory.set(memory_snapshot)
                await msg.update()
                continue
            break

        if not full_response.strip():
            logger.warning("The chat engine returned no answer tokens")
            msg.content = (
                "The model did not return an answer. It may be temporarily rate-limited; "
                "please wait about a minute and try again."
            )
            await msg.update()
            return

        msg.content = msg.content.replace("Task manually stopped", "Task stopped")
        full_response = full_response.replace("Task manually stopped", "Task stopped")

        if "##CLARIFICATION##" in msg.content:
            msg.content = msg.content.replace("##CLARIFICATION##\n", "")
            full_response = full_response.replace("##CLARIFICATION##\n", "")

        msg.content = re.sub(r'<br\s*/?>', '\n', msg.content)
        msg.content = re.sub(r'<[^>]+>', '', msg.content)
        raw_content = msg.content
        source_lines = _citation_source_lines(raw_content, citations)
        export_content = _format_citations(raw_content, citations)
        msg.content = _format_citations(
            raw_content,
            citations,
            include_sources=False,
        )

        await msg.update()
        if source_lines:
            async with cl.Step(
                name=f"Sources ({len(source_lines)})",
                type="run",
                parent_id=msg.id,
                default_open=False,
            ) as source_step:
                source_step.output = "\n\n".join(source_lines)
        logger.info("Response streaming complete")

        if full_response.strip():
            table = extract_markdown_table(msg.content)
            chart = chart_spec(table) if table else None
            result_cache = cl.user_session.get("financial_results", {})
            result_cache[msg.id] = {
                "content": export_content,
                "csv": table_to_csv(table) if table else None,
                "chart": chart,
            }
            while len(result_cache) > CACHE_LIMIT:
                result_cache.pop(next(iter(result_cache)))
            cl.user_session.set("financial_results", result_cache)

            actions = [
                cl.Action(
                    name="export_markdown",
                    label="Download answer",
                    icon="Download",
                    payload={"message_id": msg.id},
                ),
                cl.Action(
                    name="answer_feedback",
                    label="Helpful",
                    icon="ThumbsUp",
                    payload={"message_id": msg.id, "rating": "helpful"},
                ),
                cl.Action(
                    name="answer_feedback",
                    label="Not helpful",
                    icon="ThumbsDown",
                    payload={"message_id": msg.id, "rating": "not_helpful"},
                ),
                cl.Action(
                    name="audio_play",
                    icon="Volume2",
                    payload={
                        "value": re.sub(r"\[S\d+\]|【S\d+】", "", full_response),
                    },
                    label="",
                ),
            ]
            if table:
                actions.append(
                    cl.Action(
                        name="export_csv",
                        label="Export CSV",
                        icon="FileSpreadsheet",
                        payload={"message_id": msg.id},
                    )
                )
            if chart:
                actions.append(
                    cl.Action(
                        name="show_chart",
                        label="Chart",
                        icon="BarChart3",
                        payload={"message_id": msg.id},
                    )
                )

            msg.actions = actions
            await msg.update()

    except RateLimitError:
        logger.warning("Groq rate limit exceeded while processing a chat message")
        msg.content = (
            "The AI service is temporarily rate-limited. "
            "Please try again in about a minute."
        )
        await msg.update()
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        error_msg = str(e).replace("Task manually stopped", "Task stopped")
        if "Task stopped" in error_msg or "cancelled" in error_msg.lower():
            msg.content = msg.content.replace("Task manually stopped", "Task stopped")
            await msg.update()
        else:
            await msg.stream_token(f"\n\n**Error:** {error_msg}")
            await msg.update()