import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ["GOOGLE_API_KEY"] = "test-google-key"
os.environ["GROQ_API_KEY"] = "test-groq-key"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import (
    _citation_source_lines,
    _format_citations,
    _rate_limit_retry_delay,
    _save_feedback,
    main,
    set_starters,
)
from financial_results import chart_spec, extract_markdown_table, table_to_csv
from ingest import load_pdf
from rag import (
    _financial_unit_note,
    _find_page_number,
    _query_date_pattern,
    _shareholders_equity_row,
)


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FinancialCitationTests(unittest.TestCase):
    def test_pdf_loader_keeps_one_based_page_numbers(self):
        fake_reader = type(
            "FakeReader",
            (),
            {"pages": [FakePage("First paragraph\n\nSecond paragraph"), FakePage("Third")]},
        )()
        with patch("ingest.PdfReader", return_value=fake_reader):
            self.assertEqual(
                load_pdf(Path("filing.pdf")),
                [(1, ["First paragraph", "Second paragraph"]), (2, ["Third"])],
            )

    def test_page_lookup_maps_retrieved_text_to_source_page(self):
        pages = ("other page contents", "Balances as of Jul 27, 2025 24,347 $24 $11,200")
        self.assertEqual(
            _find_page_number("Balances as of Jul 27, 2025 24,347 $24 $11,200", pages),
            2,
        )

    def test_page_lookup_selects_relevant_page_from_legacy_multi_page_chunk(self):
        pages = (
            "General discussion before the statement.",
            "Balances as of Jul 27, 2025 24,347 $24 $11,200.",
            "Unrelated continuation after the statement.",
        )
        chunk = " ".join(pages)
        self.assertEqual(
            _find_page_number(
                chunk,
                pages,
                "Balances as of Jul 27, 2025",
                _query_date_pattern("Balances as of Jul 27, 2025"),
            ),
            2,
        )

    def test_share_column_header_does_not_treat_counts_as_currency(self):
        header = (
            "Common Stock Outstanding Additional Paid-in Capital "
            "Total Shareholders’ EquityShares Amount (In millions, except per share data)"
        )
        self.assertIn("Shares are counts in millions, not dollars", _financial_unit_note(header))

    def test_shareholders_equity_row_maps_each_value_to_its_column(self):
        table = (
            "Common Stock Outstanding Additional Paid-in Capital "
            "Accumulated Other Comprehensive Income (Loss) Retained Earnings "
            "Total Shareholders’ EquityShares Amount (In millions, except per share data) "
            "Balances as of Jul 27, 2025 24,347 $ 24 $ 11,200 $ 170 $ 88,737 $ 100,131"
        )
        mapped = _shareholders_equity_row(
            table,
            _query_date_pattern("Balances as of Jul 27, 2025"),
        )
        self.assertIn("24,347 million shares", mapped)
        self.assertIn("Common stock amount: $24 million", mapped)
        self.assertIn("Total shareholders’ equity: $100,131 million", mapped)

    def test_citation_links_to_pdf_page_and_displays_supporting_passage(self):
        rendered = _format_citations(
            "There were 24,347 million shares 【S1】.",
            {
                "S1": {
                    "source": "NVIDIA filing.pdf",
                    "page_number": 6,
                    "excerpt": "Balances as of Jul 27, 2025: 24,347 shares.",
                }
            },
        )
        self.assertTrue(
            rendered.startswith(
                "There were 24,347 million shares "
                "[NVIDIA filing.pdf, p. 6](/documents/NVIDIA%20filing.pdf#page=6)."
            )
        )
        self.assertIn("[NVIDIA filing.pdf, p. 6](/documents/NVIDIA%20filing.pdf#page=6)", rendered)
        self.assertIn("> Balances as of Jul 27, 2025: 24,347 shares.", rendered)

    def test_citation_sources_can_be_rendered_separately_from_the_answer(self):
        citations = {
            "S1": {
                "source": "NVIDIA filing.pdf",
                "page_number": 6,
                "excerpt": "Revenue was $100 million.",
            }
        }
        self.assertIn(
            "[NVIDIA filing.pdf, p. 6]",
            _citation_source_lines("Revenue was $100 million 【S1】.", citations)[0],
        )
        answer = _format_citations(
            "Revenue was $100 million 【S1】.",
            citations,
            include_sources=False,
        )
        self.assertNotIn("**Sources**", answer)
        self.assertIn("[NVIDIA filing.pdf, p. 6]", answer)

    def test_period_table_can_be_charted_and_exported(self):
        table = extract_markdown_table(
            "| Period | Revenue | Unit |\n"
            "|---|---:|---|\n"
            "| Q1 FY2026 | 100 | million dollars |\n"
            "| Q2 FY2026 | 125 | million dollars |\n"
        )
        self.assertIsNotNone(table)
        chart = chart_spec(table)
        self.assertEqual(chart["kind"], "line")
        self.assertEqual(chart["labels"], ["Q1 FY2026", "Q2 FY2026"])
        self.assertEqual(chart["values"], [100.0, 125.0])
        self.assertEqual(chart["y_label"], "million dollars")
        self.assertIn("Q2 FY2026,125", table_to_csv(table))

    def test_chart_rejects_tables_with_mixed_financial_units(self):
        table = extract_markdown_table(
            "| Item | Value | Unit |\n"
            "|---|---:|---|\n"
            "| Share count | 24,347 | million shares |\n"
            "| Total equity | 100,131 | million dollars |\n"
        )
        self.assertIsNotNone(table)
        self.assertIsNone(chart_spec(table))

    def test_csv_export_escapes_spreadsheet_formulas(self):
        exported = table_to_csv((["Item", "Value"], [["=1+1", "safe"]]))
        self.assertIn("'=1+1", exported)

    def test_feedback_is_saved_and_updated_without_answer_content(self):
        with tempfile.TemporaryDirectory() as directory:
            feedback_path = Path(directory) / "feedback.sqlite3"
            with patch("app.DOCUMENTS_DIR", Path(directory)), patch(
                "app.FEEDBACK_DB_PATH", feedback_path
            ):
                _save_feedback("answer-1", "helpful")
                _save_feedback("answer-1", "not_helpful")

            with sqlite3.connect(feedback_path) as connection:
                row = connection.execute(
                    "SELECT message_id, rating FROM answer_feedback"
                ).fetchone()
                columns = [
                    column[1]
                    for column in connection.execute("PRAGMA table_info(answer_feedback)")
                ]
            self.assertEqual(row, ("answer-1", "not_helpful"))
            self.assertEqual(columns, ["message_id", "rating", "created_at"])


class EmptyResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_welcome_starters_include_a_chart_question(self):
        starters = await set_starters(None, None)
        self.assertEqual(len(starters), 3)
        self.assertTrue(any("chart-ready table" in starter.message for starter in starters))

    async def test_empty_model_response_shows_rate_limit_guidance_instead_of_sources(self):
        class EmptyHandler:
            async def stream_events(self):
                if False:
                    yield None

        class FakeChatEngine:
            def run(self, **kwargs):
                return EmptyHandler()

        class FakeMessage:
            last_message = None

            def __init__(self, content):
                self.content = content
                self.id = "answer-1"
                self.actions = []
                type(self).last_message = self

            async def send(self):
                pass

            async def update(self):
                pass

            async def stream_token(self, _token):
                pass

        session = {
            "chat_engine": FakeChatEngine(),
            "memory": None,
            "citations": {"S1": {"source": "filing.pdf", "page_number": 1, "excerpt": "source"}},
        }
        with (
            patch(
                "app.cl.user_session.get",
                side_effect=lambda key, default=None: session.get(key, default),
            ),
            patch("app.cl.Message", FakeMessage),
        ):
            await main(type("UserMessage", (), {"content": "Show revenue"})())

        self.assertIn(
            "temporarily rate-limited",
            FakeMessage.last_message.content,
        )

    async def test_rate_limit_displays_status_and_retries_using_provider_delay(self):
        class FakeRateLimitError(Exception):
            response = SimpleNamespace(headers={"retry-after-ms": "1250"})

        class RateLimitedHandler:
            async def stream_events(self):
                raise FakeRateLimitError()
                yield None

        class EmptyHandler:
            async def stream_events(self):
                if False:
                    yield None

        class FakeChatEngine:
            def __init__(self):
                self.calls = 0

            def run(self, **kwargs):
                self.calls += 1
                return RateLimitedHandler() if self.calls == 1 else EmptyHandler()

        class FakeMessage:
            last_message = None

            def __init__(self, content):
                self.content = content
                self.id = "answer-1"
                self.actions = []
                self.updates = []
                type(self).last_message = self

            async def send(self):
                pass

            async def update(self):
                self.updates.append(self.content)

            async def stream_token(self, _token):
                pass

        engine = FakeChatEngine()
        session = {"chat_engine": engine, "memory": None, "citations": {}}
        sleep = AsyncMock()
        with (
            patch(
                "app.cl.user_session.get",
                side_effect=lambda key, default=None: session.get(key, default),
            ),
            patch("app.cl.Message", FakeMessage),
            patch("app.RateLimitError", FakeRateLimitError),
            patch("app.asyncio.sleep", sleep),
        ):
            await main(type("UserMessage", (), {"content": "Show revenue"})())

        self.assertEqual(engine.calls, 2)
        self.assertIn(
            "Groq is rate-limited; retrying in 2 seconds...",
            FakeMessage.last_message.updates,
        )
        sleep.assert_awaited_once_with(1.25)
        self.assertIn(
            "temporarily rate-limited",
            FakeMessage.last_message.content,
        )


if __name__ == "__main__":
    unittest.main()
