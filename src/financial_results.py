"""Helpers for exporting and charting financial tables returned by the assistant."""

import csv
import io
import re

MarkdownTable = tuple[list[str], list[list[str]]]


def _split_table_row(line: str) -> list[str]:
    return [cell.replace("\\|", "|").strip() for cell in line.strip().strip("|").split("|")]


def extract_markdown_table(content: str) -> MarkdownTable | None:
    """Return the first well-formed Markdown table in an answer."""
    content = content.split("**Sources**", 1)[0]
    lines = content.splitlines()

    for index in range(len(lines) - 1):
        if "|" not in lines[index] or "|" not in lines[index + 1]:
            continue

        headers = _split_table_row(lines[index])
        separators = _split_table_row(lines[index + 1])
        if len(headers) < 2 or len(headers) != len(separators):
            continue
        if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separators):
            continue

        rows = []
        for line in lines[index + 2 :]:
            if not line.strip() or "|" not in line:
                break
            row = _split_table_row(line)
            if len(row) != len(headers):
                break
            rows.append(row)
        if rows:
            return headers, rows
    return None


def table_to_csv(table: MarkdownTable) -> str:
    """Serialize a Markdown table as spreadsheet-safe CSV."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    headers, rows = table
    writer.writerow([_safe_csv_cell(cell) for cell in headers])
    writer.writerows([_safe_csv_cell(cell) for cell in row] for row in rows)
    return output.getvalue()


def _numeric_value(cell: str) -> float | None:
    value = cell.strip().replace(",", "").replace("\u202f", "").replace("\xa0", "")
    value = value.replace("−", "-")
    if not value or value in {"—", "–", "-"}:
        return None

    is_negative = value.startswith("(") and value.endswith(")")
    value = value.strip("()")
    value = re.sub(r"[$€£%]", "", value)
    value = re.sub(
        r"\b(?:million|billion|trillion|dollars?|shares?|usd)\b",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None

    number = float(value)
    return -abs(number) if is_negative else number


def _unit_identity(text: str) -> str | None:
    lowered = text.casefold()
    if "%" in text or "percent" in lowered:
        return "percentage"
    if re.search(r"\bshares?\b", lowered):
        return "shares"
    if "$" in text or "dollar" in lowered or "usd" in lowered:
        return "currency"
    return None


def _unit_label(identity: str, text: str) -> str:
    lowered = text.casefold()
    scale = next(
        (
            name
            for name in ("trillions", "billions", "millions", "thousands")
            if name[:-1] in lowered or name in lowered
        ),
        "",
    )
    if identity == "percentage":
        return "Percent"
    if identity == "shares":
        return f"Shares{f' ({scale})' if scale else ''}"
    return f"USD{f' ({scale})' if scale else ''}"


def chart_spec(table: MarkdownTable) -> dict[str, object] | None:
    """Build a chart specification only when a table has consistent, labeled units."""
    headers, rows = table
    unit_columns = [
        index
        for index, header in enumerate(headers)
        if header.strip().casefold() in {"unit", "units"}
    ]
    unit_index = unit_columns[0] if unit_columns else None
    value_columns = [
        index
        for index in range(1, len(headers))
        if index != unit_index
        and sum(_numeric_value(row[index]) is not None for row in rows) >= 2
    ]
    if len(value_columns) != 1:
        return None

    value_index = value_columns[0]
    valid_rows = [
        (row, _numeric_value(row[value_index]))
        for row in rows
        if _numeric_value(row[value_index]) is not None
    ]
    if len(valid_rows) < 2:
        return None

    if unit_index is not None:
        units = {row[unit_index].strip().casefold() for row, _ in valid_rows}
        if len(units) != 1 or not next(iter(units)):
            return None
        unit = next(iter(units))
    else:
        unit_sources = [headers[value_index], *(row[value_index] for row, _ in valid_rows)]
        identities = {_unit_identity(value) for value in unit_sources}
        identities.discard(None)
        if len(identities) != 1:
            return None
        identity = next(iter(identities))
        unit = _unit_label(identity, " ".join(unit_sources))

    x_header = headers[0].strip()
    value_header = headers[value_index].strip()
    return {
        "title": f"{value_header} by {x_header}",
        "x_label": x_header,
        "y_label": unit,
        "labels": [row[0] for row, _ in valid_rows],
        "values": [number for _, number in valid_rows],
        "kind": (
            "line"
            if re.search(r"period|date|quarter|year", x_header, re.IGNORECASE)
            else "bar"
        ),
    }


def _safe_csv_cell(cell: str) -> str:
    if cell.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) and _numeric_value(cell) is None:
        return f"'{cell}"
    return cell
