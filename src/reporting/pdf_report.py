"""Fixed-layout PDF report generation for Asset Intelligence Workbench."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import unicodedata
import uuid

from fpdf import FPDF

from src.reporting.report_data import build_asset_report_context
from src.visuals import charts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "generated"
REPORT_TMP_DIR = REPORT_DIR / ".tmp"
MAX_REPORTS_PER_TICKER = 5

COLOR_PRIMARY = (16, 33, 49)
COLOR_MUTED = (98, 112, 126)
COLOR_BORDER = (219, 226, 232)
COLOR_FILL = (248, 250, 252)
COLOR_ACCENT = (41, 72, 95)
COMMENTARY_FONT_SIZES = (10.0, 9.5, 9.0, 8.5, 8.0)
BOX_PADDING_X = 5
BOX_PADDING_TOP = 5
BOX_PADDING_BOTTOM = 5
BOX_TITLE_GAP = 5
BOX_MIN_BODY_HEIGHT = 10

UNICODE_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2022": "-",
    }
)


def _format_percent(value: float) -> str:
    """Format a percentage value for PDF display."""

    return f"{value:.2%}"


def _format_number(value: float) -> str:
    """Format a numeric value for PDF display."""

    return f"{value:,.2f}"


def _pdf_safe_text(value: object) -> str:
    """
    Reduce dynamic report text to characters supported by core PDF fonts.

    The report uses Helvetica/Times core fonts, so smart quotes and other
    unicode punctuation must be normalized before drawing.
    """

    if value is None:
        return ""

    text = str(value).translate(UNICODE_PUNCT_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


def _cell(pdf: FPDF, w: float, h: float = 0, text: object = "", *args, **kwargs) -> None:
    """Draw a single-line PDF cell with sanitized text."""

    pdf.cell(w, h, _pdf_safe_text(text), *args, **kwargs)


def _multi_cell(
    pdf: FPDF,
    w: float,
    h: float,
    text: object = "",
    *args,
    **kwargs,
) -> None:
    """Draw a multi-line PDF cell with sanitized text."""

    pdf.multi_cell(w, h, _pdf_safe_text(text), *args, **kwargs)


def _ensure_report_context_compatibility(context: dict) -> dict:
    """
    Backfill report context fields that may be missing from a stale module load.

    Streamlit can sometimes retain an older in-memory import during hot reload.
    This helper keeps PDF generation resilient by deriving any required fields
    directly from the existing context payload.
    """

    if "drawdown_frame" not in context:
        price_frame = context["price_frame"][["analysis_price"]].copy()
        price_frame["running_peak"] = price_frame["analysis_price"].cummax()
        price_frame["drawdown"] = (
            price_frame["analysis_price"] / price_frame["running_peak"] - 1.0
        )
        context["drawdown_frame"] = price_frame

    if "terminal_percentiles" not in context:
        terminal_summary = context["simulation"]["terminal_summary"]
        context["terminal_percentiles"] = [
            {"label": "5th Percentile", "value": _format_number(terminal_summary["p05_terminal_price"])},
            {"label": "25th Percentile", "value": _format_number(terminal_summary["p25_terminal_price"])},
            {"label": "Median", "value": _format_number(terminal_summary["median_terminal_price"])},
            {"label": "75th Percentile", "value": _format_number(terminal_summary["p75_terminal_price"])},
            {"label": "95th Percentile", "value": _format_number(terminal_summary["p95_terminal_price"])},
        ]

    return context


class AssetBriefingPDF(FPDF):
    """Fixed-layout PDF with restrained institutional styling."""

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_draw_color(*COLOR_ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, 10, self.w - self.r_margin, 10)
        self.set_y(12)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*COLOR_MUTED)
        _cell(self, 0, 4, "Asset Intelligence Workbench", 0, 0, "L")
        _cell(self, 0, 4, "Asset Briefing", 0, 1, "R")
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-11)
        self.set_draw_color(*COLOR_BORDER)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y() - 1, self.w - self.r_margin, self.get_y() - 1)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*COLOR_MUTED)
        _cell(self, 0, 5, f"Page {self.page_no()}", 0, 0, "C")


def _set_text_style(
    pdf: FPDF,
    family: str = "Helvetica",
    style: str = "",
    size: int = 11,
    color: tuple[int, int, int] = COLOR_PRIMARY,
) -> None:
    """Apply font and color styling consistently."""

    pdf.set_font(family, style, size)
    pdf.set_text_color(*color)


def _draw_labeled_box(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    note: str | None = None,
    fill_color: tuple[int, int, int] = COLOR_FILL,
) -> None:
    """Draw a restrained KPI tile."""

    pdf.set_fill_color(*fill_color)
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_xy(x + 3.5, y + 3.2)
    _set_text_style(pdf, size=7, color=COLOR_MUTED)
    _cell(pdf, w - 7, 3.2, label.upper(), ln=1)
    pdf.set_x(x + 3.5)
    _set_text_style(pdf, style="B", size=13)
    _cell(pdf, w - 7, 6, value, ln=1)
    if note:
        pdf.set_x(x + 3.5)
        _set_text_style(pdf, size=8, color=COLOR_MUTED)
        _multi_cell(pdf, w - 7, 3.2, note)


def _wrap_text_to_width(pdf: FPDF, text: str, width: float) -> list[str]:
    """Wrap text to a target width using the active PDF font metrics."""

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line = words[0]

    for word in words[1:]:
        candidate = f"{current_line} {word}"
        if pdf.get_string_width(candidate) <= width:
            current_line = candidate
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines


def _fit_commentary_text(
    pdf: FPDF,
    text: str,
    width: float,
    height: float,
) -> tuple[float, float, str, float]:
    """Fit commentary text inside a fixed box by stepping down font size if needed."""

    cleaned_text = _pdf_safe_text(" ".join(str(text).split()))
    for font_size in COMMENTARY_FONT_SIZES:
        line_height = max(3.8, round(font_size * 0.48, 1))
        _set_text_style(pdf, family="Times", size=font_size, color=COLOR_PRIMARY)
        wrapped_lines = _wrap_text_to_width(pdf, cleaned_text, width)
        required_height = len(wrapped_lines) * line_height
        if required_height <= height:
            return font_size, line_height, "\n".join(wrapped_lines), required_height

    smallest = COMMENTARY_FONT_SIZES[-1]
    _set_text_style(pdf, family="Times", size=smallest, color=COLOR_PRIMARY)
    line_height = max(3.6, round(smallest * 0.46, 1))
    wrapped_lines = _wrap_text_to_width(pdf, cleaned_text, width)
    required_height = len(wrapped_lines) * line_height
    return smallest, line_height, "\n".join(wrapped_lines), required_height


def _plan_text_box(
    pdf: FPDF,
    title: str,
    text: str,
    width: float,
    preferred_height: float,
) -> dict[str, float | str]:
    """
    Plan a boxed text paragraph before drawing it.

    This is the permanent overflow-prevention rule for narrative boxes:
    1. Wrap to the usable width.
    2. Measure rendered height.
    3. Reduce font size within a readable floor if needed.
    4. If the text still does not fit, expand the box height instead of clipping.
    """

    body_width = width - (BOX_PADDING_X * 2)
    title_height = 4
    body_available_height = max(
        BOX_MIN_BODY_HEIGHT,
        preferred_height - (BOX_PADDING_TOP + BOX_PADDING_BOTTOM + title_height + BOX_TITLE_GAP),
    )
    font_size, line_height, fitted_text, required_body_height = _fit_commentary_text(
        pdf=pdf,
        text=text,
        width=body_width,
        height=body_available_height,
    )
    final_body_height = max(body_available_height, required_body_height)
    final_height = (
        BOX_PADDING_TOP
        + title_height
        + BOX_TITLE_GAP
        + final_body_height
        + BOX_PADDING_BOTTOM
    )
    return {
        "title": title,
        "text": fitted_text,
        "font_size": font_size,
        "line_height": line_height,
        "body_width": body_width,
        "body_height": final_body_height,
        "height": final_height,
    }


def _draw_fitted_text_box(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    preferred_h: float,
    title: str,
    text: str,
) -> float:
    """
    Draw a boxed narrative block only after confirming the text fits.

    Returns the actual rendered box height, which may be larger than the preferred
    height when the text requires more space.
    """

    plan = _plan_text_box(
        pdf=pdf,
        title=title,
        text=text,
        width=w,
        preferred_height=preferred_h,
    )

    actual_h = float(plan["height"])
    pdf.set_fill_color(251, 252, 253)
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.rect(x, y, w, actual_h, style="DF")

    pdf.set_xy(x + BOX_PADDING_X, y + BOX_PADDING_TOP)
    _set_text_style(pdf, size=8, color=COLOR_MUTED)
    _cell(pdf, w - (BOX_PADDING_X * 2), 4, title.upper(), ln=1)

    pdf.set_xy(x + BOX_PADDING_X, y + BOX_PADDING_TOP + 4 + BOX_TITLE_GAP)
    _set_text_style(pdf, family="Times", size=float(plan["font_size"]), color=COLOR_PRIMARY)
    _multi_cell(pdf, float(plan["body_width"]), float(plan["line_height"]), str(plan["text"]))
    return actual_h


def _draw_section_header(pdf: FPDF, section_id: str, kicker: str, title: str) -> None:
    """Draw a memo-style section header."""

    y = pdf.get_y()
    _set_text_style(pdf, size=12, color=COLOR_MUTED)
    pdf.set_xy(pdf.l_margin, y)
    _cell(pdf, 12, 6, section_id, 0, 0)
    pdf.set_xy(pdf.l_margin + 14, y)
    _set_text_style(pdf, size=8, color=COLOR_MUTED)
    _cell(pdf, 0, 3.5, kicker.upper(), ln=1)
    pdf.set_x(pdf.l_margin + 14)
    _set_text_style(pdf, style="B", size=18, color=COLOR_PRIMARY)
    _cell(pdf, 0, 7, title, ln=1)
    pdf.ln(2)
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)


def _draw_key_value_table(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    rows: list[tuple[str, str]],
    title: str | None = None,
) -> float:
    """Draw a compact key-value table and return consumed height."""

    current_y = y
    if title:
        _set_text_style(pdf, size=8, color=COLOR_MUTED)
        pdf.set_xy(x, current_y)
        _cell(pdf, w, 4, title.upper(), ln=1)
        current_y += 5

    row_height = 7
    for label, value in rows:
        pdf.set_draw_color(*COLOR_BORDER)
        pdf.rect(x, current_y, w, row_height)
        _set_text_style(pdf, size=8, color=COLOR_MUTED)
        pdf.set_xy(x + 3, current_y + 1.8)
        _cell(pdf, w * 0.48, 3, label.upper(), 0, 0)
        _set_text_style(pdf, style="B", size=10, color=COLOR_PRIMARY)
        pdf.set_xy(x + w * 0.5, current_y + 1.5)
        _cell(pdf, w * 0.46, 3.5, value, 0, 0, "R")
        current_y += row_height
    return current_y - y


def _draw_recent_price_table(pdf: FPDF, x: float, y: float, rows: list[dict[str, str]]) -> None:
    """Draw the recent price table with formal numeric alignment."""

    headers = ["Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]
    widths = [24, 21, 21, 21, 21, 24, 36]

    pdf.set_xy(x, y)
    _set_text_style(pdf, size=8, color=COLOR_MUTED)
    _cell(pdf, sum(widths), 4, "RECENT MARKET SNAPSHOT", ln=1)
    pdf.set_xy(x, y + 5)
    _set_text_style(pdf, style="B", size=7, color=COLOR_PRIMARY)
    pdf.set_fill_color(243, 246, 248)
    pdf.set_draw_color(*COLOR_BORDER)
    for header, width in zip(headers, widths):
        align = "L" if header == "Date" else "R"
        _cell(pdf, width, 6.5, header.upper(), border=1, fill=True, align=align)
    pdf.ln()

    _set_text_style(pdf, size=7, color=COLOR_PRIMARY)
    for row in rows:
        pdf.set_x(x)
        _cell(pdf, widths[0], 6, row["price_date"], border=1)
        _cell(pdf, widths[1], 6, row["open_price"], border=1, align="R")
        _cell(pdf, widths[2], 6, row["high_price"], border=1, align="R")
        _cell(pdf, widths[3], 6, row["low_price"], border=1, align="R")
        _cell(pdf, widths[4], 6, row["close_price"], border=1, align="R")
        _cell(pdf, widths[5], 6, row["adjusted_close"], border=1, align="R")
        _cell(pdf, widths[6], 6, row["volume"], border=1, align="R")
        pdf.ln()


def _draw_recent_headlines_table(pdf: FPDF, x: float, y: float, rows: list[dict[str, str]]) -> None:
    """Draw a compact recent-headlines block for the sentiment page."""

    headers = ["Published", "Source", "Headline", "Tone", "Score"]
    widths = [25, 24, 86, 18, 23]

    pdf.set_xy(x, y)
    _set_text_style(pdf, size=8, color=COLOR_MUTED)
    _cell(pdf, sum(widths), 4, "RECENT HEADLINES", ln=1)
    pdf.set_xy(x, y + 5)
    _set_text_style(pdf, style="B", size=7, color=COLOR_PRIMARY)
    pdf.set_fill_color(243, 246, 248)
    pdf.set_draw_color(*COLOR_BORDER)
    for header, width in zip(headers, widths):
        align = "L" if header in {"Published", "Source", "Headline"} else "R"
        _cell(pdf, width, 6, header.upper(), border=1, fill=True, align=align)
    pdf.ln()

    _set_text_style(pdf, size=7, color=COLOR_PRIMARY)
    for row in rows:
        pdf.set_x(x)
        headline = row["headline"]
        if len(headline) > 72:
            headline = f"{headline[:69].rstrip()}..."
        _cell(pdf, widths[0], 6, row["published_at"], border=1)
        _cell(pdf, widths[1], 6, row["publisher_name"], border=1)
        _cell(pdf, widths[2], 6, headline, border=1)
        _cell(pdf, widths[3], 6, row["sentiment_label"], border=1, align="R")
        _cell(pdf, widths[4], 6, row["sentiment_score"], border=1, align="R")
        pdf.ln()


def _draw_terminal_percentile_table(pdf: FPDF, x: float, y: float, rows: list[dict[str, str]]) -> None:
    """Draw a compact percentile summary table for the simulation page."""

    pdf.set_xy(x, y)
    _set_text_style(pdf, size=7, color=COLOR_MUTED)
    _cell(pdf, 60, 3.5, "TERMINAL PERCENTILE SUMMARY", ln=1)
    pdf.set_xy(x, y + 4.5)
    for row in rows:
        pdf.set_draw_color(*COLOR_BORDER)
        pdf.rect(x, pdf.get_y(), 60, 5.8)
        _set_text_style(pdf, size=7.5, color=COLOR_MUTED)
        pdf.set_x(x + 3)
        _cell(pdf, 30, 3.2, row["label"].upper(), 0, 0)
        _set_text_style(pdf, style="B", size=9, color=COLOR_PRIMARY)
        pdf.set_x(x + 33)
        _cell(pdf, 24, 3.2, row["value"], 0, 1, "R")


def _draw_note_strip(pdf: FPDF, x: float, y: float, w: float, h: float, text: str) -> None:
    """Draw a slim muted note strip to anchor page composition."""

    pdf.set_fill_color(250, 252, 253)
    pdf.set_draw_color(*COLOR_BORDER)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_xy(x + 4, y + 3)
    _set_text_style(pdf, size=7.5, color=COLOR_MUTED)
    _multi_cell(pdf, w - 8, 3.6, text)


def _render_cover_page(pdf: FPDF, context: dict) -> None:
    """Render the redesigned cover and executive summary page."""

    pdf.add_page()
    pdf.set_draw_color(*COLOR_ACCENT)
    pdf.set_line_width(1.1)
    pdf.line(pdf.l_margin, 18, pdf.w - pdf.r_margin, 18)

    pdf.set_xy(pdf.l_margin, 24)
    _set_text_style(pdf, size=8, color=COLOR_MUTED)
    _cell(pdf, 0, 4, "ASSET INTELLIGENCE WORKBENCH", ln=1)
    _set_text_style(pdf, size=10, color=COLOR_MUTED)
    _cell(pdf, 0, 5, "Internal Asset / Risk Briefing", ln=1)

    pdf.ln(14)
    _set_text_style(pdf, size=9, color=COLOR_MUTED)
    _cell(pdf, 0, 5, "ASSET BRIEFING", ln=1)
    _set_text_style(pdf, style="B", size=28, color=COLOR_PRIMARY)
    _cell(pdf, 0, 12, context["metadata"]["asset_name"], ln=1)
    _set_text_style(pdf, size=12, color=COLOR_ACCENT)
    _cell(pdf, 0, 6, context["ticker"], ln=1)
    pdf.ln(6)
    _set_text_style(pdf, family="Times", size=11, color=COLOR_MUTED)
    _multi_cell(
        pdf,
        125,
        5.5,
        "Historical market performance, downside risk, and forward scenario analysis prepared from the local finance analytics workbench.",
    )

    meta_y = 88
    meta_w = (pdf.w - pdf.l_margin - pdf.r_margin - 9) / 4
    meta_items = [
        ("Generated", context["report_generated_at"].strftime("%Y-%m-%d %H:%M")),
        (
            "Coverage",
            f"{context['coverage']['start_date'].strftime('%Y-%m-%d')} to {context['coverage']['end_date'].strftime('%Y-%m-%d')}",
        ),
        ("Source", context["metadata"].get("primary_source") or "N/A"),
        ("Records", str(context["coverage"]["observation_count"])),
    ]
    for idx, (label, value) in enumerate(meta_items):
        x = pdf.l_margin + idx * (meta_w + 3)
        pdf.set_fill_color(250, 252, 253)
        pdf.set_draw_color(*COLOR_BORDER)
        pdf.rect(x, meta_y, meta_w, 15, style="DF")
        pdf.set_xy(x + 3, meta_y + 3)
        _set_text_style(pdf, size=7, color=COLOR_MUTED)
        _cell(pdf, meta_w - 6, 3, label.upper(), ln=1)
        pdf.set_x(x + 3)
        _set_text_style(pdf, size=9, color=COLOR_PRIMARY)
        _multi_cell(pdf, meta_w - 6, 3.8, value)

    grid_y = 111
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    card_w = (usable_w - 8) / 3
    card_h = 21
    grid_items = [
        ("Latest Price", _format_number(context["coverage"]["latest_price"]), context["metadata"].get("currency") or "N/A"),
        ("Observation Count", str(context["coverage"]["observation_count"]), "stored daily records"),
        ("Total Return", _format_percent(context["kpis"]["total_return"]), "holding-period outcome"),
        ("Annualized Return", _format_percent(context["kpis"]["annualized_return"]), "geometric equivalent"),
        ("Annualized Volatility", _format_percent(context["kpis"]["annualized_volatility"]), "252-day basis"),
        ("Max Drawdown", _format_percent(context["kpis"]["max_drawdown"]), "peak-to-trough decline"),
    ]
    for idx, (label, value, note) in enumerate(grid_items):
        row = idx // 3
        col = idx % 3
        x = pdf.l_margin + col * (card_w + 4)
        y = grid_y + row * (card_h + 5)
        fill = (244, 248, 251) if idx == 0 else COLOR_FILL
        _draw_labeled_box(pdf, x, y, card_w, card_h, label, value, note=note, fill_color=fill)

    _draw_fitted_text_box(
        pdf,
        pdf.l_margin,
        164,
        usable_w,
        70,
        "Executive Summary",
        context["narrative"]["executive_summary"],
    )


def _render_market_snapshot_page(pdf: FPDF, context: dict) -> None:
    """Render the formal market snapshot page."""

    pdf.add_page()
    _draw_section_header(pdf, "01", "Market Snapshot", "Asset Profile")

    left_rows = [
        ("Asset Name", context["metadata"]["asset_name"]),
        ("Ticker", context["metadata"]["ticker"]),
        ("Asset Class", context["metadata"].get("asset_class") or "N/A"),
        ("Industry", context["metadata"].get("industry") or "N/A"),
    ]
    right_rows = [
        ("Exchange", context["metadata"].get("exchange") or "N/A"),
        ("Currency", context["metadata"].get("currency") or "N/A"),
        ("Country", context["metadata"].get("country") or "N/A"),
        ("Primary Source", context["metadata"].get("primary_source") or "N/A"),
    ]
    start_y = pdf.get_y()
    left_h = _draw_key_value_table(pdf, pdf.l_margin, start_y, 84, left_rows)
    right_h = _draw_key_value_table(pdf, pdf.l_margin + 92, start_y, 84, right_rows)
    pdf.set_y(start_y + max(left_h, right_h) + 8)

    coverage_cards = [
        ("Coverage Start", context["coverage"]["start_date"].strftime("%Y-%m-%d")),
        ("Coverage End", context["coverage"]["end_date"].strftime("%Y-%m-%d")),
        ("Stored Records", str(context["coverage"]["observation_count"])),
    ]
    card_w = (pdf.w - pdf.l_margin - pdf.r_margin - 8) / 3
    y = pdf.get_y()
    for idx, (label, value) in enumerate(coverage_cards):
        x = pdf.l_margin + idx * (card_w + 4)
        _draw_labeled_box(pdf, x, y, card_w, 18, label, value)
    pdf.set_y(y + 24)

    _draw_recent_price_table(pdf, pdf.l_margin, pdf.get_y(), context["recent_prices"])


def _render_performance_page(pdf: FPDF, context: dict, price_chart: str, cumulative_chart: str) -> None:
    """Render the rebuilt performance page with larger charts and calmer text layout."""

    pdf.add_page()
    _draw_section_header(pdf, "02", "Historical Performance", "Performance Analysis")

    top_y = pdf.get_y()
    stat_w = (pdf.w - pdf.l_margin - pdf.r_margin - 8) / 3
    stat_items = [
        ("Latest Price", _format_number(context["coverage"]["latest_price"])),
        ("Total Return", _format_percent(context["kpis"]["total_return"])),
        ("Annualized Return", _format_percent(context["kpis"]["annualized_return"])),
    ]
    for idx, (label, value) in enumerate(stat_items):
        _draw_labeled_box(
            pdf,
            pdf.l_margin + idx * (stat_w + 4),
            top_y,
            stat_w,
            15,
            label,
            value,
        )

    lead_chart_y = top_y + 22
    pdf.image(price_chart, x=pdf.l_margin, y=lead_chart_y, w=176)

    lower_y = 149
    pdf.image(cumulative_chart, x=pdf.l_margin, y=lower_y, w=120)
    _draw_fitted_text_box(
        pdf,
        pdf.l_margin + 124,
        lower_y,
        52,
        57,
        "Performance Commentary",
        context["narrative"]["performance_commentary"],
    )
    _draw_note_strip(
        pdf,
        pdf.l_margin,
        214,
        176,
        16,
        "Price history anchors the path review, while cumulative return isolates compounded performance through time. The page is intentionally weighted toward the exhibits rather than long narrative text.",
    )


def _render_risk_page(pdf: FPDF, context: dict, rolling_vol_chart: str, drawdown_chart: str) -> None:
    """Render the risk page without repeating the price chart."""

    pdf.add_page()
    _draw_section_header(pdf, "03", "Historical Downside Review", "Risk Analysis")

    top_y = pdf.get_y()
    _draw_labeled_box(
        pdf,
        pdf.l_margin,
        top_y,
        42,
        18,
        "Annualized Volatility",
        _format_percent(context["kpis"]["annualized_volatility"]),
    )
    _draw_labeled_box(
        pdf,
        pdf.l_margin + 45,
        top_y,
        42,
        18,
        "Max Drawdown",
        _format_percent(context["kpis"]["max_drawdown"]),
        fill_color=(252, 247, 246),
    )
    _draw_labeled_box(
        pdf,
        pdf.l_margin + 90,
        top_y,
        42,
        18,
        "Historical VaR",
        _format_percent(context["kpis"]["historical_var"]),
    )
    _draw_labeled_box(
        pdf,
        pdf.l_margin + 135,
        top_y,
        42,
        18,
        "Expected Shortfall",
        _format_percent(context["kpis"]["expected_shortfall"]),
    )
    pdf.image(rolling_vol_chart, x=pdf.l_margin, y=top_y + 24, w=88)
    pdf.image(drawdown_chart, x=pdf.l_margin + 88, y=top_y + 24, w=88)
    _draw_fitted_text_box(
        pdf,
        pdf.l_margin,
        top_y + 104,
        176,
        28,
        "Risk Commentary",
        context["narrative"]["risk_commentary"],
    )


def _render_simulation_page(pdf: FPDF, context: dict, paths_chart: str, terminal_chart: str) -> None:
    """Render the rebuilt simulation page with a dominant scenario chart."""

    pdf.add_page()
    _draw_section_header(pdf, "04", "Forward Scenario Analysis", "Simulation Outlook")

    y = pdf.get_y()
    band_w = (pdf.w - pdf.l_margin - pdf.r_margin - 12) / 4
    band_items = [
        ("Forecast Horizon", f"{context['settings']['forecast_horizon']} trading days"),
        ("Simulation Count", f"{context['settings']['simulation_count']:,}"),
        ("Estimated Daily Drift", _format_percent(context["simulation"]["inputs"]["daily_drift"])),
        ("Ann. Volatility", _format_percent(context["simulation"]["inputs"]["annualized_volatility"])),
    ]
    for idx, (label, value) in enumerate(band_items):
        x = pdf.l_margin + idx * (band_w + 4)
        _draw_labeled_box(pdf, x, y, band_w, 15, label, value)

    monte_carlo_y = y + 20
    pdf.image(paths_chart, x=pdf.l_margin, y=monte_carlo_y, w=176)

    lower_y = 154
    pdf.image(terminal_chart, x=pdf.l_margin, y=lower_y, w=112)

    metric_w = 29
    metrics_y = lower_y
    sim_metrics = [
        ("Median Terminal Price", _format_number(context["simulation"]["terminal_summary"]["median_terminal_price"])),
        ("Probability Above Start", _format_percent(context["simulation"]["terminal_summary"]["probability_above_start"])),
        ("5th Percentile", _format_number(context["simulation"]["terminal_summary"]["p05_terminal_price"])),
        ("95th Percentile", _format_number(context["simulation"]["terminal_summary"]["p95_terminal_price"])),
    ]
    for idx, (label, value) in enumerate(sim_metrics):
        row = idx // 2
        col = idx % 2
        x = pdf.l_margin + 116 + col * (metric_w + 2)
        yy = metrics_y + row * 18
        _draw_labeled_box(pdf, x, yy, metric_w, 15, label, value)

    _draw_terminal_percentile_table(
        pdf,
        pdf.l_margin + 116,
        lower_y + 40,
        context["terminal_percentiles"],
    )
    _draw_fitted_text_box(
        pdf,
        pdf.l_margin,
        231,
        176,
        26,
        "Scenario Commentary",
        context["narrative"]["simulation_commentary"],
    )


def _render_sentiment_page(pdf: FPDF, context: dict, sentiment_chart: str | None) -> None:
    """Render a restrained sentiment context page for the report."""

    pdf.add_page()
    _draw_section_header(pdf, "05", "Stored News Context", "Sentiment Review")

    summary = context["sentiment"]["summary"]
    top_y = pdf.get_y()
    metric_w = (pdf.w - pdf.l_margin - pdf.r_margin - 12) / 4
    metric_items = [
        ("Article Count", str(summary["article_count"])),
        ("Avg. Sentiment", _format_number(summary["average_sentiment"] or 0.0) if summary["average_sentiment"] is not None else "N/A"),
        ("Positive / Neutral", f"{summary['positive_count']} / {summary['neutral_count']}"),
        ("Negative", str(summary["negative_count"])),
    ]
    for idx, (label, value) in enumerate(metric_items):
        x = pdf.l_margin + idx * (metric_w + 4)
        _draw_labeled_box(pdf, x, top_y, metric_w, 16, label, value)

    if sentiment_chart is not None:
        chart_y = top_y + 22
        pdf.image(sentiment_chart, x=pdf.l_margin, y=chart_y, w=116)
        commentary_height = _draw_fitted_text_box(
            pdf,
            pdf.l_margin + 120,
            chart_y,
            56,
            54,
            "Sentiment Commentary",
            context["narrative"]["sentiment_commentary"],
        )
        headlines_y = max(chart_y + 84, chart_y + commentary_height + 4)
    else:
        commentary_height = _draw_fitted_text_box(
            pdf,
            pdf.l_margin,
            top_y + 24,
            176,
            32,
            "Sentiment Commentary",
            context["narrative"]["sentiment_commentary"],
        )
        headlines_y = top_y + 24 + commentary_height + 8

    headlines = context["sentiment"]["recent_headlines"]
    if headlines:
        _draw_recent_headlines_table(pdf, pdf.l_margin, headlines_y, headlines)
    else:
        _draw_note_strip(
            pdf,
            pdf.l_margin,
            headlines_y,
            176,
            16,
            "No stored headlines were available for the selected asset during report generation.",
        )


def _render_methodology_page(pdf: FPDF, context: dict) -> None:
    """Render a cleaner, scannable methodology page."""

    pdf.add_page()
    _draw_section_header(pdf, "06", "Definitions & Notes", "Methodology")
    y = pdf.get_y()
    items = [
        ("Data Source Notes", context["methodology"]["data_source_note"]),
        ("Methodology Notes", context["methodology"]["analytics_note"]),
        ("Simulation Assumptions", context["methodology"]["simulation_note"]),
        ("Caveats", context["methodology"]["caveat_note"]),
    ]
    box_w = (pdf.w - pdf.l_margin - pdf.r_margin - 8) / 2
    preferred_box_h = 48

    # Methodology boxes are measured before drawing so each row gets enough
    # height for the taller card. This prevents overflow and vertical collisions.
    planned_boxes: list[dict[str, float | str]] = []
    for title, text in items:
        planned_boxes.append(
            _plan_text_box(
                pdf=pdf,
                title=title,
                text=text,
                width=box_w,
                preferred_height=preferred_box_h,
            )
        )

    first_row_height = max(float(planned_boxes[0]["height"]), float(planned_boxes[1]["height"]))
    second_row_y = y + first_row_height + 8

    positions = [
        (pdf.l_margin, y),
        (pdf.l_margin + box_w + 8, y),
        (pdf.l_margin, second_row_y),
        (pdf.l_margin + box_w + 8, second_row_y),
    ]

    for (title, text), (x, yy) in zip(items, positions):
        _draw_fitted_text_box(pdf, x, yy, box_w, preferred_box_h, title, text)


def generate_asset_pdf_report(
    ticker: str,
    forecast_horizon: int = 63,
    simulation_count: int = 500,
) -> str:
    """
    Generate a fixed-layout multi-page asset briefing PDF.

    Returns the absolute saved PDF path as a string.
    """

    context = build_asset_report_context(
        ticker=ticker,
        forecast_horizon=forecast_horizon,
        simulation_count=simulation_count,
    )
    context = _ensure_report_context_compatibility(context)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    base_name = f"{context['ticker']}_asset_briefing_{timestamp}_{unique_suffix}"
    pdf_path = REPORT_DIR / f"{base_name}.pdf"
    html_path = REPORT_DIR / f"{base_name}.html"
    html_path.write_text(
        (
            "<html><body><p>"
            "This run uses the fixed-layout PDF renderer. "
            "The HTML companion is retained only as a lightweight generation artifact."
            "</p></body></html>"
        ),
        encoding="utf-8",
    )

    tmp = REPORT_TMP_DIR / f"{base_name}_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        price_chart = charts.save_price_history_chart(
            context["price_frame"],
            tmp / "price_history.png",
        )
        cumulative_chart = charts.save_cumulative_return_chart(
            context["return_frame"].dropna(subset=["cumulative_return"]),
            tmp / "cumulative_return.png",
        )
        rolling_vol_chart = charts.save_rolling_volatility_chart(
            context["rolling_volatility"],
            tmp / "rolling_volatility.png",
        )
        drawdown_chart = charts.save_drawdown_chart(
            context["drawdown_frame"],
            tmp / "drawdown.png",
        )
        sentiment_chart = None
        if not context["sentiment"]["trend"].empty and context["sentiment"]["trend"].shape[0] >= 2:
            sentiment_chart = charts.save_sentiment_trend_chart(
                context["sentiment"]["trend"],
                tmp / "sentiment_trend.png",
            )
        paths_chart = charts.save_monte_carlo_paths_chart(
            context["simulation"]["paths"],
            tmp / "monte_carlo_paths.png",
        )
        terminal_chart = charts.save_terminal_distribution_chart(
            context["simulation"]["paths"],
            tmp / "terminal_distribution.png",
        )

        pdf = AssetBriefingPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()

        _render_cover_page(pdf, context)
        _render_market_snapshot_page(pdf, context)
        _render_performance_page(pdf, context, price_chart, cumulative_chart)
        _render_risk_page(pdf, context, rolling_vol_chart, drawdown_chart)
        _render_simulation_page(pdf, context, paths_chart, terminal_chart)
        _render_sentiment_page(pdf, context, sentiment_chart)
        _render_methodology_page(pdf, context)

        pdf.output(str(pdf_path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    _cleanup_old_reports_for_ticker(
        context["ticker"],
        keep=MAX_REPORTS_PER_TICKER,
        current_pdf=pdf_path,
    )

    return str(pdf_path.resolve())


def open_report_locally(report_path: str | Path) -> dict[str, str]:
    """Open a generated report with the system default PDF application."""

    resolved_path = Path(report_path).resolve()
    if not resolved_path.exists():
        return {
            "success": "false",
            "message": f"The report file does not exist: {resolved_path}",
        }

    if sys.platform != "win32":
        return {
            "success": "false",
            "message": "Local open is only available in this workflow on Windows.",
        }

    try:
        os.startfile(str(resolved_path))
    except OSError as exc:
        return {
            "success": "false",
            "message": f"Windows could not open the report with the default application. Detail: {exc}",
        }

    return {
        "success": "true",
        "message": f"Opened report locally: {resolved_path}",
    }


def _cleanup_old_reports_for_ticker(
    ticker: str,
    keep: int,
    current_pdf: Path,
) -> None:
    """Conservatively retain only the newest report sets for a ticker."""

    matching_pdfs = sorted(
        REPORT_DIR.glob(f"{ticker}_asset_briefing_*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if len(matching_pdfs) <= keep:
        return

    for old_pdf in matching_pdfs[keep:]:
        if old_pdf.resolve() == current_pdf.resolve():
            continue

        old_html = old_pdf.with_suffix(".html")
        try:
            old_pdf.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            old_html.unlink(missing_ok=True)
        except OSError:
            pass
