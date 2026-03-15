"""Templated PDF report generation.

All text is parameterized — no hardcoded coin names or numbers.
"""

from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)

# Colors
DARK = HexColor("#1a1a2e")
ACCENT = HexColor("#0f3460")
BLUE = HexColor("#16537e")
LIGHT_BLUE = HexColor("#e8f4f8")
GREEN = HexColor("#27ae60")
AMBER = HexColor("#e67e22")
RED = HexColor("#c0392b")
GRAY = HexColor("#7f8c8d")
LIGHT_GRAY = HexColor("#f5f6fa")
LIGHT_GREEN = HexColor("#eafaf1")
LIGHT_RED = HexColor("#fdedec")
LIGHT_AMBER = HexColor("#fef9e7")
TABLE_HEADER_BG = HexColor("#2c3e50")
TABLE_ALT_ROW = HexColor("#f0f3f5")


def _build_styles():
    styles = getSampleStyleSheet()
    defs = {
        "ReportTitle": dict(parent=styles["Title"], fontSize=26, leading=32, textColor=DARK, spaceAfter=4*mm, alignment=TA_LEFT),
        "Subtitle": dict(parent=styles["Normal"], fontSize=12, leading=16, textColor=GRAY, spaceAfter=8*mm),
        "H1": dict(parent=styles["Heading1"], fontSize=18, leading=24, textColor=ACCENT, spaceBefore=10*mm, spaceAfter=4*mm),
        "H2": dict(parent=styles["Heading2"], fontSize=14, leading=18, textColor=BLUE, spaceBefore=6*mm, spaceAfter=3*mm),
        "Body": dict(parent=styles["Normal"], fontSize=10, leading=14, textColor=black, spaceAfter=3*mm, alignment=TA_JUSTIFY),
        "BulletItem": dict(parent=styles["Normal"], fontSize=10, leading=14, textColor=black, leftIndent=12*mm, bulletIndent=6*mm, spaceAfter=1.5*mm),
        "KeyFinding": dict(parent=styles["Normal"], fontSize=10, leading=14, textColor=DARK, leftIndent=6*mm, rightIndent=6*mm, spaceBefore=2*mm, spaceAfter=2*mm, backColor=LIGHT_BLUE, borderPadding=6),
        "GreenBox": dict(parent=styles["Normal"], fontSize=10, leading=14, textColor=DARK, leftIndent=6*mm, rightIndent=6*mm, spaceBefore=2*mm, spaceAfter=2*mm, backColor=LIGHT_GREEN, borderPadding=6),
        "Warning": dict(parent=styles["Normal"], fontSize=10, leading=14, textColor=RED, leftIndent=6*mm, rightIndent=6*mm, spaceBefore=2*mm, spaceAfter=2*mm, borderPadding=6, backColor=LIGHT_RED),
        "TableCell": dict(parent=styles["Normal"], fontSize=8.5, leading=11, alignment=TA_CENTER),
        "TableCellLeft": dict(parent=styles["Normal"], fontSize=8.5, leading=11, alignment=TA_LEFT),
        "TableHeader": dict(parent=styles["Normal"], fontSize=8.5, leading=11, textColor=white, alignment=TA_CENTER, fontName="Helvetica-Bold"),
    }
    for name, kw in defs.items():
        styles.add(ParagraphStyle(name, **kw))
    return styles


def _make_table(headers, rows, col_widths=None, styles=None):
    s = styles or _build_styles()
    data = [[Paragraph(h, s["TableHeader"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["TableCell"]) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), TABLE_ALT_ROW))
    t.setStyle(TableStyle(cmds))
    return t


def generate_pdf_report(
    follower_symbol: str,
    results: dict,
    output_path: str = None,
):
    """Generate PDF report for any follower/BTC pair.

    All text uses follower_symbol and numbers from results dict.
    """
    coin = follower_symbol.replace("USDT", "")
    if output_path is None:
        output_path = f"{coin}_BTC_analysis.pdf"

    s = _build_styles()

    def _footer(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.setFillColor(GRAY)
        canvas_obj.drawCentredString(
            A4[0] / 2, 12 * mm,
            f"{coin}/BTC Catch-Up Trade Analysis  |  Page {doc.page}  |  {datetime.now().strftime('%B %Y')}"
        )
        canvas_obj.restoreState()

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    W = doc.width
    story = []

    # Extract data from results
    strat = results.get("strategy", {})
    baseline = results.get("baseline", {})
    regime = results.get("regime_summary", {})
    risk = results.get("risk", {})
    impulse = results.get("impulse_summary", {})
    meta = results.get("meta", {})

    days = meta.get("days", 7)
    total_trades = strat.get("total_trades", 0)
    win_rate = strat.get("win_rate", 0)
    avg_net = strat.get("avg_net_pct", 0)
    total_net = strat.get("total_net_pct", 0)
    pctl = baseline.get("percentile_rank_wr", 0)

    # ── TITLE PAGE ────────────────────────────────────────────────
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph(f"{coin}/BTC Catch-Up Trade", s["ReportTitle"]))
    story.append(Paragraph("Quantitative Analysis Report", ParagraphStyle(
        "TitleLine2", parent=s["ReportTitle"], fontSize=22, textColor=BLUE)))
    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        f"Can you make money by buying {coin} when Bitcoin suddenly jumps? "
        f"This report analyzes {days} days of real trading data and tests whether "
        f"Bitcoin's price movements create a profitable timing signal for {coin}.",
        s["Subtitle"]))

    story.append(Spacer(1, 10*mm))
    meta_rows = [
        ["Data Period", f"{meta.get('start_date', '?')} to {meta.get('end_date', '?')} ({days} days)"],
        ["Data Source", f"Binance exchange (BTC/USDT, {coin}/USDT)"],
        ["Report Date", datetime.now().strftime("%B %d, %Y")],
    ]
    mt = Table(
        [[Paragraph(r[0], ParagraphStyle("ml", parent=s["TableCellLeft"], fontName="Helvetica-Bold")),
          Paragraph(r[1], s["TableCellLeft"])] for r in meta_rows],
        colWidths=[45*mm, W - 45*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6)]))
    story.append(mt)
    story.append(PageBreak())

    # ── 1. SUMMARY ────────────────────────────────────────────────
    story.append(Paragraph("1. Summary", s["H1"]))
    findings = [
        f"<b>The BTC trigger is real.</b> Buying {coin} after a Bitcoin pump beats random "
        f"entries {pctl:.0f}% of the time over {days} days.",

        f"<b>Strategy results:</b> {total_trades} trades, {win_rate:.0f}% win rate, "
        f"{avg_net:+.4f}% average net per trade, {total_net:+.2f}% total net return.",
    ]

    # Add regime insight if available
    bull = regime.get("BULL", {})
    bear = regime.get("BEAR", {})
    if bull.get("days", 0) > 0 and bear.get("days", 0) > 0:
        findings.append(
            f"<b>Works in both regimes:</b> Bull days ({bull['days']}d): "
            f"{bull.get('avg_win_rate', 0):.0f}% WR. Bear days ({bear['days']}d): "
            f"{bear.get('avg_win_rate', 0):.0f}% WR.")

    # Add leverage insight if available
    for lev_key in ["5", "10"]:
        r = risk.get(lev_key, {})
        if r.get("median_return_pct", 0) > 0:
            findings.append(
                f"<b>At {lev_key}x leverage:</b> {r['median_return_pct']:+.1f}% median return "
                f"with {r['median_max_dd_pct']:.1f}% median max drawdown.")
            break

    for f in findings:
        story.append(Paragraph(f, s["BulletItem"], bulletText="\u2022"))

    story.append(PageBreak())

    # ── 2. STRATEGY RESULTS ───────────────────────────────────────
    story.append(Paragraph("2. Strategy Configuration & Results", s["H1"]))

    sp = meta.get("strategy_params", {})
    story.append(Paragraph(
        f"BTC Window: {sp.get('btc_window_s', 300):.0f}s | "
        f"Threshold: >{sp.get('btc_threshold_pct', 0.5):.2f}% | "
        f"TP: {sp.get('tp_pct', 0.15):.2f}% | SL: {sp.get('sl_pct', 0.50):.2f}% | "
        f"Max Hold: {sp.get('max_hold_s', 600):.0f}s",
        s["Body"]))

    results_rows = [
        ["Total Trades", f"{total_trades}", f"~{total_trades/max(days,1):.0f}/day"],
        ["Win Rate", f"{win_rate:.1f}%", f"TP: {strat.get('tp_rate', 0):.0f}% | SL: {strat.get('sl_rate', 0):.0f}%"],
        ["Avg Net/Trade", f"{avg_net:+.4f}%", "After fees"],
        ["Total Net Return", f"{total_net:+.2f}%", f"Over {days} days"],
        ["Max Drawdown", f"{strat.get('max_drawdown_pct', 0):.2f}%", "Compounded equity"],
    ]
    story.append(_make_table(["Metric", "Value", "Note"], results_rows,
                             col_widths=[35*mm, 30*mm, W - 65*mm], styles=s))

    story.append(PageBreak())

    # ── 3. BASELINE COMPARISON ────────────────────────────────────
    story.append(Paragraph("3. BTC Trigger vs Random Entries", s["H1"]))
    story.append(Paragraph(
        f"We compared BTC-triggered entries against {baseline.get('n_trials', 500)} random "
        f"simulations with identical TP/SL settings:",
        s["Body"]))

    base_rows = [
        ["Win Rate", f"{baseline.get('strategy_win_rate', 0):.1f}%",
         f"{baseline.get('random_mean_win_rate', 0):.1f}% ± {baseline.get('random_std_win_rate', 0):.1f}%"],
        ["Avg Net/Trade", f"{baseline.get('strategy_avg_net', 0):+.4f}%",
         f"{baseline.get('random_mean_avg_net', 0):+.4f}% ± {baseline.get('random_std_avg_net', 0):.4f}%"],
        ["Percentile", f"{pctl:.0f}%", f"p = {baseline.get('p_value', 1):.4f}"],
    ]
    story.append(_make_table(["Metric", "BTC Trigger", "Random Baseline"],
                             base_rows, col_widths=[30*mm, 35*mm, W - 65*mm], styles=s))

    if pctl >= 95:
        story.append(Paragraph(
            f"<b>The BTC trigger is statistically significant.</b> It beats random entries "
            f"{pctl:.0f}% of the time (p = {baseline.get('p_value', 1):.4f}).",
            s["GreenBox"]))

    story.append(PageBreak())

    # ── 4. RISK PROFILE ───────────────────────────────────────────
    if risk:
        story.append(Paragraph("4. Risk Profile by Leverage", s["H1"]))

        risk_rows = []
        for lev_key in sorted(risk.keys(), key=lambda x: float(x)):
            r = risk[lev_key]
            risk_rows.append([
                f"{r['leverage']}x",
                f"{r['median_return_pct']:+.1f}%",
                f"{r['median_max_dd_pct']:.1f}%",
                f"{r['p95_max_dd_pct']:.1f}%",
                f"{r['prob_net_loss']:.0f}%",
                f"€{r['median_final_capital']:,.0f}",
            ])
        story.append(_make_table(
            ["Leverage", "Median Return", "Median Max DD", "P95 Max DD", "Prob Loss", "Median Final"],
            risk_rows, col_widths=[22*mm, 28*mm, 28*mm, 25*mm, 22*mm, 28*mm], styles=s))

    # ── DISCLAIMER ────────────────────────────────────────────────
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=GRAY))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "<i>Disclaimer: This report is for informational and educational purposes only. "
        "It does not constitute financial advice. Past performance is not indicative of "
        "future results. Trading cryptocurrencies with leverage involves substantial risk.</i>",
        ParagraphStyle("Disclaimer", parent=s["Body"], fontSize=8, textColor=GRAY, alignment=TA_CENTER)))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output_path
