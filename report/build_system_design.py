"""Render the complete system-design Markdown as a paginated PDF."""

from __future__ import annotations

import html
import math
import re
import textwrap
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "system_design.md"
OUTPUT = ROOT / "report" / "system_design.pdf"
FONT_DIR = Path("C:/Windows/Fonts")
NAVY = colors.HexColor("#14233B")
INK = colors.HexColor("#223248")
TEAL = colors.HexColor("#087E83")
PALE = colors.HexColor("#EFF4F6")
PALE_BLUE = colors.HexColor("#E7EEF5")
MUTED = colors.HexColor("#64748B")
WHITE = colors.white
BORDER = colors.HexColor("#C8D4DC")
RED_FILL = colors.HexColor("#F8ECEA")
GREEN_FILL = colors.HexColor("#EBF3EE")
TICK = chr(96)
FONT_NORMAL = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_CODE = "Courier"

if (FONT_DIR / "arial.ttf").is_file():
    pdfmetrics.registerFont(TTFont("Arial", str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_DIR / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Consolas", str(FONT_DIR / "consola.ttf")))
    if (FONT_DIR / "ariali.ttf").is_file():
        pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONT_DIR / "ariali.ttf")))
        FONT_ITALIC = "Arial-Italic"
    FONT_NORMAL, FONT_BOLD, FONT_CODE = "Arial", "Arial-Bold", "Consolas"
pdfmetrics.registerFontFamily(
    "Arial",
    normal=FONT_NORMAL,
    bold=FONT_BOLD,
    italic=FONT_ITALIC,
    boldItalic=FONT_BOLD,
)

sheet = getSampleStyleSheet()
STYLES = {
    "body": ParagraphStyle(
        "MdBody", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=9.15,
        leading=12.3, textColor=INK, spaceAfter=6.5, alignment=TA_LEFT,
        allowWidows=0, allowOrphans=0,
    ),
    "list": ParagraphStyle(
        "MdList", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=9.0,
        leading=12.0, textColor=INK, leftIndent=16, firstLineIndent=-13, spaceAfter=4.5,
    ),
    "quote": ParagraphStyle(
        "MdQuote", parent=sheet["BodyText"], fontName=FONT_ITALIC, fontSize=9.0,
        leading=12, textColor=MUTED, leftIndent=14, rightIndent=10, spaceAfter=7,
    ),
    "h2": ParagraphStyle(
        "MdH2", parent=sheet["Heading1"], fontName=FONT_BOLD, fontSize=16,
        leading=19, textColor=NAVY, spaceBefore=12, spaceAfter=7, keepWithNext=True,
    ),
    "h3": ParagraphStyle(
        "MdH3", parent=sheet["Heading2"], fontName=FONT_BOLD, fontSize=11.4,
        leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=4, keepWithNext=True,
    ),
    "h4": ParagraphStyle(
        "MdH4", parent=sheet["Heading3"], fontName=FONT_BOLD, fontSize=10,
        leading=12.3, textColor=NAVY, spaceBefore=7, spaceAfter=3, keepWithNext=True,
    ),
    "h5": ParagraphStyle(
        "MdH5", parent=sheet["Heading4"], fontName=FONT_BOLD, fontSize=9.4,
        leading=11.4, textColor=TEAL, spaceBefore=5, spaceAfter=3, keepWithNext=True,
    ),
    "table": ParagraphStyle(
        "MdTable", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=7.55,
        leading=9.55, textColor=INK, spaceAfter=0,
    ),
    "table_head": ParagraphStyle(
        "MdTableHead", parent=sheet["BodyText"], fontName=FONT_BOLD, fontSize=7.45,
        leading=9.25, textColor=WHITE, spaceAfter=0,
    ),
    "cover_title": ParagraphStyle(
        "MdCoverTitle", parent=sheet["Title"], fontName=FONT_BOLD, fontSize=27,
        leading=32, textColor=WHITE, alignment=TA_LEFT, spaceAfter=14,
    ),
    "cover_meta": ParagraphStyle(
        "MdCoverMeta", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=9.4,
        leading=13, textColor=colors.HexColor("#E2EDF1"), spaceAfter=6,
    ),
    "diagram_title": ParagraphStyle(
        "MdDiagramTitle", parent=sheet["BodyText"], fontName=FONT_BOLD, fontSize=8.6,
        leading=11, textColor=NAVY, spaceAfter=0,
    ),
    "diagram_text": ParagraphStyle(
        "MdDiagramText", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=8.1,
        leading=10.2, textColor=INK, spaceAfter=0,
    ),
    "diagram_node": ParagraphStyle(
        "MdDiagramNode", parent=sheet["BodyText"], fontName=FONT_BOLD, fontSize=8.2,
        leading=10.2, textColor=NAVY, alignment=TA_CENTER, spaceAfter=0,
    ),
    "diagram_branch": ParagraphStyle(
        "MdDiagramBranch", parent=sheet["BodyText"], fontName=FONT_BOLD, fontSize=7.6,
        leading=9.3, textColor=TEAL, alignment=TA_CENTER, spaceAfter=0,
    ),
    "diagram_small": ParagraphStyle(
        "MdDiagramSmall", parent=sheet["BodyText"], fontName=FONT_NORMAL, fontSize=7.35,
        leading=9, textColor=INK, alignment=TA_CENTER, spaceAfter=0,
    ),
    "code": ParagraphStyle(
        "MdCode", parent=sheet["Code"], fontName=FONT_CODE, fontSize=7.25,
        leading=9.2, textColor=INK, spaceAfter=0,
    ),
}

TRANSLATION = str.maketrans(
    {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
    }
)
INLINE_RE = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)"
    + "|"
    + re.escape(TICK)
    + r"([^"
    + re.escape(TICK)
    + r"]+)"
    + re.escape(TICK)
    + r"|\*\*(.+?)\*\*|(?<!\*)\*([^*]+)\*(?!\*)"
)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
LIST_RE = re.compile(r"^(\s*)([-+*]|\d+[.)])\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
SEQ_MESSAGE_RE = re.compile(r"^\s*(\w+)(-->>|->>|-->|-x|--x)\s*(\w+)\s*:\s*(.*)$")
FLOW_NODE_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)(?:"
    r"\[\((.*?)\)\]|\[\[(.*?)\]\]|\{(.*?)\}|\((.*?)\)|\[(.*?)\])?\s*$"
)
FLOW_EDGE_RE = re.compile(r"^\s*(.*?)\s*-->\s*(?:\|([^|]+)\|\s*)?(.*?)\s*$")
ARCHITECTURE_ROWS = {
    "Serving and monitoring components": [
        ["Client"], ["Nginx"], ["API"], ["Cache", "Inference", "Writer", "Monitor"], ["RequestData"],
    ],
    "Learning, registry, and serving-state components": [
        ["Bootstrap", "TruthImporter", "Trainer"], ["Registry", "RequestData"], ["Backend"], ["ServingState"],
    ],
}


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    rows: list[list[str]] | None = None
    items: list[tuple[int, bool, str]] | None = None
    language: str = ""


@dataclass
class SeqStep:
    source: str
    target: str
    message: str

    def label(self, aliases: dict[str, str]) -> str:
        source, target = aliases[self.source], aliases[self.target]
        return f"{source}: {self.message}" if self.source == self.target else f"{source} -> {target}: {self.message}"


@dataclass
class SeqAlternative:
    condition: str
    steps: list[SeqStep]


@dataclass
class SeqBranch:
    alternatives: list[SeqAlternative]


def normalize(text: str) -> str:
    return text.translate(TRANSLATION)


def inline_markup(text: str, link_color: str = "#087E83") -> str:
    text = normalize(text)
    out: list[str] = []
    offset = 0
    for match in INLINE_RE.finditer(text):
        out.append(html.escape(text[offset:match.start()], quote=False))
        if match.group(1) is not None:
            label = inline_markup(match.group(1), link_color)
            href = html.escape(normalize(match.group(2)), quote=True)
            out.append(f'<link href="{href}" color="{link_color}"><u>{label}</u></link>')
        elif match.group(3) is not None:
            out.append(f'<font name="{FONT_CODE}" color="#475569">{html.escape(match.group(3))}</font>')
        elif match.group(4) is not None:
            out.append(f"<b>{inline_markup(match.group(4), link_color)}</b>")
        else:
            out.append(f"<i>{inline_markup(match.group(5), link_color)}</i>")
        offset = match.end()
    out.append(html.escape(text[offset:], quote=False))
    return "".join(out).replace("\n", "<br/>")


def para(text: str, style: str = "body", link_color: str = "#087E83") -> Paragraph:
    return Paragraph(inline_markup(text, link_color), STYLES[style])


def pipe_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", value)]


def table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].lstrip().startswith("|")
        and bool(TABLE_SEP_RE.match(lines[index + 1]))
    )


def is_fence(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("~~~") or stripped.startswith(TICK * 3)


def parse_markdown(source: str) -> list[Block]:
    lines = source.splitlines()
    blocks: list[Block] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.strip() == "<!-- pdf:pagebreak -->":
            blocks.append(Block("pagebreak"))
            index += 1
            continue
        if is_fence(line):
            token = line.strip()[:3]
            language = line.strip()[3:].strip().lower()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(token):
                code.append(lines[index])
                index += 1
            if index == len(lines):
                raise ValueError("Markdown has an unclosed code fence.")
            index += 1
            blocks.append(Block("mermaid" if language == "mermaid" else "code", "\n".join(code), language=language))
            continue
        heading = HEADING_RE.match(line)
        if heading:
            blocks.append(Block("heading", heading.group(2).strip(), len(heading.group(1))))
            index += 1
            continue
        if table_start(lines, index):
            rows = [pipe_row(line)]
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                rows.append(pipe_row(lines[index]))
                index += 1
            width = len(rows[0])
            blocks.append(Block("table", rows=[(row + [""] * width)[:width] for row in rows]))
            continue
        list_match = LIST_RE.match(line)
        if list_match:
            base = len(list_match.group(1).expandtabs(4))
            first_ordered = bool(re.match(r"\d", list_match.group(2)))
            items: list[tuple[int, bool, str]] = []
            ordered_counts: dict[int, int] = defaultdict(int)
            while index < len(lines):
                match = LIST_RE.match(lines[index])
                if not match:
                    break
                indent = len(match.group(1).expandtabs(4))
                ordered = bool(re.match(r"\d", match.group(2)))
                if indent < base or (indent == base and ordered != first_ordered):
                    break
                level = max(0, (indent - base) // 2)
                ordered_counts[level] += 1
                items.append((level, ordered, match.group(3).strip()))
                index += 1
                while index < len(lines) and lines[index].strip():
                    continuation = lines[index]
                    next_item = LIST_RE.match(continuation)
                    if next_item and len(next_item.group(1).expandtabs(4)) <= indent:
                        break
                    if len(continuation) - len(continuation.lstrip()) > indent:
                        old_level, old_ordered, old_text = items[-1]
                        items[-1] = (old_level, old_ordered, old_text + " " + continuation.strip())
                        index += 1
                    else:
                        break
            blocks.append(Block("list", items=items))
            continue
        if line.lstrip().startswith(">"):
            quoted: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quoted.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            blocks.append(Block("quote", " ".join(quoted)))
            continue
        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line):
            blocks.append(Block("rule"))
            index += 1
            continue
        parts = [line.strip()]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            if (
                not next_line.strip()
                or HEADING_RE.match(next_line)
                or is_fence(next_line)
                or next_line.strip() == "<!-- pdf:pagebreak -->"
                or table_start(lines, index)
                or LIST_RE.match(next_line)
                or next_line.lstrip().startswith(">")
                or re.fullmatch(r"\s*([-*_])\1{2,}\s*", next_line)
            ):
                break
            parts.append(next_line.strip())
            index += 1
        blocks.append(Block("paragraph", " ".join(parts)))
    return blocks


def parse_sequence(lines: list[str]) -> tuple[dict[str, str], list[Any], str, int]:
    aliases: dict[str, str] = {}
    events: list[Any] = []
    branch: SeqBranch | None = None
    alternative: SeqAlternative | None = None
    note = ""
    messages = 0
    header_seen = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%%") or line == "autonumber":
            continue
        if line == "sequenceDiagram":
            header_seen = True
            continue
        participant = re.match(r"participant\s+(\w+)\s+as\s+(.+)$", line)
        if participant:
            aliases[participant.group(1)] = participant.group(2).strip()
            continue
        if line.startswith("alt "):
            if branch is not None:
                raise ValueError("Nested sequence branches are not supported.")
            branch = SeqBranch([])
            alternative = SeqAlternative(line[4:].strip(), [])
            branch.alternatives.append(alternative)
            events.append(branch)
            continue
        if line.startswith("else "):
            if branch is None:
                raise ValueError("Sequence else has no alt.")
            alternative = SeqAlternative(line[5:].strip(), [])
            branch.alternatives.append(alternative)
            continue
        if line == "end":
            if branch is None:
                raise ValueError("Sequence end has no alt.")
            branch = None
            alternative = None
            continue
        note_match = re.match(r"Note\s+over\s+[^:]+:\s*(.*)$", line)
        if note_match:
            note = note_match.group(1).strip()
            continue
        match = SEQ_MESSAGE_RE.match(line)
        if match:
            step = SeqStep(match.group(1), match.group(3), match.group(4).strip())
            if branch is None:
                events.append(step)
            else:
                if alternative is None:
                    raise ValueError("Sequence branch has no alternative.")
                alternative.steps.append(step)
            messages += 1
            continue
        raise ValueError(f"Unsupported sequence diagram statement: {line}")
    if not header_seen or not aliases or branch is not None:
        raise ValueError("Incomplete Mermaid sequence diagram.")
    actual = sum(
        1 if isinstance(event, SeqStep) else sum(len(item.steps) for item in event.alternatives)
        for event in events
    )
    if actual != messages:
        raise ValueError("Sequence rendering would drop a message.")
    for event in events:
        steps = [event] if isinstance(event, SeqStep) else [step for item in event.alternatives for step in item.steps]
        if any(step.source not in aliases or step.target not in aliases for step in steps):
            raise ValueError("Sequence references an unknown participant.")
    return aliases, events, note, messages


def arrow(canvas: Any, x1: float, y1: float, x2: float, y2: float, color=TEAL) -> None:
    canvas.setStrokeColor(color)
    canvas.setLineWidth(1.05)
    canvas.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    for offset in (-0.55, 0.55):
        canvas.line(
            x2,
            y2,
            x2 - 4 * math.cos(angle + offset),
            y2 - 4 * math.sin(angle + offset),
        )


class SequenceDiagram(Flowable):
    def __init__(self, aliases: dict[str, str], events: list[Any], note: str, message_count: int):
        super().__init__()
        self.aliases, self.events, self.note = aliases, events, note
        self.message_count = message_count
        self.layout: list[dict[str, Any]] = []

    def wrap(self, width: float, height: float) -> tuple[float, float]:
        self.width = width
        self.layout = []
        total = 18.0
        gap = 5.0
        for event in self.events:
            if isinstance(event, SeqStep):
                label = event.label(self.aliases)
                item = para(label, "diagram_text")
                _, ph = item.wrap(width - 44, 1000)
                h = max(29, ph + 10)
                self.layout.append({"kind": "step", "para": item, "height": h})
                total += h + gap
            else:
                pad, col_gap = 7.0, 8.0
                col_w = (width - 2 * pad - col_gap) / max(1, len(event.alternatives))
                columns = []
                max_height = 0
                for alt in event.alternatives:
                    heading = para(alt.condition, "diagram_branch")
                    _, hh = heading.wrap(col_w - 12, 1000)
                    steps = []
                    col_height = hh + 10
                    for step in alt.steps:
                        item = para(step.label(self.aliases), "diagram_small")
                        _, ph = item.wrap(col_w - 16, 1000)
                        item_h = max(25, ph + 7)
                        steps.append((item, item_h))
                        col_height += item_h + 3
                    columns.append((heading, hh, steps, col_height))
                    max_height = max(max_height, col_height)
                h = max_height + 18
                self.layout.append({"kind": "branch", "columns": columns, "height": h})
                total += h + gap
        if self.note:
            item = para(self.note, "diagram_small")
            _, ph = item.wrap(width - 22, 1000)
            self.layout.append({"kind": "note", "para": item, "height": ph + 12})
            total += ph + 12
        self.height = total
        return width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        title = para("Request lifecycle - vertical rendering of the source sequence", "diagram_title")
        _, th = title.wrap(self.width, 14)
        title.drawOn(c, 0, self.height - th)
        cursor = self.height - th - 5
        previous = None
        center = self.width / 2
        step_no = 0
        for item in self.layout:
            h = item["height"]
            y = cursor - h
            if previous is not None:
                arrow(c, center, previous - 1, center, y + h + 1)
            if item["kind"] == "step":
                step_no += 1
                x, w = 18.0, self.width - 36.0
                c.setFillColor(PALE_BLUE if step_no % 2 else PALE)
                c.setStrokeColor(BORDER)
                c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
                c.setFillColor(TEAL)
                c.circle(8, y + h / 2, 7, fill=1, stroke=0)
                c.setFillColor(WHITE)
                c.setFont(FONT_BOLD, 6.7)
                c.drawCentredString(8, y + h / 2 - 2.3, str(step_no))
                _, ph = item["para"].wrap(w - 16, 1000)
                item["para"].drawOn(c, x + 8, y + (h - ph) / 2)
            elif item["kind"] == "branch":
                c.setFillColor(colors.HexColor("#F8FAFC"))
                c.setStrokeColor(BORDER)
                c.roundRect(2, y, self.width - 4, h, 6, fill=1, stroke=1)
                pad, gap = 7.0, 8.0
                col_w = (self.width - 2 * pad - gap) / len(item["columns"])
                for index, (heading, hh, steps, _) in enumerate(item["columns"]):
                    x = pad + index * (col_w + gap)
                    c.setFillColor(PALE_BLUE if index == 0 else PALE)
                    c.roundRect(x, y + 5, col_w, h - 10, 4, fill=1, stroke=1)
                    heading.drawOn(c, x + 6, y + h - 8 - hh)
                    local = y + h - 14 - hh
                    for content, item_h in steps:
                        item_y = local - item_h
                        if item_y < y + 8:
                            raise ValueError("Sequence branch content exceeds its box.")
                        _, ph = content.wrap(col_w - 16, 1000)
                        content.drawOn(c, x + 8, item_y + (item_h - ph) / 2)
                        local = item_y - 3
            else:
                c.setFillColor(colors.HexColor("#F7F2E8"))
                c.setStrokeColor(colors.HexColor("#E7D8B8"))
                c.roundRect(8, y, self.width - 16, h, 4, fill=1, stroke=1)
                _, ph = item["para"].wrap(self.width - 32, 1000)
                item["para"].drawOn(c, 16, y + (h - ph) / 2)
            previous = y
            cursor = y - 5
        c.restoreState()


@dataclass
class FlowNode:
    identifier: str
    label: str
    shape: str
    order: int
    label_explicit: bool = False


def flow_node(atom: str, order: int) -> FlowNode:
    match = FLOW_NODE_RE.match(atom)
    if not match:
        raise ValueError(f"Unsupported flowchart node: {atom}")
    shape, label = "round", match.group(1)
    for group, kind in ((2, "cylinder"), (3, "subroutine"), (4, "decision"), (5, "round"), (6, "rect")):
        if match.group(group) is not None:
            label, shape = match.group(group).strip(), kind
            break
    return FlowNode(
        match.group(1),
        label,
        shape,
        order,
        any(match.group(group) is not None for group in range(2, 7)),
    )


def parse_flowchart(lines: list[str]) -> tuple[list[FlowNode], list[tuple[str, str, str]], dict[str, int]]:
    nodes: dict[str, FlowNode] = {}
    edges: list[tuple[str, str, str]] = []
    header = False
    orientation = ""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        match = re.match(r"flowchart\s+(LR|TD|TB|RL|BT)\s*$", line)
        if match:
            header, orientation = True, match.group(1)
            continue
        edge = FLOW_EDGE_RE.match(line)
        if edge:
            src = flow_node(edge.group(1), len(nodes))
            dst = flow_node(edge.group(3), len(nodes) + 1)
            for node in (src, dst):
                old = nodes.get(node.identifier)
                if old and node.label_explicit and old.label_explicit and old.label != node.label:
                    raise ValueError(f"Conflicting labels for Mermaid node {node.identifier}.")
                if not old:
                    node.order = len(nodes)
                    nodes[node.identifier] = node
                elif node.label_explicit and not old.label_explicit:
                    node.order = old.order
                    nodes[node.identifier] = node
            edges.append((src.identifier, dst.identifier, (edge.group(2) or "").strip()))
            continue
        if "-->" in line:
            raise ValueError(f"Unsupported flowchart edge: {line}")
        node = flow_node(line, len(nodes))
        nodes.setdefault(node.identifier, node)
    if not header or orientation not in {"TD", "TB"} or not nodes:
        raise ValueError("Training flowchart must be nonempty and vertically oriented.")

    next_nodes: dict[str, list[str]] = defaultdict(list)
    incoming = {identifier: 0 for identifier in nodes}
    for source, target, _ in edges:
        next_nodes[source].append(target)
        incoming[target] += 1
    ready = deque(identifier for identifier, count in incoming.items() if count == 0)
    depth = {identifier: 0 for identifier in ready}
    seen = []
    while ready:
        identifier = ready.popleft()
        seen.append(identifier)
        for target in next_nodes[identifier]:
            depth[target] = max(depth.get(target, 0), depth[identifier] + 1)
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if len(seen) != len(nodes):
        raise ValueError("Training flowchart has a cycle.")
    return list(nodes.values()), edges, depth


class FlowchartPart(Flowable):
    def __init__(self, title: str, nodes: list[FlowNode], edges: list[tuple[str, str, str]], depths: dict[str, int]):
        super().__init__()
        self.title, self.nodes, self.edges, self.depths = title, nodes, edges, depths
        self.rows: list[list[str]] = []
        self.positions: dict[str, tuple[float, float, float, float]] = {}
        self.row_gap = 18.0

    def wrap(self, width: float, available_height: float) -> tuple[float, float]:
        self.width = width
        grouped: dict[int, list[FlowNode]] = defaultdict(list)
        node_map = {node.identifier: node for node in self.nodes}
        for node in self.nodes:
            grouped[self.depths[node.identifier]].append(node)
        self.rows = [sorted(grouped[level], key=lambda node: node.order) for level in sorted(grouped)]
        row_heights = []
        title = para(self.title, "diagram_title")
        _, title_h = title.wrap(width, 14)
        for row in self.rows:
            gap = 16.0
            cell_w = min(width - 26, 400.0) if len(row) == 1 else (width - 26 - gap * (len(row) - 1)) / len(row)
            row_height = 0.0
            for node in row:
                box_w = min(width - 26, 285.0) if len(row) == 1 and node.shape == "decision" else cell_w
                text_w = box_w * 0.66 - 14 if node.shape == "decision" else box_w - 16
                label = para(node.label, "diagram_node")
                _, ph = label.wrap(max(45, text_w), 1000)
                row_height = max(row_height, max(31.0, ph + (18 if node.shape == "decision" else 13)))
            row_heights.append(row_height)
        self.height = title_h + 7 + sum(row_heights) + self.row_gap * max(0, len(self.rows) - 1) + 4
        self.positions = {}
        cursor = self.height - title_h - 7
        for row, row_height in zip(self.rows, row_heights):
            gap = 16.0
            if len(row) == 1:
                widths = [min(width - 26, 285.0 if row[0].shape == "decision" else 400.0)]
            else:
                cell_w = (width - 26 - gap * (len(row) - 1)) / len(row)
                widths = [cell_w] * len(row)
            total = sum(widths) + gap * (len(row) - 1)
            x = (width - total) / 2
            bottom = cursor - row_height
            for node, box_w in zip(row, widths):
                self.positions[node.identifier] = (x, bottom, box_w, row_height)
                x += box_w + gap
            cursor = bottom - self.row_gap
        if set(self.positions) != set(node_map):
            raise ValueError("Flowchart layout omitted a node.")
        return width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        title = para(self.title, "diagram_title")
        _, title_h = title.wrap(self.width, 14)
        title.drawOn(c, 0, self.height - title_h)
        self.draw_edges(c)
        for node in self.nodes:
            x, y, width, height = self.positions[node.identifier]
            fill = RED_FILL if node.identifier in {"Skip", "Outcome"} else (
                GREEN_FILL if node.identifier in {"Artifact", "Activate", "Pointer", "NewSnapshot"} else PALE_BLUE
            )
            c.setFillColor(fill)
            c.setStrokeColor(BORDER)
            if node.shape == "decision":
                path = c.beginPath()
                path.moveTo(x + width / 2, y + height)
                path.lineTo(x + width, y + height / 2)
                path.lineTo(x + width / 2, y)
                path.lineTo(x, y + height / 2)
                path.close()
                c.drawPath(path, stroke=1, fill=1)
                text_w = width * 0.66 - 14
            elif node.shape == "cylinder":
                c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
                c.line(x + 6, y + 4, x + width - 6, y + 4)
                c.line(x + 6, y + height - 4, x + width - 6, y + height - 4)
                text_w = width - 18
            else:
                c.roundRect(x, y, width, height, 6, fill=1, stroke=1)
                text_w = width - 16
            label = para(node.label, "diagram_node")
            _, ph = label.wrap(text_w, 1000)
            label.drawOn(c, x + (width - text_w) / 2, y + (height - ph) / 2)
        c.restoreState()

    def draw_edges(self, c: Any) -> None:
        row_of = {node.identifier: index for index, row in enumerate(self.rows) for node in row}
        outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for source, target, label in self.edges:
            outgoing[source].append((target, label))
        gutter = self.width - 5
        for source, targets in outgoing.items():
            sx, sy, sw, sh = self.positions[source]
            source_center = sx + sw / 2
            source_bottom = sy
            short = [(target, label) for target, label in targets if row_of[target] == row_of[source] + 1]
            long = [(target, label) for target, label in targets if row_of[target] > row_of[source] + 1]
            if len(short) > 1:
                centers = [self.positions[target][0] + self.positions[target][2] / 2 for target, _ in short]
                target_tops = [self.positions[target][1] + self.positions[target][3] for target, _ in short]
                branch_y = (source_bottom + min(target_tops)) / 2
                c.setStrokeColor(TEAL)
                c.setLineWidth(1)
                c.line(source_center, source_bottom, source_center, branch_y)
                c.line(min(centers), branch_y, max(centers), branch_y)
                for target, label in short:
                    tx, ty, tw, th = self.positions[target]
                    center = tx + tw / 2
                    arrow(c, center, branch_y, center, ty + th, TEAL)
                    if label:
                        self.edge_label(c, label, center + 3, branch_y + 2)
            else:
                for target, label in short:
                    tx, ty, tw, th = self.positions[target]
                    arrow(c, source_center, source_bottom, tx + tw / 2, ty + th, TEAL)
                    if label:
                        self.edge_label(c, label, tx + tw / 2 + 4, (source_bottom + ty + th) / 2)
            for target, label in long:
                tx, ty, tw, th = self.positions[target]
                source_mid = sy + sh / 2
                target_mid = ty + th / 2
                c.setStrokeColor(MUTED)
                c.setLineWidth(0.8)
                c.line(sx + sw, source_mid, gutter, source_mid)
                c.line(gutter, source_mid, gutter, target_mid)
                arrow(c, gutter, target_mid, tx + tw, target_mid, MUTED)
                if label:
                    self.edge_label(c, label, gutter - 34, (source_mid + target_mid) / 2)

    @staticmethod
    def edge_label(c: Any, text: str, x: float, y: float) -> None:
        c.setFont(FONT_BOLD, 6.4)
        c.setFillColor(NAVY)
        c.drawString(x, y, normalize(text)[:28])


def render_flowchart(code: str) -> tuple[list[Flowable], tuple[int, int]]:
    nodes, edges, depths = parse_flowchart(code.splitlines())
    decisions = [depths[node.identifier] for node in nodes if node.shape == "decision"]
    cut = min(decisions) if decisions else max(depths.values()) // 2
    before = {identifier for identifier, depth in depths.items() if depth <= cut}
    after = {identifier for identifier, depth in depths.items() if depth >= cut}
    if not after:
        after = set(before)
    first_edges = [(a, b, label) for a, b, label in edges if a in before and b in before]
    second_edges = [(a, b, label) for a, b, label in edges if a in after and b in after]
    if before | after != {node.identifier for node in nodes}:
        raise ValueError("Vertical flowchart split would drop source nodes.")
    if set(first_edges + second_edges) != set(edges):
        raise ValueError("Vertical flowchart split would drop source edges.")

    def part(ids: set[str], included_edges: list[tuple[str, str, str]], name: str, shift: int) -> FlowchartPart:
        selected = [node for node in nodes if node.identifier in ids]
        selected_depths = {identifier: depths[identifier] - shift for identifier in ids}
        return FlowchartPart(name, selected, included_edges, selected_depths)

    result = [
        part(before, first_edges, "Training flow - part 1 of 2, labels through timer decision", 0),
        Spacer(1, 8),
        part(after, second_edges, "Training flow - part 2 of 2, gate outcome through serving pointer", cut),
        Spacer(1, 7),
    ]
    return result, (len(nodes), len(edges))


def parse_architecture(lines: list[str]) -> tuple[str, list[FlowNode], list[tuple[str, str, str]]]:
    title = ""
    header = False
    orientation = ""
    nodes: dict[str, FlowNode] = {}
    edges: list[tuple[str, str, str]] = []

    def add_node(node: FlowNode) -> None:
        old = nodes.get(node.identifier)
        if old and node.label_explicit and old.label_explicit and old.label != node.label:
            raise ValueError(f"Conflicting labels for architecture node {node.identifier}.")
        if not old:
            node.order = len(nodes)
            nodes[node.identifier] = node
        elif node.label_explicit and not old.label_explicit:
            node.order = old.order
            nodes[node.identifier] = node

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        marker = re.match(r"%%\s*architecture-view:\s*(.+)$", line)
        if marker:
            title = marker.group(1).strip()
            continue
        if line.startswith("%%"):
            continue
        header_match = re.match(r"flowchart\s+(TD|TB)\s*$", line)
        if header_match:
            header, orientation = True, header_match.group(1)
            continue
        edge = FLOW_EDGE_RE.match(line)
        if edge:
            source = flow_node(edge.group(1), len(nodes))
            target = flow_node(edge.group(3), len(nodes) + 1)
            add_node(source)
            add_node(target)
            edges.append((source.identifier, target.identifier, (edge.group(2) or "").strip()))
            continue
        if "-->" in line:
            raise ValueError(f"Unsupported architecture edge: {line}")
        add_node(flow_node(line, len(nodes)))

    expected = {identifier for row in ARCHITECTURE_ROWS.get(title, []) for identifier in row}
    if not header or orientation not in {"TD", "TB"} or title not in ARCHITECTURE_ROWS:
        raise ValueError("Architecture view has an unsupported title or orientation.")
    if set(nodes) != expected:
        raise ValueError(f"Architecture view {title!r} does not match its declared component layout.")
    return title, list(nodes.values()), edges


class ArchitectureDiagram(Flowable):
    def __init__(self, title: str, nodes: list[FlowNode], edges: list[tuple[str, str, str]]):
        super().__init__()
        self.title, self.nodes, self.edges = title, nodes, edges
        self.rows = ARCHITECTURE_ROWS[title]
        self.node_map = {node.identifier: node for node in nodes}
        self.positions: dict[str, tuple[float, float, float, float, int]] = {}
        self.row_gap = 52.0
        self.edge_legend: Table | None = None
        self.legend_title: Paragraph | None = None
        self.legend_table_height = 0.0
        self.legend_height = 0.0

    def wrap(self, width: float, available_height: float) -> tuple[float, float]:
        self.width = width
        title = para(self.title, "diagram_title")
        _, title_h = title.wrap(width, 14)
        row_heights = []
        row_widths: list[list[float]] = []
        for row in self.rows:
            gap = 16.0
            if len(row) == 1:
                widths = [min(width - 30, 450.0)]
            else:
                box_w = (width - 24 - gap * (len(row) - 1)) / len(row)
                widths = [box_w] * len(row)
            row_widths.append(widths)
            height = 0.0
            for identifier, box_w in zip(row, widths):
                label = para(self.node_map[identifier].label, "diagram_node")
                _, label_h = label.wrap(box_w - 16, 1000)
                height = max(height, max(44.0, label_h + 16.0))
            row_heights.append(height)

        self.prepare_edge_legend(width)
        self.height = (
            title_h + 9 + sum(row_heights) + self.row_gap * (len(self.rows) - 1)
            + 8 + self.legend_height
        )
        cursor = self.height - title_h - 9
        self.positions = {}
        for row_index, (row, widths, row_h) in enumerate(zip(self.rows, row_widths, row_heights)):
            gap = 16.0
            total = sum(widths) + gap * (len(row) - 1)
            x = (width - total) / 2
            bottom = cursor - row_h
            for identifier, box_w in zip(row, widths):
                self.positions[identifier] = (x, bottom, box_w, row_h, row_index)
                x += box_w + gap
            cursor = bottom - self.row_gap
        if set(self.positions) != set(self.node_map):
            raise ValueError("Architecture layout omitted a component.")
        return width, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        title = para(self.title, "diagram_title")
        _, title_h = title.wrap(self.width, 14)
        title.drawOn(c, 0, self.height - title_h)
        self.draw_edges(c)
        for node in self.nodes:
            x, y, width, height, _ = self.positions[node.identifier]
            fill = PALE if node.identifier in {"RequestData", "Registry", "ServingState"} else (
                GREEN_FILL if node.identifier in {"API", "Backend", "Trainer"} else PALE_BLUE
            )
            c.setFillColor(fill)
            c.setStrokeColor(BORDER)
            if node.shape == "cylinder":
                c.roundRect(x, y, width, height, 7, fill=1, stroke=1)
                c.line(x + 6, y + 4, x + width - 6, y + 4)
                c.line(x + 6, y + height - 4, x + width - 6, y + height - 4)
            else:
                c.roundRect(x, y, width, height, 6, fill=1, stroke=1)
            label = para(node.label, "diagram_node")
            _, label_h = label.wrap(width - 16, 1000)
            label.drawOn(c, x + 8, y + (height - label_h) / 2)
        if self.edge_legend is not None and self.legend_title is not None:
            self.edge_legend.drawOn(c, 0, 8)
            self.legend_title.drawOn(c, 0, 8 + self.legend_table_height + 4)
        c.restoreState()

    def draw_edges(self, c: Any) -> None:
        left_gutter, right_gutter = 3.0, self.width - 3.0
        for number, (source, target, _label) in enumerate(self.edges, 1):
            sx, sy, sw, sh, source_row = self.positions[source]
            tx, ty, tw, th, target_row = self.positions[target]
            source_mid, target_mid = sy + sh / 2, ty + th / 2
            source_center, target_center = sx + sw / 2, tx + tw / 2
            if target_row == source_row + 1:
                arrow(c, source_center, sy, target_center, ty + th, TEAL)
                self.edge_marker(c, number, (source_center + target_center) / 2, (sy + ty + th) / 2)
            elif target_row > source_row + 1:
                c.setStrokeColor(TEAL)
                c.setLineWidth(1.0)
                c.line(sx + sw, source_mid, right_gutter, source_mid)
                c.line(right_gutter, source_mid, right_gutter, target_mid)
                arrow(c, right_gutter, target_mid, tx + tw, target_mid, TEAL)
                next_row_top = max(
                    self.positions[identifier][1] + self.positions[identifier][3]
                    for identifier in self.rows[source_row + 1]
                )
                self.edge_marker(c, number, right_gutter - 5, (sy + next_row_top) / 2)
            elif target_row == source_row - 1:
                arrow(c, source_center, sy + sh, target_center, ty, TEAL)
                self.edge_marker(c, number, (source_center + target_center) / 2, (sy + sh + ty) / 2)
            elif target_row < source_row - 1:
                c.setStrokeColor(TEAL)
                c.setLineWidth(1.0)
                c.line(sx, source_mid, left_gutter, source_mid)
                c.line(left_gutter, source_mid, left_gutter, target_mid)
                arrow(c, left_gutter, target_mid, tx, target_mid, TEAL)
                next_row_bottom = min(self.positions[identifier][1] for identifier in self.rows[target_row + 1])
                self.edge_marker(c, number, left_gutter + 5, (ty + th + next_row_bottom) / 2)
            elif sx + sw <= tx:
                arrow(c, sx + sw, source_mid, tx, target_mid, TEAL)
                self.edge_marker(c, number, (sx + sw + tx) / 2, (source_mid + target_mid) / 2)
            else:
                arrow(c, sx, source_mid, tx + tw, target_mid, TEAL)
                self.edge_marker(c, number, (sx + tx + tw) / 2, (source_mid + target_mid) / 2)

    def prepare_edge_legend(self, width: float) -> None:
        style = ParagraphStyle(
            "ArchitectureLegend",
            fontName=FONT_NORMAL,
            fontSize=7.0,
            leading=8.2,
            textColor=INK,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
        )
        cells = [
            Paragraph(f'<b>{number}.</b> {inline_markup(label)}', style)
            for number, (_, _, label) in enumerate(self.edges, 1)
        ]
        rows = [
            [cells[index], cells[index + 1] if index + 1 < len(cells) else ""]
            for index in range(0, len(cells), 2)
        ]
        col_width = (width - 18) / 2
        self.edge_legend = Table(rows, colWidths=[col_width, col_width], hAlign="LEFT")
        self.edge_legend.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        _, self.legend_table_height = self.edge_legend.wrap(width, 1000)
        self.legend_title = para("Edge key", "diagram_title")
        _, title_height = self.legend_title.wrap(width, 14)
        self.legend_height = self.legend_table_height + title_height + 7

    @staticmethod
    def edge_marker(c: Any, number: int, x: float, y: float) -> None:
        radius = 6.2
        c.setFillColor(WHITE)
        c.setStrokeColor(TEAL)
        c.setLineWidth(0.8)
        c.circle(x, y, radius, fill=1, stroke=1)
        c.setFillColor(NAVY)
        c.setFont(FONT_BOLD, 6.5)
        c.drawCentredString(x, y - 2.25, str(number))


def render_architecture(lines: list[str]) -> tuple[ArchitectureDiagram, int, int]:
    title, nodes, edges = parse_architecture(lines)
    return ArchitectureDiagram(title, nodes, edges), len(nodes), len(edges)


def make_table(rows: list[list[str]], width: float) -> LongTable:
    if not rows:
        raise ValueError("Markdown table has no header.")
    header, content = rows[0], rows[1:]
    columns = len(header)
    content = [(row + [""] * columns)[:columns] for row in content]
    lengths = [
        max(4, min(90, max(len(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", row[i])) for row in [header] + content)))
        for i in range(columns)
    ]
    weights = [value**0.58 for value in lengths]
    widths = [width * weight / sum(weights) for weight in weights]
    minimum = 0.62 * inch
    fixed: set[int] = set()
    while True:
        new_fixed = {i for i, value in enumerate(widths) if value < minimum and i not in fixed}
        if not new_fixed:
            break
        fixed |= new_fixed
        flexible = [i for i in range(columns) if i not in fixed]
        remaining = width - len(fixed) * minimum
        total = sum(weights[i] for i in flexible)
        if not flexible or remaining <= 0:
            widths = [width / columns] * columns
            break
        widths = [minimum if i in fixed else remaining * weights[i] / total for i in range(columns)]

    data = [[para(cell, "table_head") for cell in header]]
    data.extend([[para(cell, "table") for cell in row] for row in content])
    table = LongTable(
        data, colWidths=widths, repeatRows=1, splitByRow=1, splitInRow=0, hAlign="LEFT"
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
                ("LINEBELOW", (0, 0), (-1, 0), 0.9, TEAL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


class CodeBlock(Flowable):
    def __init__(self, text: str):
        super().__init__()
        self.text = text
        self.lines: list[str] = []

    def wrap(self, width: float, height: float) -> tuple[float, float]:
        self.width = width
        columns = max(25, int((width - 20) / 4.2))
        self.lines = []
        for line in self.text.splitlines() or [""]:
            self.lines.extend(textwrap.wrap(line, columns, subsequent_indent="    ") or [""])
        self.height = len(self.lines) * 9.3 + 12
        return width, self.height

    def draw(self) -> None:
        c = self.canv
        c.setFillColor(PALE)
        c.setStrokeColor(BORDER)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        c.setFillColor(INK)
        c.setFont(FONT_CODE, 7.1)
        y = self.height - 12
        for line in self.lines:
            c.drawString(8, y - 7, normalize(line))
            y -= 9.3


class HorizontalRule(Flowable):
    def wrap(self, width: float, height: float) -> tuple[float, float]:
        self.width, self.height = width, 7
        return width, self.height

    def draw(self) -> None:
        self.canv.setStrokeColor(BORDER)
        self.canv.setLineWidth(0.6)
        self.canv.line(0, 3, self.width, 3)


def markdown_story(source: str) -> tuple[list[Flowable], dict[str, int]]:
    blocks = parse_markdown(source)
    if not blocks or blocks[0].kind != "heading" or blocks[0].level != 1:
        raise ValueError("Markdown must begin with a level-one title.")
    cover = [blocks[0]]
    index = 1
    while index < len(blocks) and blocks[index].kind != "pagebreak" and not (
        blocks[index].kind == "heading" and blocks[index].level <= 2
    ):
        cover.append(blocks[index])
        index += 1
    if index == len(blocks) or any(block.kind != "paragraph" for block in cover[1:]):
        raise ValueError("Expected an initial title, metadata paragraphs, and body sections.")

    story: list[Flowable] = [Spacer(1, 0.52 * inch), para(blocks[0].text, "cover_title", "#8DD9D4")]
    for block in cover[1:]:
        story.append(para(block.text, "cover_meta", "#8DD9D4"))
    story.extend([Spacer(1, 0.1 * inch), PageBreak()])
    stats = {
        "headings": 0, "paragraphs": 0, "lists": 0, "tables": 0, "quotes": 0,
        "mermaid": 0, "sequence_messages": 0, "flowchart_nodes": 0, "flowchart_edges": 0,
        "architecture_views": 0, "architecture_nodes": 0, "architecture_edges": 0,
        "code_blocks": 0, "cover_blocks": len(cover), "total_blocks": len(blocks),
    }
    for block in blocks[index:]:
        if block.kind == "heading":
            stats["headings"] += 1
            story.append(para(block.text, {2: "h2", 3: "h3", 4: "h4"}.get(block.level, "h5")))
        elif block.kind == "paragraph":
            stats["paragraphs"] += 1
            story.append(para(block.text))
        elif block.kind == "list":
            stats["lists"] += 1
            ordered_index = 0
            for position, (indent, ordered, text) in enumerate(block.items or []):
                if ordered:
                    ordered_index += 1
                    marker = f"{ordered_index}."
                else:
                    marker = "-"
                style = ParagraphStyle(
                    f"MarkdownList_{stats['lists']}_{position}",
                    parent=STYLES["list"],
                    leftIndent=16 + indent * 12,
                    firstLineIndent=-13,
                )
                story.append(Paragraph(f'<font color="#087E83">{marker}</font> {inline_markup(text)}', style))
        elif block.kind == "quote":
            stats["quotes"] += 1
            story.append(para(block.text, "quote"))
        elif block.kind == "table":
            stats["tables"] += 1
            story.extend([Spacer(1, 4), make_table(block.rows or [], letter[0] - 1.36 * inch), Spacer(1, 7)])
        elif block.kind == "mermaid":
            stats["mermaid"] += 1
            if block.text.lstrip().startswith("sequenceDiagram"):
                aliases, events, note, count = parse_sequence(block.text.splitlines())
                story.extend([SequenceDiagram(aliases, events, note, count), Spacer(1, 7)])
                stats["sequence_messages"] += count
            elif any(line.strip().startswith("%% architecture-view:") for line in block.text.splitlines()):
                view, nodes, edges = render_architecture(block.text.splitlines())
                story.extend([view, Spacer(1, 7)])
                stats["architecture_views"] += 1
                stats["architecture_nodes"] += nodes
                stats["architecture_edges"] += edges
            elif block.text.lstrip().startswith("flowchart"):
                parts, (nodes, edges) = render_flowchart(block.text)
                story.extend(parts)
                stats["flowchart_nodes"] += nodes
                stats["flowchart_edges"] += edges
            else:
                raise ValueError("Unsupported Mermaid diagram.")
        elif block.kind == "code":
            stats["code_blocks"] += 1
            story.extend([CodeBlock(block.text), Spacer(1, 7)])
        elif block.kind == "pagebreak":
            story.append(PageBreak())
        elif block.kind == "rule":
            story.append(HorizontalRule())
        else:
            raise ValueError(f"Unhandled Markdown block: {block.kind}")
    return story, stats


def cover_page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 0.16 * inch, width, 0.16 * inch, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#E2EDF1"))
    canvas.setFont(FONT_NORMAL, 8.2)
    canvas.drawRightString(width - 0.65 * inch, 0.38 * inch, "SYS-304  |  SYSTEM DESIGN")
    canvas.restoreState()


def page_header(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    width, height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 0.36 * inch, width, 0.36 * inch, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont(FONT_BOLD, 8)
    canvas.drawString(0.62 * inch, height - 0.235 * inch, "SYS-304  /  HOUSE PRICES")
    review = getattr(doc, "export_date", "OCTOBER 20, 2026").upper()
    canvas.setFont(FONT_NORMAL, 7.2)
    canvas.drawRightString(width - 0.62 * inch, height - 0.235 * inch, f"SYSTEM DESIGN  |  {review}")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.62 * inch, 0.49 * inch, width - 0.62 * inch, 0.49 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT_NORMAL, 7.2)
    canvas.drawString(0.62 * inch, 0.31 * inch, "Markdown source export")
    canvas.drawRightString(width - 0.62 * inch, 0.31 * inch, f"PAGE {doc.page}")
    canvas.restoreState()


def build() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    story, stats = markdown_story(markdown)
    title_match = HEADING_RE.match(next(line for line in markdown.splitlines() if line.strip()))
    title = title_match.group(2).strip() if title_match else "System design"
    date_match = re.search(r"visually checked on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", markdown)
    export_date = date_match.group(1) if date_match else "October 20, 2026"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=letter, rightMargin=0.68 * inch, leftMargin=0.68 * inch,
        topMargin=0.62 * inch, bottomMargin=0.62 * inch,
        title=normalize(title), author="SYS-304 House Prices",
        subject="Complete system design rendered from report/system_design.md",
        allowSplitting=1,
    )
    doc.export_date = export_date
    doc.build(story, onFirstPage=cover_page, onLaterPages=page_header)

    from pypdf import PdfReader

    reader = PdfReader(str(OUTPUT))
    if not reader.pages:
        raise RuntimeError("PDF contains no pages.")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    if "House Prices" not in normalize(extracted):
        raise RuntimeError("PDF text check did not find the Markdown report title.")
    print(f"Markdown source: {SOURCE}")
    print(f"PDF output: {OUTPUT}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Source coverage: {stats}")


if __name__ == "__main__":
    build()
