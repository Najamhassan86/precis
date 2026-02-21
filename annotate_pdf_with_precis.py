# annotate_pdf_with_precis.py
"""
ROBUST PDF ANNOTATION (digital + scanned) with **OCR-ANCHOR FIX**

Why your pages 1–16 were 0 matches:
- Your LLM annotations (target/context) are NOT guaranteed to exist in Azure OCR text.
- So matching fails => rect=None => no highlight/arrow.

This version fixes that by:
1) Digital PDFs: use real PDF text via PyMuPDF (best).
2) Scanned PDFs: prefer **anchor_quote** (verbatim from OCR text) if provided.
3) If anchor_quote missing: tries legacy (target/context) matching but:
   - if it can't match => treat as PAGE-LEVEL feedback (right margin box only, no arrow).
4) Global dedup so annotations don't attach to same region.
5) Debug mode to show why matches fail.

IMPORTANT (upstream requirement):
- Update your Grok/LLM annotation prompt to output:
    anchor_quote: EXACT substring copied from OCR page text (6–25 words)
    anchor_line_hint: optional (index) if you want
  If the model cannot find a verbatim quote, it must set anchor_quote=null.
"""

import io
import re
import difflib
import unicodedata
from typing import Any, Dict, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2


# Right-side annotation layout: set True to print box geometry and overflow flags
DEBUG_RIGHT_BOX_LAYOUT = False
RIGHT_BOX_MIN_GAP = 8  # minimum vertical gap between right-side boxes (px)

# Spelling annotation placement: set True to print box/connector geometry per error
DEBUG_SPELL_DRAW = True
# Spelling match: set True to print match method and result per error
DEBUG_SPELL_MATCH = False
# Spelling box: gap above/below word, padding, min box width, font size range
SPELL_BOX_GAP_PT = 8.0
SPELL_BOX_PADDING_PT = 6.0
SPELL_BOX_MIN_WIDTH_PT = 40.0
SPELL_BOX_FONT_START = 10
SPELL_BOX_FONT_MIN = 8
SPELL_CONNECTOR_WIDTH = 1.0
SPELL_COLOR = (0.8, 0, 0)

# ============================================================
# TEXT HELPERS
# ============================================================
STOP = {
    "the", "and", "or", "to", "of", "in", "a", "an", "is", "are", "was", "were",
    "that", "this", "it", "as", "by", "for", "with", "on", "at", "from", "be",
    "have", "has", "had", "will", "would", "should", "can", "could", "may", "might",
}

def _normalize(text: str) -> str:
    return (text or "").strip().lower()

def _tokenize_full(text: str) -> List[str]:
    clean = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return [t for t in clean.split() if t]

def _keywords_only(s: str, max_words: int = 7) -> str:
    toks = _tokenize_full(s)
    toks = [t for t in toks if t not in STOP]
    return " ".join(toks[:max_words])

def _token_coverage(target: str, candidate: str) -> float:
    t_tokens = _tokenize_full(target)
    c_tokens = _tokenize_full(candidate)
    if not t_tokens or not c_tokens:
        return 0.0
    t_set = set(t_tokens)
    c_set = set(c_tokens)
    inter = len(t_set & c_set)
    return inter / max(1, len(t_set))


# ============================================================
# MATCH SCORING
# ============================================================
def _line_match_score(target: str, candidate: str) -> float:
    """
    Score 0..1. Designed for noisy handwriting OCR.
    Combines coverage + best local window similarity + overlap + global similarity.
    """
    t = _normalize(target)
    c = _normalize(candidate)
    if not t or not c:
        return 0.0

    t_tokens = _tokenize_full(t)
    c_tokens = _tokenize_full(c)
    if not t_tokens or not c_tokens:
        return 0.0

    t_set = set(t_tokens)
    c_set = set(c_tokens)
    inter = len(t_set & c_set)

    coverage = inter / max(1, len(t_set))
    overlap = inter / max(1, len(t_set | c_set))

    joined_target = " ".join(t_tokens)
    best_local = 0.0
    N = len(t_tokens)
    tol = 1 if N <= 3 else max(2, int(N * 0.35))

    for L in range(max(1, N - tol), min(len(c_tokens), N + tol) + 1):
        for i in range(0, len(c_tokens) - L + 1):
            window = " ".join(c_tokens[i:i + L])
            r = difflib.SequenceMatcher(None, joined_target, window).ratio()
            if r > best_local:
                best_local = r

    seq = difflib.SequenceMatcher(None, t, c).ratio()

    score = 0.55 * coverage + 0.25 * best_local + 0.15 * overlap + 0.05 * seq
    if coverage >= 0.75:
        score += 0.10
    return min(1.0, score)


# ============================================================
# GEOMETRY HELPERS (OCR LINES)
# ============================================================
def _line_text(line: Dict[str, Any]) -> str:
    return (line.get("content") or line.get("text") or "").strip()

def _line_polygon_any(line: Dict[str, Any]) -> Any:
    return (
        line.get("boundingPolygon")
        or line.get("polygon")
        or line.get("bbox")
        or line.get("boundingBox")
        or line.get("box")
    )

def _poly_to_points_generic(poly: Any) -> List[Tuple[float, float]]:
    if isinstance(poly, list) and poly and isinstance(poly[0], dict) and "x" in poly[0]:
        return [(float(p["x"]), float(p["y"])) for p in poly]

    if isinstance(poly, list) and poly and isinstance(poly[0], (list, tuple)) and len(poly[0]) >= 2:
        return [(float(p[0]), float(p[1])) for p in poly]

    if isinstance(poly, (list, tuple)):
        nums = [float(v) for v in poly if isinstance(v, (int, float))]
        if len(nums) >= 4:
            pts = []
            for i in range(0, len(nums) - 1, 2):
                pts.append((nums[i], nums[i + 1]))
            return pts
    return []

def _points_to_rect(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))

def _points_to_rect(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _find_error_in_context_ocr(
    page_ocr: Dict[str, Any], 
    error_text: str, 
    anchor_quote: str,
    page_extent: Optional[Tuple[float, float]], 
    orig_w: int, 
    orig_h: int
) -> Optional[Tuple[int, int, int, int]]:
    """Find error within its context (anchor_quote) for more accurate positioning."""
    if not page_ocr or not error_text or not anchor_quote or not page_extent:
        return None
    
    # First, find lines that contain part of the anchor quote
    error_lower = error_text.lower()
    anchor_lower = anchor_quote.lower()
    
    # Get all OCR text as a continuous string with line info
    for ln in page_ocr.get("lines", []) or []:
        line_text = _line_text(ln).lower()
        
        # Check if this line contains the anchor quote (or part of it)
        if any(word in line_text for word in anchor_lower.split() if len(word) > 3):
            # Now check if error is in this line
            if error_lower in line_text:
                # Found the right line, now find the specific word
                line_words = ln.get("words", [])
                if not line_words:
                    # Use line bbox as fallback
                    poly = _line_polygon_any(ln)
                    if poly:
                        pts = _poly_to_points_generic(poly)
                        rect = _points_to_rect(pts)
                        if rect:
                            return _scale_rect_by_extent(rect, page_extent, orig_w, orig_h)
                else:
                    # Search words for the error
                    for w in line_words:
                        w_text = w.get("text", "").strip().lower()
                        if error_lower in w_text or w_text in error_lower:
                            w_poly = w.get("bbox") or w.get("boundingPolygon") or []
                            if w_poly:
                                pts = _poly_to_points_generic(w_poly)
                                rect = _points_to_rect(pts)
                                if rect:
                                    return _scale_rect_by_extent(rect, page_extent, orig_w, orig_h)
    
    return None

def _find_error_word_span_ocr(page_ocr: Dict[str, Any], error_text: str, page_extent: Optional[Tuple[float, float]], orig_w: int, orig_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Find the bounding box for error_text by matching in OCR words."""
    if not page_ocr or not error_text or not page_extent:
        return None
    
    target = _norm_token_for_spelling(error_text)
    if not target:
        return None
    
    words = []
    for ln in page_ocr.get("lines", []) or []:
        line_words = ln.get("words", [])
        if line_words:
            words.extend(line_words)
        else:
            # Fallback: treat line as single word
            text = _line_text(ln)
            poly = _line_polygon_any(ln)
            if text and poly:
                words.append({"text": text, "bbox": poly})
    
    # Build tokens with rects
    tokens = []
    for w in words:
        w_text = w.get("text", "").strip()
        w_poly = w.get("bbox") or w.get("boundingPolygon") or []
        if not w_text or not w_poly:
            continue
        pts = _poly_to_points_generic(w_poly)
        rect = _points_to_rect(pts)
        if rect:
            scaled = _scale_rect_by_extent(rect, page_extent, orig_w, orig_h)
            tokens.append((_norm_token_for_spelling(w_text), scaled, w_text))
    
    # Try exact match first
    for t, r, original_text in tokens:
        if t == target:
            return r
    
    # Try case-insensitive substring match (for partial words)
    error_lower = error_text.lower()
    for t, r, original_text in tokens:
        if error_lower in original_text.lower() or original_text.lower() in error_lower:
            # Close enough match
            return r
    
    # Multi-word span (join consecutive) - more flexible
    for i in range(len(tokens)):
        for j in range(i, min(i + 8, len(tokens))):
            # Build accumulated text
            acc = ""
            acc_with_spaces = ""
            x0, y0, x1, y1 = tokens[i][1]
            for k in range(i, j + 1):
                acc += tokens[k][0]
                acc_with_spaces += tokens[k][2] + " "
                rk = tokens[k][1]
                x0 = min(x0, rk[0])
                y0 = min(y0, rk[1])
                x1 = max(x1, rk[2])
                y1 = max(y1, rk[3])
            
            # Check if matches
            if acc == target:
                return (x0, y0, x1, y1)
            
            # Check if error is contained in this span (fuzzy match)
            if error_lower in acc_with_spaces.lower():
                return (x0, y0, x1, y1)
    
    return None

def _compute_page_extent(page_ocr: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    w = page_ocr.get("page_width") or page_ocr.get("width") or page_ocr.get("pageWidth") or page_ocr.get("page_width_px")
    h = page_ocr.get("page_height") or page_ocr.get("height") or page_ocr.get("pageHeight") or page_ocr.get("page_height_px")

    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
        return (float(w), float(h))

    lines = page_ocr.get("lines") or []
    if not lines:
        return None

    max_x, max_y, count = 0.0, 0.0, 0
    for ln in lines:
        pts = _poly_to_points_generic(_line_polygon_any(ln))
        for x, y in pts:
            max_x = max(max_x, float(x))
            max_y = max(max_y, float(y))
            count += 1

    if count == 0 or max_x <= 0 or max_y <= 0:
        return None

    if max_x <= 1.5 and max_y <= 1.5:
        return (1.0, 1.0)
    if max_x <= 12.0 and max_y <= 12.0:
        return (max(max_x, 1.0), max(max_y, 1.0))

    return (max_x, max_y)

def _scale_rect_by_extent(
    rect: Tuple[float, float, float, float],
    extent: Tuple[float, float],
    pix_w: int,
    pix_h: int
) -> Tuple[int, int, int, int]:
    ex, ey = extent
    ex = ex if ex > 0 else 1.0
    ey = ey if ey > 0 else 1.0
    sx = pix_w / ex
    sy = pix_h / ey
    x1, y1, x2, y2 = rect
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

def _union_rects(r1, r2):
    if not r1:
        return r2
    if not r2:
        return r1
    return (min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]))


# ============================================================
# PDF TEXT MATCHING (DIGITAL PDFs)
# ============================================================
def _extract_pdf_words(page: fitz.Page) -> List[Dict[str, Any]]:
    words = page.get_text("words")
    out = []
    for w in words or []:
        if len(w) >= 5:
            x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
            txt = (txt or "").strip()
            if txt:
                out.append({"text": txt, "rect": (float(x0), float(y0), float(x1), float(y1))})
    return out

def _pdf_rects_to_pix_rect(
    rect_pts: Tuple[float, float, float, float],
    page: fitz.Page,
    pix_w: int,
    pix_h: int
) -> Tuple[int, int, int, int]:
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    sx = pix_w / pw if pw > 0 else 1.0
    sy = pix_h / ph if ph > 0 else 1.0
    x1, y1, x2, y2 = rect_pts
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

def _find_match_rect_in_pdf_text(
    page: fitz.Page,
    pix_w: int,
    pix_h: int,
    target_text: str,
    window_tokens: int = 18
) -> Optional[Tuple[int, int, int, int]]:
    tgt = _normalize(target_text)
    if not tgt:
        return None

    words = _extract_pdf_words(page)
    if len(words) < 10:
        return None

    tokens = [_normalize(w["text"]) for w in words]
    nonempty = sum(1 for t in tokens if t)
    if nonempty < 10:
        return None

    best_score = 0.0
    best_span = None

    tgt_tokens = _tokenize_full(tgt)
    if not tgt_tokens:
        return None

    N = len(tgt_tokens)
    for L in range(max(6, N - 4), min(window_tokens, N + 6) + 1):
        for i in range(0, len(tokens) - L + 1):
            chunk = " ".join(tokens[i:i + L]).strip()
            if not chunk:
                continue
            score = _line_match_score(tgt, chunk)
            cov = _token_coverage(tgt, chunk)
            if score < 0.62 or cov < 0.55:
                continue
            if score > best_score:
                best_score = score
                best_span = (i, i + L - 1)

    if not best_span:
        kw = _keywords_only(target_text)
        if kw and kw != tgt:
            return _find_match_rect_in_pdf_text(page, pix_w, pix_h, kw, window_tokens=window_tokens)
        return None

    i0, i1 = best_span
    rect = None
    for k in range(i0, i1 + 1):
        rect = _union_rects(rect, words[k]["rect"])
    if not rect:
        return None

    rect_px = _pdf_rects_to_pix_rect(rect, page, pix_w, pix_h)
    x1, y1, x2, y2 = rect_px
    pad = 4
    return (max(0, x1 - pad), max(0, y1 - pad), min(pix_w - 1, x2 + pad), min(pix_h - 1, y2 + pad))


# ============================================================
# OCR MATCHING (SCANNED PDFs)
# ============================================================
def _best_window_match(
    lines: List[Dict[str, Any]],
    start: int,
    win: int
) -> Tuple[str, Optional[Tuple[float, float, float, float]]]:
    texts = []
    rects = []
    for j in range(start, start + win):
        t = _line_text(lines[j])
        if t:
            texts.append(t)
        pts = _poly_to_points_generic(_line_polygon_any(lines[j]))
        r = _points_to_rect(pts)
        if r:
            rects.append(r)

    if not texts or not rects:
        return "", None

    combined_text = " ".join(texts).strip()
    u = rects[0]
    for r in rects[1:]:
        u = _union_rects(u, r)
    return combined_text, u

def _find_best_match_rect_from_ocr(
    page_ocr: Dict[str, Any],
    target_text: str,
    pix_w: int,
    pix_h: int,
    *,
    prefer_anchor: bool = False
) -> Optional[Tuple[int, int, int, int]]:
    """
    OCR matching fallback. Returns pixel rect.
    If prefer_anchor=True, uses slightly looser thresholds because anchor_quote
    is supposed to be copied from OCR (verbatim-ish).
    """
    if not target_text or not _normalize(target_text):
        return None

    extent = _compute_page_extent(page_ocr)
    if not extent:
        return None

    lines = page_ocr.get("lines") or []
    if not lines:
        return None

    targets_to_try = [target_text]
    kw = _keywords_only(target_text)
    if kw and kw.lower() != target_text.lower():
        targets_to_try.append(kw)

    best_score = 0.0
    best_rect_raw = None
    best_is_long = False

    for tgt in targets_to_try:
        tgt_tokens = _tokenize_full(tgt)
        if not tgt_tokens:
            continue

        is_short = len(tgt_tokens) <= 2
        is_long = len(tgt_tokens) >= 8
        best_is_long = best_is_long or is_long

        if prefer_anchor:
            # anchors are from OCR text, allow looser match
            if is_short:
                min_score, min_cov = 0.52, 0.50
            elif is_long:
                min_score, min_cov = 0.42, 0.25
            else:
                min_score, min_cov = 0.46, 0.35
        else:
            # legacy (LLM) targets need stricter guards to avoid false positives
            if is_short:
                min_score, min_cov = 0.60, 0.65
            elif is_long:
                min_score, min_cov = 0.48, 0.35
            else:
                min_score, min_cov = 0.54, 0.48

        for win in (1, 2, 3, 4):  # allow 4-line union for messy handwriting
            for i in range(0, len(lines) - win + 1):
                combined_text, rect_raw = _best_window_match(lines, i, win)
                if not combined_text or not rect_raw:
                    continue

                score = _line_match_score(tgt, combined_text)
                cov = _token_coverage(tgt, combined_text)
                if score < min_score or cov < min_cov:
                    continue

                if score > best_score:
                    best_score = score
                    best_rect_raw = rect_raw

    if not best_rect_raw:
        return None

    rect_px = _scale_rect_by_extent(best_rect_raw, extent, pix_w, pix_h)

    x1, y1, x2, y2 = rect_px
    pad = 7 if best_is_long else 5
    final = (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(pix_w - 1, x2 + pad),
        min(pix_h - 1, y2 + pad),
    )
    if final[2] <= final[0] or final[3] <= final[1]:
        return None
    return final


# ============================================================
# ANNOTATION TARGETS (NEW: OCR-ANCHOR FIRST)
# ============================================================
def _build_annotation_candidates(a: Dict[str, Any]) -> List[Tuple[str, bool]]:
    """
    Returns list of (candidate_text, is_anchor).

    Priority:
    1) anchor_quote (verbatim from OCR) -> strongest
    2) anchor_keywords (optional) -> helpful
    3) legacy target/context combos (LLM-generated) -> fallback only
    """
    out: List[Tuple[str, bool]] = []

    anchor_quote = (a.get("anchor_quote") or a.get("anchorQuote") or "").strip()
    if anchor_quote:
        out.append((anchor_quote, True))
        # also try shorter anchor
        toks = anchor_quote.split()
        if len(toks) > 18:
            out.append((" ".join(toks[:18]), True))
        out.append((_keywords_only(anchor_quote, max_words=9), True))

    anchor_keywords = a.get("anchor_keywords") or a.get("anchorKeywords")
    if isinstance(anchor_keywords, list) and anchor_keywords:
        kw = " ".join([str(x).strip() for x in anchor_keywords if str(x).strip()])
        if kw:
            out.append((kw, True))

    # legacy fields
    target = (a.get("target_word_or_sentence") or "").strip()
    cb = (a.get("context_before") or "").strip()
    ca = (a.get("context_after") or "").strip()

    legacy = []
    if target:
        legacy.append(target)
    if cb and target:
        legacy.append((cb + " " + target).strip())
    if target and ca:
        legacy.append((target + " " + ca).strip())
    if cb and target and ca:
        legacy.append((cb + " " + target + " " + ca).strip())

    # include some shortened variants + keywords for legacy too
    for s in legacy:
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        out.append((s, False))
        toks = s.split()
        if len(toks) > 12:
            out.append((" ".join(toks[:12]), False))
        out.append((_keywords_only(s), False))

    # dedup preserving order
    seen = set()
    final: List[Tuple[str, bool]] = []
    for txt, is_anchor in out:
        k = txt.lower().strip()
        if k and k not in seen:
            seen.add(k)
            final.append((txt, is_anchor))
    return final


# ============================================================
# DEDUP / ASSIGNMENT HELPERS
# ============================================================
def _rect_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)

def _shift_rect(rect: Tuple[int, int, int, int], x_shift: int, y_shift: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    return (x1 + x_shift, y1 + y_shift, x2 + x_shift, y2 + y_shift)


# ============================================================
# DRAWING HELPERS
# ============================================================
_UNICODE_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "•": "*",
    "·": "-",
}


def _sanitize_text_for_render(text: str) -> str:
    """
    Make text safe for rendering when fonts lack certain Unicode glyphs.
    - Replaces common curly quotes/dashes/arrows/bullets with ASCII.
    - Normalizes and strips characters that can't be encoded to ASCII.
    """
    if not text:
        return ""
    s = text
    for k, v in _UNICODE_REPLACEMENTS.items():
        s = s.replace(k, v)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def _wrap_text_lines(text: str, font_scale: float, thickness: int, max_width_px: int) -> List[str]:
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    words = (_sanitize_text_for_render(text) or "").split()
    if not words:
        return []
    lines: List[str] = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, font_face, font_scale, thickness)
        if tw <= max_width_px or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _wrap_text_lines_with_word_break(text: str, font_scale: float, thickness: int, max_width_px: int) -> List[str]:
    """Like _wrap_text_lines but breaks long words that exceed max_width_px so they never overflow."""
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    words = (_sanitize_text_for_render(text) or "").split()
    if not words:
        return []
    lines: List[str] = []
    cur = ""
    for w in words:
        (w_w, _), _ = cv2.getTextSize(w, font_face, font_scale, thickness)
        if w_w <= max_width_px:
            test = (cur + " " + w).strip()
            (tw, _), _ = cv2.getTextSize(test, font_face, font_scale, thickness)
            if tw <= max_width_px or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        else:
            if cur:
                lines.append(cur)
                cur = ""
            while w:
                for k in range(len(w), 0, -1):
                    chunk = w[:k]
                    (cw, _), _ = cv2.getTextSize(chunk, font_face, font_scale, thickness)
                    if cw <= max_width_px:
                        lines.append(chunk)
                        w = w[k:]
                        break
                else:
                    lines.append(w[0])
                    w = w[1:]
    if cur:
        lines.append(cur)
    return lines


def _estimate_text_height(text: str, font_scale: float, thickness: int, max_width_px: int, line_gap: int = 8) -> int:
    lines = _wrap_text_lines(text, font_scale, thickness, max_width_px)
    if not lines:
        return 0
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return len(lines) * th + (len(lines) - 1) * line_gap

def _draw_wrapped_text(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    font_scale: float,
    thickness: int,
    max_width_px: int,
    color,
    line_gap: int = 8,
) -> int:
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_text_lines(text, font_scale, thickness, max_width_px)
    used = 0
    for ln in lines:
        (_, th), _ = cv2.getTextSize(ln, font_face, font_scale, thickness)
        cv2.putText(img, ln, (x, y + used + th), font_face, font_scale, color, thickness, cv2.LINE_AA)
        used += th + line_gap
    return used


def _draw_red_tick(
    img: np.ndarray,
    *,
    x: int,
    y: int,
    size: int,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 5,
) -> None:
    """
    Draw a simple red "tick/checkmark" on an OpenCV BGR image.

    Coordinate meaning:
    - (x, y) is the lower-left start of the tick.
    - `size` controls overall tick width/height.
    """
    s = max(10, int(size))
    t = max(1, int(thickness))

    p1 = (int(x), int(y))
    p2 = (int(x + 0.35 * s), int(y + 0.35 * s))
    p3 = (int(x + 1.10 * s), int(y - 0.45 * s))

    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)
    cv2.line(img, p2, p3, color, t, cv2.LINE_AA)


# --- LEFT MARGIN BOX LAYOUT (vertical only, fit to page) ---
LEFT_BOX_MIN_GAP = 12  # vertical gap between left-side improvement boxes
LEFT_BOX_TOP_PAD = 20
LEFT_BOX_BOTTOM_PAD = 20


def _fit_left_annotation_box(
    text: str,
    max_width_px: int,
    max_height_px: int,
    start_scale: float = 1.55,
    thickness: int = 2,
    line_gap: int = 18,
    top_pad: int = 20,
    bottom_pad: int = 20,
    min_scale: float = 0.55,
    font_step: float = 0.06,
) -> Tuple[float, List[str], int, bool]:
    """
    Fit single-block text (e.g. improvement bullet) into a left-side box: wrap by width,
    step down font until it fits, then truncate with "..." if still too tall at min_scale.
    Returns (font_scale, lines, box_h, overflow_truncated).
    """
    overflow_truncated = False
    font_s = start_scale
    max_content_h = max(1, max_height_px - top_pad - bottom_pad)

    while font_s >= min_scale:
        lines = _wrap_text_lines_with_word_break(text, font_s, thickness, max_width_px)
        box_h = _box_height_for_wrapped_lines(len(lines), font_s, thickness, line_gap, top_pad, bottom_pad)
        if box_h <= max_height_px:
            return (font_s, lines, box_h, False)
        font_s -= font_step

    font_s = max(min_scale, font_s)
    lines = _wrap_text_lines_with_word_break(text, font_s, thickness, max_width_px)
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_s, thickness)
    line_height = th + line_gap
    max_lines = max(0, (max_content_h + line_gap) // line_height) if line_height > 0 else 0
    if len(lines) > max_lines:
        overflow_truncated = True
        if max_lines <= 0:
            lines = ["..."]
        else:
            lines = lines[:max_lines]
            last = (lines[-1] or "").strip()
            lines[-1] = (last[:-3] + "...") if len(last) > 3 else "..."
    box_h = _box_height_for_wrapped_lines(len(lines), font_s, thickness, line_gap, top_pad, bottom_pad)
    return (font_s, lines, box_h, overflow_truncated)


# --- DYNAMIC LEFT BOX HELPERS ---
def _box_height_for_wrapped_lines(num_lines: int, font_scale: float, thickness: int, line_gap: int, top_pad: int, bottom_pad: int) -> int:
    if num_lines <= 0:
        return top_pad + bottom_pad
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return top_pad + bottom_pad + num_lines * th + (num_lines - 1) * line_gap

def _fit_text_box(
    text: str,
    max_width_px: int,
    max_height_px: int,
    start_scale: float,
    thickness: int,
    line_gap: int,
    top_pad: int = 20,
    bottom_pad: int = 20,
    min_scale: float = 0.72,
) -> Tuple[float, List[str], int]:
    font_s = start_scale
    while font_s >= min_scale:
        lines = _wrap_text_lines(text, font_s, thickness, max_width_px)
        box_h = _box_height_for_wrapped_lines(len(lines), font_s, thickness, line_gap, top_pad, bottom_pad)
        if box_h <= max_height_px:
            return font_s, lines, box_h
        font_s -= 0.06
    lines = _wrap_text_lines(text, min_scale, thickness, max_width_px)
    box_h = _box_height_for_wrapped_lines(len(lines), min_scale, thickness, line_gap, top_pad, bottom_pad)
    return min_scale, lines, box_h


# --- RIGHT-SIDE ANNOTATION BOX HELPERS ---
# Fixed padding for right-side boxes: top 28, header-body gap 36, bottom 8 (total 72)
_RIGHT_BOX_TOP_PAD = 28
_RIGHT_BOX_HEADER_BODY_GAP = 36
_RIGHT_BOX_BOTTOM_PAD = 8
_RIGHT_BOX_FIXED_PAD = _RIGHT_BOX_TOP_PAD + _RIGHT_BOX_HEADER_BODY_GAP + _RIGHT_BOX_BOTTOM_PAD  # 72


def _fit_right_annotation_box(
    header: str,
    body: str,
    box_w: int,
    max_box_h: int,
    font_start_header: float = 1.35,
    font_start_body: float = 1.25,
    thickness: int = 2,
    line_gap: int = 20,
    font_min: float = 0.45,
    font_step: float = 0.05,
) -> Tuple[float, float, List[str], List[str], int, bool]:
    """
    Fit header + body text inside a right-side annotation box: wrap to width, step down
    font until content fits, then optionally truncate body with ellipsis at font_min.
    Returns (header_scale, body_scale, header_lines, body_lines, box_h, overflow_truncated).
    Uses _wrap_text_lines_with_word_break so long headers/labels stay inside the box.
    """
    text_w = max(1, box_w - 28)  # 14px horizontal padding each side to avoid right-edge overflow
    overflow_truncated = False
    header_scale = font_start_header
    body_scale = font_start_body

    def _content_height(h_scale: float, b_scale: float, h_lines: List[str], b_lines: List[str]) -> int:
        h_h = _box_height_for_wrapped_lines(len(h_lines), h_scale, thickness, line_gap, 0, 0)
        b_h = _box_height_for_wrapped_lines(len(b_lines), b_scale, thickness, line_gap, 0, 0)
        return _RIGHT_BOX_FIXED_PAD + h_h + b_h

    # Step down font until content fits or we hit font_min; use word-break wrap for long labels
    while header_scale >= font_min and body_scale >= font_min:
        header_lines = _wrap_text_lines_with_word_break(header, header_scale, thickness, text_w)
        body_lines = _wrap_text_lines_with_word_break(body, body_scale, thickness, text_w)
        box_h = _content_height(header_scale, body_scale, header_lines, body_lines)
        if box_h <= max_box_h:
            return (header_scale, body_scale, header_lines, body_lines, box_h, False)
        header_scale -= font_step
        body_scale -= font_step

    header_scale = max(font_min, header_scale)
    body_scale = max(font_min, body_scale)
    header_lines = _wrap_text_lines_with_word_break(header, header_scale, thickness, text_w)
    body_lines = _wrap_text_lines_with_word_break(body, body_scale, thickness, text_w)
    h_h = _box_height_for_wrapped_lines(len(header_lines), header_scale, thickness, line_gap, 0, 0)
    body_max_h = max(0, max_box_h - _RIGHT_BOX_FIXED_PAD - h_h)
    (_, th), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, body_scale, thickness)
    line_height = th + line_gap
    max_body_lines = max(0, (body_max_h + line_gap) // line_height) if line_height > 0 else 0

    if len(body_lines) > max_body_lines:
        overflow_truncated = True
        if max_body_lines <= 0:
            body_lines = ["..."]
        else:
            body_lines = body_lines[:max_body_lines]
            last = (body_lines[-1] or "").strip()
            if len(last) > 3:
                body_lines[-1] = last[:-3] + "..."
            else:
                body_lines[-1] = "..."
    b_h = _box_height_for_wrapped_lines(len(body_lines), body_scale, thickness, line_gap, 0, 0)
    box_h = _RIGHT_BOX_FIXED_PAD + h_h + b_h
    return (header_scale, body_scale, header_lines, body_lines, box_h, overflow_truncated)


def _draw_lines_at(
    img: np.ndarray,
    x: int,
    y: int,
    lines: List[str],
    font_scale: float,
    thickness: int,
    color,
    line_gap: int = 8,
) -> int:
    """Draw pre-wrapped lines at (x, y). y is top of first line. Returns total height used.
    Returned height matches _box_height_for_wrapped_lines (no line_gap after last line).
    """
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    used = 0
    for i, ln in enumerate(lines):
        (_, th), _ = cv2.getTextSize(ln, font_face, font_scale, thickness)
        cv2.putText(img, ln, (x, y + used + th), font_face, font_scale, color, thickness, cv2.LINE_AA)
        used += th + (line_gap if i < len(lines) - 1 else 0)
    return used


# ============================================================
# PYMUPDF SPELLING ANNOTATION HELPERS
# ============================================================

def _norm_token_for_spelling(t: str) -> str:
    """Normalize a token for spelling error matching: lowercase, strip punctuation, collapse to alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _norm_phrase_tokens_for_spelling(phrase: str) -> List[str]:
    """Normalize phrase for spelling: split on whitespace, normalize each token, drop empty. Used for multi-word matching."""
    return [_norm_token_for_spelling(w) for w in (phrase or "").split() if _norm_token_for_spelling(w)]

def _bbox_to_rect_float(bbox: List) -> Optional[Tuple[float, float, float, float]]:
    """Convert polygon bbox to (x0,y0,x1,y1) with float precision."""
    if not bbox:
        return None
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

def _unit_scale_to_points(unit: str) -> float:
    """Convert Azure coordinate units to PyMuPDF points (1 inch = 72 points)."""
    u = (unit or "").lower()
    if u == "inch":
        return 72.0
    # pixel or other: assume already aligned
    return 1.0

def _word_rects_in_page_coords_fitz(page_info: Dict[str, Any]) -> List[Tuple[fitz.Rect, float, str]]:
    """Return list of (rect_in_points, confidence, text) for all words in page, sorted by reading order (y0, x0)."""
    scale = _unit_scale_to_points(page_info.get("unit", "pixel"))
    out = []
    for w in page_info.get("words", []) or []:
        poly = w.get("bbox") or []
        r = _bbox_to_rect_float(poly)
        if not r:
            continue
        x0, y0, x1, y1 = r
        rect_pts = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        out.append((rect_pts, float(w.get("confidence", 1.0) or 1.0), w.get("text", "")))
    out.sort(key=lambda item: (item[0].y0, item[0].x0))
    return out

# Token window expansion around anchor hit (tokens before/after) for phrase search
_SPELL_ANCHOR_WINDOW = 20


def _find_error_word_span_fitz(
    wordrects: List[Tuple[fitz.Rect, float, str]],
    error_text: str,
    anchor_quote: Optional[str] = None
) -> Tuple[Optional[fitz.Rect], str]:
    """Find the bounding box for error_text by matching normalized word sequences.
    Wordrects must be sorted by reading order (y0, x0). Uses page-level words only.
    If anchor_quote is provided, try a window around it first, then always fall back to full page.
    Returns (rect or None, match_method).
    """
    target = _norm_token_for_spelling(error_text)
    error_tokens = _norm_phrase_tokens_for_spelling(error_text)
    if not target and not error_tokens:
        return None, "not_found"

    tokens = [(_norm_token_for_spelling(w), r, w) for (r, _c, w) in wordrects]

    def search_in_range(start: int, end: int) -> Tuple[Optional[fitz.Rect], str]:
        r_end = min(end, len(tokens))
        # 1) Single-word exact
        for i in range(start, r_end):
            t, r, _ = tokens[i]
            if t == target:
                return r, "full_page_exact"
        # 2) Multi-word: consecutive token sequence (e.g. "false" then "belief")
        if len(error_tokens) >= 2:
            for i in range(start, r_end - len(error_tokens) + 1):
                if all(tokens[i + k][0] == error_tokens[k] for k in range(len(error_tokens))):
                    r_union = tokens[i][1]
                    for k in range(1, len(error_tokens)):
                        r_union = r_union | tokens[i + k][1]
                    return r_union, "full_page_exact"
        # 3) Multi-word: concatenated (legacy)
        for i in range(start, r_end):
            acc = ""
            r_union = None
            for j in range(i, min(i + 8, r_end)):
                acc += tokens[j][0]
                r_union = tokens[j][1] if r_union is None else (r_union | tokens[j][1])
                if acc == target:
                    return r_union, "full_page_exact"
                if len(acc) > len(target):
                    break
        return None, "not_found"

    if anchor_quote:
        anchor_norm = _norm_token_for_spelling(anchor_quote)
        anchor_phrase = _norm_phrase_tokens_for_spelling(anchor_quote)
        if anchor_norm or anchor_phrase:
            for i in range(len(tokens)):
                # Match anchor as phrase or as substring of concatenated
                acc = ""
                for j in range(i, min(i + 20, len(tokens))):
                    acc += tokens[j][0]
                    anchor_ok = (
                        (anchor_norm and (anchor_norm in acc or acc in anchor_norm))
                        or (anchor_phrase and j - i + 1 >= len(anchor_phrase)
                            and all(tokens[i + k][0] == anchor_phrase[k] for k in range(len(anchor_phrase))))
                    )
                    if anchor_ok:
                        window_start = max(0, i - _SPELL_ANCHOR_WINDOW)
                        window_end = min(len(tokens), j + 1 + _SPELL_ANCHOR_WINDOW)
                        found_rect, method = search_in_range(window_start, window_end)
                        if found_rect is not None:
                            return found_rect, "anchor_window"
                        break
                    if len(acc) > (len(anchor_norm or "") + 20):
                        break

    # Full-page fallback (always run if anchor didn't return)
    found_rect, method = search_in_range(0, len(tokens))
    if found_rect is not None:
        return found_rect, method
    return None, "not_found"


def _compute_spelling_box_geometry(
    word_rect: "fitz.Rect",
    correction_text: str,
    page_rect: "fitz.Rect",
    existing_boxes: Optional[List["fitz.Rect"]] = None,
) -> Tuple["fitz.Rect", float, str]:
    """
    Compute spelling annotation box above (or below) the word rect, with font size fitted to box width.
    If existing_boxes is provided, shift this box to avoid overlapping them (same-page collision avoidance).
    Returns (box_rect, font_size, placement) where placement is "above" or "below".
    """
    existing_boxes = existing_boxes or []
    if not correction_text:
        return fitz.Rect(0, 0, 0, 0), float(SPELL_BOX_FONT_MIN), "above"
    safe_text = _sanitize_text_for_render("→ " + (correction_text or ""))
    if not safe_text:
        return fitz.Rect(0, 0, 0, 0), float(SPELL_BOX_FONT_MIN), "above"
    word_w = word_rect.width
    word_h = word_rect.height
    box_w = max(word_w + 2 * SPELL_BOX_PADDING_PT, SPELL_BOX_MIN_WIDTH_PT)
    box_w = min(box_w, page_rect.width * 0.5)
    font_size = float(SPELL_BOX_FONT_START)
    for _ in range(20):
        tw = fitz.get_text_length(safe_text, fontname="hebo", fontsize=font_size)
        if tw <= box_w - 2 * SPELL_BOX_PADDING_PT or font_size <= SPELL_BOX_FONT_MIN:
            break
        font_size = max(SPELL_BOX_FONT_MIN, font_size - 0.5)
    line_height_pt = font_size * 1.2
    box_h = 2 * SPELL_BOX_PADDING_PT + line_height_pt
    page_top = page_rect.y0
    page_bottom = page_rect.y1
    page_left = page_rect.x0
    page_right = page_rect.x1
    center_x = word_rect.x0 + word_rect.width / 2.0
    box_x0 = center_x - box_w / 2.0
    box_x1 = box_x0 + box_w
    if box_x0 < page_left:
        box_x0 = page_left
        box_x1 = box_x0 + box_w
    if box_x1 > page_right:
        box_x1 = page_right
        box_x0 = box_x1 - box_w
    placement = "above"
    box_y1_above = word_rect.y0 - SPELL_BOX_GAP_PT
    box_y0_above = box_y1_above - box_h
    if box_y0_above >= page_top:
        box_y0 = box_y0_above
        box_y1 = box_y1_above
        placement = "above"
    else:
        box_y0 = word_rect.y1 + SPELL_BOX_GAP_PT
        box_y1 = box_y0 + box_h
        placement = "below"
    if box_y1 > page_bottom:
        box_y1 = page_bottom
        box_y0 = box_y1 - box_h
    if box_y0 < page_top:
        box_y0 = page_top
        box_y1 = box_y0 + box_h
    box_rect = fitz.Rect(box_x0, box_y0, box_x1, box_y1)
    gap = SPELL_BOX_GAP_PT
    for _ in range(10):
        overlap = False
        for ex in existing_boxes:
            if (
                box_rect.y1 + gap > ex.y0
                and box_rect.y0 - gap < ex.y1
                and box_rect.x1 > ex.x0
                and box_rect.x0 < ex.x1
            ):
                overlap = True
                break
        if not overlap:
            break
        box_y0 -= box_h + gap
        box_y1 -= box_h + gap
        if box_y0 < page_top:
            box_y0 = word_rect.y1 + SPELL_BOX_GAP_PT
            box_y1 = box_y0 + box_h
            placement = "below"
        box_rect = fitz.Rect(box_x0, box_y0, box_x1, box_y1)
    return box_rect, font_size, placement


def _draw_single_spelling_annotation(
    page: "fitz.Page",
    word_rect: "fitz.Rect",
    correction_text: str,
    page_rect: "fitz.Rect",
    existing_boxes: Optional[List["fitz.Rect"]] = None,
) -> Tuple[Optional["fitz.Rect"], Optional["fitz.Rect"], str]:
    """
    Draw one spelling annotation: box (above or below word), connector line, and red correction text.
    Returns (word_rect, box_rect, status) for debug where status is "drawn" or "clamped".
    """
    box_rect, font_size, placement = _compute_spelling_box_geometry(
        word_rect, correction_text, page_rect, existing_boxes
    )
    if box_rect.is_empty or box_rect.width <= 0 or box_rect.height <= 0:
        return word_rect, None, "clamped"
    safe_text = _sanitize_text_for_render("→ " + (correction_text or ""))
    if not safe_text:
        return word_rect, box_rect, "drawn"
    word_center_x = word_rect.x0 + word_rect.width / 2.0
    word_center_y = (word_rect.y0 + word_rect.y1) / 2.0
    box_center_x = box_rect.x0 + box_rect.width / 2.0
    if placement == "above":
        conn_start = fitz.Point(box_center_x, box_rect.y1)
        conn_end = fitz.Point(word_center_x, word_rect.y0)
    else:
        conn_start = fitz.Point(box_center_x, box_rect.y0)
        conn_end = fitz.Point(word_center_x, word_rect.y1)
    shape = page.new_shape()
    shape.draw_line(conn_start, conn_end)
    shape.finish(color=SPELL_COLOR, width=SPELL_CONNECTOR_WIDTH)
    shape.commit()
    page.draw_rect(box_rect, color=SPELL_COLOR, fill=(1, 1, 1), width=1.0)
    text_y = box_rect.y0 + SPELL_BOX_PADDING_PT + font_size * 0.3
    text_x = box_rect.x0 + SPELL_BOX_PADDING_PT
    page.insert_text(
        fitz.Point(text_x, text_y),
        safe_text,
        fontsize=font_size,
        color=SPELL_COLOR,
        fontname="hebo",
    )
    if DEBUG_SPELL_DRAW:
        print(
            "SPELL_DRAW: word_rect=(%.1f,%.1f,%.1f,%.1f) box_rect=(%.1f,%.1f,%.1f,%.1f) font=%.1f placement=%s"
            % (word_rect.x0, word_rect.y0, word_rect.x1, word_rect.y1, box_rect.x0, box_rect.y0, box_rect.x1, box_rect.y1, font_size, placement)
        )
    return word_rect, box_rect, "drawn"


def _add_spelling_annotations_to_pdf(
    pil_images: List[Image.Image],
    ocr_data: Dict[str, Any],
    spelling_errors: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Take PIL images (already annotated with essay feedback) and add PyMuPDF-based
    spelling annotations directly on the PDF. Saves result to output_path.
    """
    if not pil_images or not spelling_errors:
        # Just save PIL images as PDF
        if pil_images:
            pil_images[0].save(output_path, save_all=True, append_images=pil_images[1:])
        return
    
    # Create PDF from PIL images
    pdf_doc = fitz.open()
    for img in pil_images:
        # Convert PIL to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Create page from image
        img_doc = fitz.open(stream=img_bytes, filetype="png")
        pdf_doc.insert_pdf(img_doc)
        img_doc.close()
    
    # Now add spelling annotations using PyMuPDF
    pages_data = ocr_data.get("pages", [])
    page_boxes: Dict[int, List[fitz.Rect]] = {}
    for error in spelling_errors:
        page_num = error.get("page", 1) - 1  # Convert to 0-indexed
        if page_num < 0 or page_num >= len(pdf_doc):
            continue
        
        page = pdf_doc[page_num]
        page_info = pages_data[page_num] if page_num < len(pages_data) else {}
        
        error_text = error.get("error_text", "")
        correction = error.get("correction", "")
        anchor_quote = error.get("anchor_quote")
        
        if not error_text or not correction:
            continue
        
        # Get word rectangles from OCR
        wordrects = _word_rects_in_page_coords_fitz(page_info)
        if not wordrects:
            continue
        
        # Find the error location (wordrects already sorted by reading order)
        rect, _match_method = _find_error_word_span_fitz(wordrects, error_text, anchor_quote)
        if not rect:
            continue
        
        # Draw red rectangle around incorrect word
        page.draw_rect(rect, color=SPELL_COLOR, width=2.0)
        # Dedicated spelling placement: box above word, connector, red correction text
        page_rect = page.rect
        existing_on_page = page_boxes.get(page_num, [])
        _wr, box_rect, _ = _draw_single_spelling_annotation(
            page, rect, correction, page_rect, existing_boxes=existing_on_page
        )
        if box_rect and not box_rect.is_empty:
            page_boxes.setdefault(page_num, []).append(box_rect)
    
    # Save the annotated PDF
    pdf_doc.save(output_path)
    pdf_doc.close()


def add_spelling_annotations_to_merged_pdf(
    output_pdf_path: str,
    ocr_data: Dict[str, Any],
    spelling_errors: List[Dict[str, Any]],
    answer_start_page_0indexed: int,
    first_answer_page_1based: int,
    page_transforms: List[Tuple[float, float, float]],
) -> List[Dict[str, Any]]:
    """
    Add spelling annotations to the already-merged output PDF (report + answer pages).
    Only touches answer pages. Each entry in page_transforms is (scale_pt, x0, y0) mapping
    source page coordinates to the content area on the corresponding merged answer page.
    Returns a list of placement records for debug: page, error_text, correction, matched_rect, box_rect, status.
    """
    placement_results: List[Dict[str, Any]] = []
    if not spelling_errors or not page_transforms:
        return placement_results
    try:
        pdf_doc = fitz.open(output_pdf_path)
    except Exception:
        return placement_results
    pages_data = ocr_data.get("pages", [])
    page_boxes: Dict[int, List[fitz.Rect]] = {}
    try:
        for error in spelling_errors:
            src_page_1 = int(error.get("page", 1))
            page_off = src_page_1 - first_answer_page_1based
            if page_off < 0 or page_off >= len(page_transforms):
                continue
            merged_idx = answer_start_page_0indexed + page_off
            if merged_idx < 0 or merged_idx >= len(pdf_doc):
                continue
            scale_pt, x0, y0 = page_transforms[page_off]
            page = pdf_doc[merged_idx]
            page_info_idx = src_page_1 - 1
            if page_info_idx < 0 or page_info_idx >= len(pages_data):
                continue
            page_info = pages_data[page_info_idx]
            error_text = error.get("error_text", "")
            correction = error.get("correction", "")
            anchor_quote = error.get("anchor_quote")
            if not error_text or not correction:
                continue
            wordrects = _word_rects_in_page_coords_fitz(page_info)
            anchor_quote_present = anchor_quote is not None and bool((anchor_quote or "").strip())
            if not wordrects:
                placement_results.append({
                    "page": src_page_1,
                    "error_text": error_text,
                    "correction": correction,
                    "anchor_quote_present": anchor_quote_present,
                    "anchor_match_used": False,
                    "match_method": "not_found",
                    "matched_rect": None,
                    "box_rect": None,
                    "status": "not_found",
                })
                continue
            rect, match_method = _find_error_word_span_fitz(wordrects, error_text, anchor_quote)
            if not rect:
                placement_results.append({
                    "page": src_page_1,
                    "error_text": error_text,
                    "correction": correction,
                    "anchor_quote_present": anchor_quote_present,
                    "anchor_match_used": False,
                    "match_method": match_method,
                    "matched_rect": None,
                    "box_rect": None,
                    "status": "not_found",
                })
                if DEBUG_SPELL_MATCH:
                    print("[SPELL] page=%s err=%r method=%s matched=False" % (src_page_1, error_text, match_method))
                continue
            merged_rect = fitz.Rect(
                x0 + rect.x0 * scale_pt,
                y0 + rect.y0 * scale_pt,
                x0 + rect.x1 * scale_pt,
                y0 + rect.y1 * scale_pt,
            )
            page.draw_rect(merged_rect, color=SPELL_COLOR, width=2.0)
            page_rect = page.rect
            existing_on_page = page_boxes.get(merged_idx, [])
            _wr, box_rect, status = _draw_single_spelling_annotation(
                page, merged_rect, correction, page_rect, existing_boxes=existing_on_page
            )
            if box_rect and not box_rect.is_empty:
                page_boxes.setdefault(merged_idx, []).append(box_rect)
            placement_results.append({
                "page": src_page_1,
                "error_text": error_text,
                "correction": correction,
                "anchor_quote_present": anchor_quote_present,
                "anchor_match_used": (match_method == "anchor_window"),
                "match_method": match_method,
                "matched_rect": [merged_rect.x0, merged_rect.y0, merged_rect.x1, merged_rect.y1],
                "box_rect": [box_rect.x0, box_rect.y0, box_rect.x1, box_rect.y1] if box_rect else None,
                "status": status,
            })
            if DEBUG_SPELL_MATCH:
                print("[SPELL] page=%s err=%r method=%s matched=True" % (src_page_1, error_text, match_method))
    finally:
        try:
            pdf_doc.save(output_pdf_path, incremental=False)
        except Exception:
            pass
        pdf_doc.close()
    return placement_results


# ============================================================
# MAIN FUNCTION
# ============================================================
def annotate_pdf_essay_pages(
    pdf_path: str,
    ocr_data: Dict[str, Any],
    structure: Dict[str, Any],
    grading: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    page_suggestions: Optional[List[Dict[str, Any]]] = None,
    spelling_errors: Optional[List[Dict[str, Any]]] = None,
    *,
    dpi: int = 220,
    debug_draw_ocr_boxes: bool = False,
    debug_print_fail_samples: bool = True,
    dedup_iou_threshold: float = 0.35,
    topk_candidates_per_ann: int = 6,
    max_callouts_per_page: int = 12,
) -> List[Image.Image]:
    """
    Returns list of annotated PIL images (one per page).

    Behavior:
    - Digital text found? -> match in PDF text first.
    - Else OCR matching:
        - prefer anchor_quote candidates (if present)
        - else legacy candidates
    - If no rect found -> page-level box only (no arrow/highlight)
    - Spelling errors are annotated inline directly on the page with red boxes and corrections
    """
    page_suggestions = page_suggestions or []
    spelling_errors = spelling_errors or []
    doc = fitz.open(pdf_path)

    # Render pages
    pil_pages: List[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pil_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))

    # Map OCR pages by page number (1-based)
    ocr_pages_by_num: Dict[int, Dict[str, Any]] = {}
    for p in (ocr_data.get("pages", []) or []):
        pn = p.get("page_number")
        if pn is None:
            pn = p.get("pageNumber")
        # handle string page numbers
        if isinstance(pn, str) and pn.strip().isdigit():
            pn = int(pn.strip())
        if isinstance(pn, int):
            ocr_pages_by_num[pn] = p

    # Suggestions per page
    suggestions_by_page: Dict[int, List[str]] = {}
    for s in page_suggestions:
        pno = s.get("page")
        sug = s.get("suggestions") or []
        if isinstance(pno, int) and pno >= 1:
            suggestions_by_page[pno] = [str(x) for x in sug if str(x).strip()]

    RED = (0, 0, 255)
    annotated_pages: List[Image.Image] = []

    for page_idx, pil_img in enumerate(pil_pages):
        page_number = page_idx + 1
        page_obj = doc[page_idx]

        orig_cv = np.array(pil_img)[:, :, ::-1].copy()
        orig_h, orig_w, _ = orig_cv.shape

        page_ocr = ocr_pages_by_num.get(page_number, {})

        # DEBUG header
        print(f"\n=== PAGE {page_number} DEBUG ===")
        ocr_line_count = len(page_ocr.get("lines", [])) if page_ocr else 0
        print(f"  OCR lines found: {ocr_line_count}")
        extent = _compute_page_extent(page_ocr) if page_ocr else None
        print(f"  Page extent: {extent}")

        # Canvas with margins (equal spacing on both sides of the essay body)
        # Previously: left=65% and right=35% of essay width, which left a visibly larger gap on the left.
        side_margin_ratio = 0.40
        left_width = int(side_margin_ratio * orig_w)
        right_width = int(side_margin_ratio * orig_w)
        new_w = left_width + orig_w + right_width
        y_offset = 0
        margin_px = int(0.03 * orig_w)

        canvas = np.full((orig_h, new_w, 3), 255, dtype=np.uint8)
        canvas[y_offset:y_offset + orig_h, left_width:left_width + orig_w] = orig_cv

        # ------------------------------------------------------------
        # RED TICK MARK (on essay body) - one per page, near lower area
        # ------------------------------------------------------------
        tick_size = max(26, int(orig_w * 0.05))
        tick_thickness = max(3, int(orig_w * 0.004))
        # Place slightly above bottom (not too low) and inside the essay body region
        tick_x = left_width + int(orig_w * 0.08)
        tick_y = y_offset + int(orig_h * 0.82)
        # Constrain inside visible page bounds
        tick_x = max(left_width + 5, min(tick_x, left_width + orig_w - tick_size - 5))
        tick_y = max(5 + tick_size, min(tick_y, orig_h - margin_px - 5))
        _draw_red_tick(canvas, x=tick_x, y=tick_y, size=tick_size, thickness=tick_thickness)

        # LEFT MARGIN: Improvements (single column, vertical stack, fit to page)
        cv2.putText(
            canvas,
            _sanitize_text_for_render(f"Page {page_number} - Improvements"),
            (margin_px, y_offset + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        left_pad = 10
        col_w = left_width - 2 * margin_px  # full width, single column only
        max_bottom_left = orig_h - margin_px
        y_start = y_offset + 120
        thick = 2
        line_g = 18

        bullets = [("- " + str(b).strip(),) for b in suggestions_by_page.get(page_number, [])[:6] if str(b).strip()]
        left_fitted: List[Dict[str, Any]] = []

        # Pass 1: compute fitted box for each bullet, stack vertically (no side-by-side)
        y_cur = y_start
        for (bullet_full,) in bullets:
            max_avail_h = max(60, max_bottom_left - y_cur - LEFT_BOX_MIN_GAP)
            font_s, wrapped_lines, box_h, overflow_truncated = _fit_left_annotation_box(
                bullet_full,
                max_width_px=col_w - 2 * left_pad,
                max_height_px=max_avail_h,
                start_scale=1.55,
                thickness=thick,
                line_gap=line_g,
                top_pad=LEFT_BOX_TOP_PAD,
                bottom_pad=LEFT_BOX_BOTTOM_PAD,
                min_scale=0.55,
            )
            by1 = y_cur
            by2 = y_cur + box_h
            if by2 > max_bottom_left:
                by2 = max_bottom_left
                by1 = by2 - box_h
            left_fitted.append({
                "bx1": margin_px,
                "bx2": margin_px + col_w,
                "by1": by1,
                "by2": by2,
                "lines": wrapped_lines,
                "font_s": font_s,
                "box_h": box_h,
                "overflow_truncated": overflow_truncated,
            })
            y_cur = by2 + LEFT_BOX_MIN_GAP
            if y_cur >= max_bottom_left:
                break

        # Pass 2: draw boxes and text from finalized geometry (no extra line_gap after last line)
        for f in left_fitted:
            cv2.rectangle(canvas, (f["bx1"], f["by1"]), (f["bx2"], f["by2"]), (0, 0, 0), 2)
            _draw_lines_at(
                canvas,
                f["bx1"] + left_pad,
                f["by1"] + LEFT_BOX_TOP_PAD,
                f["lines"],
                f["font_s"],
                thick,
                (0, 0, 0),
                line_gap=line_g,
            )

        # OPTIONAL DEBUG: draw OCR line boxes
        if debug_draw_ocr_boxes and page_ocr and page_ocr.get("lines"):
            if extent:
                for ln in page_ocr.get("lines") or []:
                    pts = _poly_to_points_generic(_line_polygon_any(ln))
                    r = _points_to_rect(pts)
                    if not r:
                        continue
                    rr = _scale_rect_by_extent(r, extent, orig_w, orig_h)
                    rr = _shift_rect(rr, left_width, y_offset)
                    cv2.rectangle(canvas, (rr[0], rr[1]), (rr[2], rr[3]), (190, 190, 190), 1)

        # NOTE: Spelling/grammar errors are added using PyMuPDF after PIL images are created
        # See _add_spelling_annotations_to_pdf_pages() function called at the end

        # Build callouts for this page
        anns = [a for a in annotations if a.get("page") == page_number][:max_callouts_per_page]
        print(f"  Annotations for this page: {len(anns)}")

        callout_items: List[Dict[str, Any]] = []

        for idx, a in enumerate(anns):
            # Matching/anchors disabled: skip detailed processing
            continue
            a_type = (a.get("type") or "").strip()
            rubric_point = (a.get("rubric_point") or "").strip()
            comment = (a.get("comment") or "").strip()
            correction = (a.get("correction") or "").strip()
            anchor_quote = (a.get("anchor_quote") or "").strip()

            header = f"[{a_type}] {rubric_point}".strip()
            body = (comment + (f"  Fix: {correction}" if correction else "")).strip()
            if a_type != "grammar_language" and correction:
                body = (comment + ("  Suggestion: " + correction)).strip()

            candidates = _build_annotation_candidates(a)
            
            # DEBUG: Show what we're trying to match
            if not anchor_quote:
                first_cand = candidates[0][0] if candidates else ""
                print(f"    [{idx+1}] ❌ has_anchor=False | first_candidate={first_cand[:60]}")
            else:
                print(f"    [{idx+1}] ✓ has_anchor=True | anchor_quote={anchor_quote[:60]}")

            match_candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []

            # attempt matching with best-first candidates
            for cand_text, is_anchor in candidates:
                # 1) PDF text match (best for digital PDFs)
                rect_pdf = _find_match_rect_in_pdf_text(page_obj, orig_w, orig_h, cand_text)
                if rect_pdf:
                    match_candidates.append((0.95 if is_anchor else 0.90, rect_pdf))
                    # PDF text hit is strong enough; don't waste time
                    continue

                # 2) OCR match
                if page_ocr:
                    rect_ocr = _find_best_match_rect_from_ocr(
                        page_ocr,
                        cand_text,
                        orig_w,
                        orig_h,
                        prefer_anchor=is_anchor,
                    )
                    if rect_ocr:
                        # anchors get higher base confidence
                        match_candidates.append((0.80 if is_anchor else 0.65, rect_ocr))

            # keep top-K unique rects
            match_candidates.sort(key=lambda x: x[0], reverse=True)

            uniq: List[Tuple[float, Tuple[int, int, int, int]]] = []
            for sc, rr in match_candidates:
                if not uniq:
                    uniq.append((sc, rr))
                else:
                    if all(_rect_iou(rr, u[1]) < 0.70 for u in uniq):
                        uniq.append((sc, rr))
                if len(uniq) >= topk_candidates_per_ann:
                    break

            callout_items.append({
                "ann": a,
                "header": header,
                "body": body,
                "cands": uniq,
                "has_anchor": any(is_anchor for _, is_anchor in candidates),
                "primary_candidate_preview": candidates[0][0] if candidates else "",
            })

        # GLOBAL ASSIGNMENT + page-level fallback
        used_rects: List[Tuple[int, int, int, int]] = []
        resolved_callouts: List[Dict[str, Any]] = []

        failed_examples = 0
        for item in callout_items:
            chosen_rect = None
            chosen_score = 0.0

            for sc, rr in item["cands"]:
                if not used_rects:
                    chosen_rect = rr
                    chosen_score = sc
                    break
                if all(_rect_iou(rr, ur) < dedup_iou_threshold for ur in used_rects):
                    chosen_rect = rr
                    chosen_score = sc
                    break

            if chosen_rect is None and item["cands"]:
                chosen_score, chosen_rect = item["cands"][0]

            if chosen_rect:
                used_rects.append(chosen_rect)

            # shift into canvas coords (center page has left margin offset)
            final_rect = _shift_rect(chosen_rect, left_width, y_offset) if chosen_rect else None

            # If no rect: make it explicit page-level feedback (so it doesn't look "broken")
            is_page_level = final_rect is None
            header2 = item["header"]
            if is_page_level:
                header2 = header2 + " (page-level)"

            resolved_callouts.append({
                "rect": final_rect,
                "header": header2,
                "body": item["body"],
                "y_sort": final_rect[1] if final_rect else 10**9,
                "score": chosen_score,
                "page_level": is_page_level,
            })

            if is_page_level and debug_print_fail_samples and failed_examples < 3:
                failed_examples += 1
                print("  ❌ Unmatched annotation sample:")
                print(f"     has_anchor={item['has_anchor']}")
                print(f"     candidate_preview={item['primary_candidate_preview'][:120]}")

        # RIGHT MARGIN LAYOUT — two-pass: compute geometry then draw (no overlap, text inside box).
        box_w = int(right_width - 2 * margin_px)
        gap = 12
        max_bottom = orig_h - margin_px
        avail_h = max(1, max_bottom - margin_px)
        l_gap = 20

        # Pass 1: build resolved_callouts (header, body, y_sort).
        resolved_callouts = []
        for idx, a in enumerate(anns):
            a_type = (a.get("type") or "").strip()
            rubric_point = (a.get("rubric_point") or "").strip()
            comment = (a.get("comment") or "").strip()
            correction = (a.get("correction") or "").strip()
            header = f"[{a_type}] {rubric_point}".strip()
            body = (comment + (f"  Fix: {correction}" if correction else "")).strip()
            if a_type != "grammar_language" and correction:
                body = (comment).strip()
            resolved_callouts.append({
                "rect": None,
                "header": header,
                "body": body,
                "y_sort": idx,
                "page_level": True,
            })
        resolved_callouts.sort(key=lambda x: x["y_sort"])

        if not resolved_callouts:
            annotated_pages.append(Image.fromarray(canvas[:, :, ::-1]))
            continue

        # Pass 2: fit text per box, then resolve collisions and clamp to page.
        tentative_max_h = max(80, (avail_h - gap * (len(resolved_callouts) - 1)) // len(resolved_callouts))
        fitted: List[Dict[str, Any]] = []
        for item in resolved_callouts:
            header_scale, body_scale, header_lines, body_lines, box_h, overflow_truncated = _fit_right_annotation_box(
                item["header"],
                item["body"],
                box_w,
                tentative_max_h,
                thickness=2,
                line_gap=l_gap,
            )
            fitted.append({
                "header_lines": header_lines,
                "body_lines": body_lines,
                "header_scale": header_scale,
                "body_scale": body_scale,
                "box_h": box_h,
                "overflow_truncated": overflow_truncated,
            })

        # Collision resolution: desired_y for first = margin_px, rest = previous by2 + gap; then push down.
        min_gap = RIGHT_BOX_MIN_GAP
        desired_by1 = margin_px
        for i in range(len(fitted)):
            f = fitted[i]
            by1 = int(desired_by1)
            by2 = int(by1 + f["box_h"])
            if by2 > max_bottom:
                by2 = max_bottom
                by1 = max(margin_px, by2 - f["box_h"])
                # If clamping causes overlap with previous, shrink this box and re-fit
                if i > 0 and by1 < fitted[i - 1].get("by2", 0) + min_gap:
                    shrink_max_h = max(60, (max_bottom - margin_px) - (fitted[i - 1].get("by2", 0) + min_gap))
                    if shrink_max_h >= 60:
                        header_scale2, body_scale2, header_lines2, body_lines2, box_h2, overflow_trunc2 = _fit_right_annotation_box(
                            resolved_callouts[i]["header"],
                            resolved_callouts[i]["body"],
                            box_w,
                            shrink_max_h,
                            thickness=2,
                            line_gap=l_gap,
                        )
                        fitted[i] = {
                            "header_lines": header_lines2,
                            "body_lines": body_lines2,
                            "header_scale": header_scale2,
                            "body_scale": body_scale2,
                            "box_h": box_h2,
                            "overflow_truncated": overflow_trunc2,
                        }
                        by1 = fitted[i - 1].get("by2", margin_px) + min_gap
                        by2 = by1 + fitted[i]["box_h"]
                        if by2 > max_bottom:
                            by2 = max_bottom
                            by1 = max(margin_px, by2 - fitted[i]["box_h"])
            f["by1"] = by1
            f["by2"] = by2
            desired_by1 = by2 + gap

        # Enforce minimum gap: push down any box that overlaps the previous.
        for i in range(1, len(fitted)):
            prev_bottom = fitted[i - 1]["by2"]
            need_y = prev_bottom + min_gap
            if fitted[i]["by1"] < need_y:
                by1 = need_y
                by2 = by1 + fitted[i]["box_h"]
                if by2 > max_bottom:
                    by2 = max_bottom
                    by1 = max(margin_px, by2 - fitted[i]["box_h"])
                fitted[i]["by1"] = by1
                fitted[i]["by2"] = by2
        # If clamping created overlap, shrink overlapping boxes and recompute.
        for i in range(1, len(fitted)):
            prev_bottom = fitted[i - 1]["by2"]
            if fitted[i]["by1"] < prev_bottom + min_gap:
                shrink_max_h = max(60, max_bottom - (prev_bottom + min_gap))
                header_scale2, body_scale2, header_lines2, body_lines2, box_h2, overflow_trunc2 = _fit_right_annotation_box(
                    resolved_callouts[i]["header"],
                    resolved_callouts[i]["body"],
                    box_w,
                    shrink_max_h,
                    thickness=2,
                    line_gap=l_gap,
                )
                fitted[i] = {
                    "header_lines": header_lines2,
                    "body_lines": body_lines2,
                    "header_scale": header_scale2,
                    "body_scale": body_scale2,
                    "box_h": box_h2,
                    "overflow_truncated": overflow_trunc2,
                }
                by1 = prev_bottom + min_gap
                by2 = by1 + box_h2  # box_h2 <= shrink_max_h so by2 <= max_bottom
                fitted[i]["by1"] = by1
                fitted[i]["by2"] = by2

        if DEBUG_RIGHT_BOX_LAYOUT:
            for i, f in enumerate(fitted):
                prev_y1 = fitted[i - 1]["by2"] if i > 0 else margin_px
                gap_used = f["by1"] - prev_y1
                overlap_adj = max(0, RIGHT_BOX_MIN_GAP - gap_used) if i > 0 else 0
                print(f"  right_box[{i}] y0={f['by1']} y1={f['by2']} height={f['box_h']} "
                      f"font_header={f['header_scale']:.2f} font_body={f['body_scale']:.2f} "
                      f"overflow_truncated={f['overflow_truncated']} overlap_adjustment_px={overlap_adj}")

        # Pass 3: draw using finalized geometry and precomputed lines only.
        bx1 = left_width + orig_w + margin_px
        bx2 = bx1 + box_w
        for i, item in enumerate(resolved_callouts):
            f = fitted[i]
            by1, by2 = f["by1"], f["by2"]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            header_y = by1 + _RIGHT_BOX_TOP_PAD
            h_drawn = _draw_lines_at(
                canvas, bx1 + 12, header_y, f["header_lines"],
                f["header_scale"], 2, (0, 0, 255), line_gap=l_gap,
            )
            body_y = by1 + _RIGHT_BOX_TOP_PAD + h_drawn + _RIGHT_BOX_HEADER_BODY_GAP
            _draw_lines_at(
                canvas, bx1 + 12, body_y, f["body_lines"],
                f["body_scale"], 2, (0, 0, 0), line_gap=l_gap,
            )

        annotated_pages.append(Image.fromarray(canvas[:, :, ::-1]))

    return annotated_pages
