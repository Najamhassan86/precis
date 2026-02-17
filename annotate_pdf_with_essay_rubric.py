# annotate_pdf_with_essay_rubric_fixed.py
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


# ============================================================
# PYMUPDF SPELLING ANNOTATION HELPERS
# ============================================================

def _norm_token_for_spelling(t: str) -> str:
    """Normalize a token for spelling error matching."""
    return re.sub(r"[^a-z0-9]", "", t.lower())

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
    """Return list of (rect_in_points, confidence, text) for all words in page."""
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
    return out

def _find_error_word_span_fitz(
    wordrects: List[Tuple[fitz.Rect, float, str]],
    error_text: str,
    anchor_quote: Optional[str] = None
) -> Optional[fitz.Rect]:
    """Find the bounding box for error_text by matching normalized word sequences."""
    target = _norm_token_for_spelling(error_text)
    if not target:
        return None

    tokens = [(_norm_token_for_spelling(w), r, c) for (r, c, w) in wordrects]
    
    # Single word match
    for t, r, _ in tokens:
        if t == target:
            return r

    # Multi-word span (join consecutive)
    for i in range(len(tokens)):
        acc = ""
        r_union = None
        for j in range(i, min(i + 6, len(tokens))):
            acc += tokens[j][0]
            r_union = tokens[j][1] if r_union is None else (r_union | tokens[j][1])
            if acc == target:
                return r_union
            if len(acc) > len(target):
                break
    
    return None

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
        
        # Find the error location
        rect = _find_error_word_span_fitz(wordrects, error_text, anchor_quote)
        if not rect:
            continue
        
        # Draw red rectangle around error
        page.draw_rect(rect, color=(0.8, 0, 0), width=2.0)
        
        # Draw correction text above the error
        correction_text = _sanitize_text_for_render(f"→ {correction}")
        if not correction_text:
            continue
        text_width = fitz.get_text_length(correction_text, fontname="hebo", fontsize=10)
        
        # White background for correction text
        bg_rect = fitz.Rect(
            rect.x0 - 2,
            rect.y0 - 16,
            rect.x0 + text_width + 4,
            rect.y0 - 2
        )
        page.draw_rect(bg_rect, color=(0.8, 0, 0), fill=(1, 1, 1), width=1.0)
        
        # Insert correction text
        text_point = fitz.Point(rect.x0, rect.y0 - 4)
        page.insert_text(
            text_point,
            correction_text,
            fontsize=10,
            color=(0.8, 0, 0),
            fontname="hebo"
        )
    
    # Save the annotated PDF
    pdf_doc.save(output_path)
    pdf_doc.close()


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
        side_margin_ratio = 0.35
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

        # LEFT MARGIN: Improvements
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
        col_gap = 14
        # When the left margin is narrower (after making margins equal),
        # using 2 columns makes boxes too skinny and causes excessive wrapping.
        # Use 1 column for narrow margins; keep 2 columns for wide margins.
        max_cols = 1 if left_width < int(0.50 * orig_w) else 2
        col_w = (left_width - 2 * margin_px - (max_cols - 1) * col_gap) // max_cols
        col_x = margin_px
        col_idx = 0
        y_cur = y_offset + 120

        for bullet in suggestions_by_page.get(page_number, [])[:6]:
            bullet_text = str(bullet).strip()
            if not bullet_text:
                continue
            bullet_full = "- " + bullet_text

            thick = 2
            line_g = 18
            top_pad = 20
            bottom_pad = 20

            remaining_h = (orig_h - margin_px) - y_cur
            if remaining_h < 160:
                col_idx += 1
                if col_idx >= max_cols:
                    col_idx = max_cols - 1
                    y_cur = y_offset + 120
                col_x = margin_px + col_idx * (col_w + col_gap)

            font_s, wrapped_lines, box_h = _fit_text_box(
                bullet_full,
                max_width_px=col_w - 2 * left_pad,
                max_height_px=min((orig_h - margin_px) - y_cur, 1200),
                start_scale=1.55,
                thickness=thick,
                line_gap=line_g,
                top_pad=top_pad,
                bottom_pad=bottom_pad,
                min_scale=1.00,
            )

            if y_cur + box_h > (orig_h - margin_px):
                col_idx += 1
                if col_idx >= max_cols:
                    col_idx = max_cols - 1
                    y_cur = y_offset + 120
                col_x = margin_px + col_idx * (col_w + col_gap)

            bx1 = col_x
            bx2 = col_x + col_w
            by1 = y_cur
            by2 = y_cur + box_h

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 0), 2)

            y_text = by1 + top_pad
            for ln in wrapped_lines:
                (_, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick)
                cv2.putText(
                    canvas,
                    ln,
                    (bx1 + left_pad, y_text + th),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_s,
                    (0, 0, 0),
                    thick,
                    cv2.LINE_AA,
                )
                y_text += th + line_g

            y_cur += box_h + 18

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

        # RIGHT MARGIN LAYOUT (no overlap). Matching/anchors removed; all callouts are page-level in input order.
        box_w = int(right_width - 2 * margin_px)
        resolved_callouts: List[Dict[str, Any]] = []
        for idx, a in enumerate(anns):
            a_type = (a.get("type") or "").strip()
            rubric_point = (a.get("rubric_point") or "").strip()
            comment = (a.get("comment") or "").strip()
            correction = (a.get("correction") or "").strip()
            header = f"[{a_type}] {rubric_point}".strip()
            body = (comment + (f"  Fix: {correction}" if correction else "")).strip()
            if a_type != "grammar_language" and correction:
                body = (comment ).strip()
            resolved_callouts.append({
                "rect": None,
                "header": header,
                "body": body,
                "y_sort": idx,
                "page_level": True,
            })
        resolved_callouts.sort(key=lambda x: x["y_sort"])

        last_bottom_y = margin_px
        gap = 12

        for item in resolved_callouts:
            rect = item["rect"]
            header = item["header"]
            body = item["body"]

            header_scale = 1.05
            body_scale = 1.00
            l_gap = 16

            h_h = _estimate_text_height(header, header_scale, 2, box_w - 24, line_gap=l_gap)
            b_h = _estimate_text_height(body, body_scale, 2, box_w - 24, line_gap=l_gap)
            box_h = h_h + b_h + 60

            bx1 = left_width + orig_w + margin_px
            bx2 = bx1 + box_w

            # Stack top-to-bottom without collisions; anchor near rect if present
            desired_y = rect[1] - 20 if rect else last_bottom_y + gap
            start_y = max(margin_px, last_bottom_y + gap, desired_y)

            by1 = int(start_y)
            by2 = int(by1 + box_h)

            # Ensure box stays within page bounds
            max_bottom = orig_h - margin_px
            if by2 > max_bottom:
                # Try to move box up while respecting last_bottom_y
                by2 = max_bottom
                by1 = max(margin_px, last_bottom_y + gap, by2 - box_h)
                # Constrain by2 again if by1 adjustment caused overflow
                if by1 + box_h > max_bottom:
                    by2 = max_bottom
                    by1 = max(margin_px, last_bottom_y + gap)

            last_bottom_y = by2

            # highlight + connector only if rect exists
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            _draw_wrapped_text(canvas, bx1 + 12, by1 + 24, header, header_scale, 2, box_w - 24, (0, 0, 255), line_gap=l_gap)
            _draw_wrapped_text(canvas, bx1 + 12, by1 + 30 + h_h, body, body_scale, 2, box_w - 24, (0, 0, 0), line_gap=l_gap)

            # Intentionally omit on-page highlights/connectors to avoid overlaying essay content.

        annotated_pages.append(Image.fromarray(canvas[:, :, ::-1]))

    return annotated_pages
