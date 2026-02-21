"""
spell_grammar_checker.py

Standalone script to detect spelling and grammar errors from any PDF using:
- Azure Document Intelligence (OCR)
- xAI Grok API (for intelligent error detection)

Environment variables (.env):
    Grok_API=...
    AZURE_ENDPOINT=...
    AZURE_KEY=...

Usage:
    python spell_grammar_checker.py --pdf document.pdf --output-json errors.json
"""

import argparse
import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import fitz  # PyMuPDF
from PIL import Image

# ===========================
# Constants (annotation layout)
# ===========================

# No margin needed - inline annotations only
MARGIN_WIDTH = 0.0





# ===========================
# Helpers
# ===========================

# Noise patterns to strip before sending to Grok
NOISE_PATTERNS = [
    r"\bscanned with\b.*?(?=\n|$)",
    r"\bcamscanner\b",
    r"\bcs camscanner\b",
    r"\bPage \d+\b",
]


def _strip_noise(text: str) -> str:
    """Remove boilerplate noise (watermarks, page numbers, etc.) from OCR text."""
    t = text
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()

def clean_json_from_llm(text: str) -> str:
    """Remove markdown code fences from LLM JSON output."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _norm_ws(s: str) -> str:
    """Normalize whitespace for substring matching."""
    return re.sub(r"\s+", " ", (s or "").strip())


def _min_words_for_anchor(anchor: str) -> int:
    """Dynamic minimum words: allow 2-3 for short headings, 4+ for normal text."""
    if anchor and len(anchor.split()) <= 4:
        return 2
    return 4


def _anchor_is_valid(anchor: str, ocr_page_text: str, min_words: int = 4) -> bool:
    """Check if anchor is a valid substring of OCR page text (min 4-6 words)."""
    a = _norm_ws(anchor)
    t = _norm_ws(ocr_page_text)
    if not a or len(a.split()) < min_words:
        return False
    return a in t


def _bbox_to_rect(bbox: List[Tuple[int, int]]) -> Optional[Tuple[float, float, float, float]]:
    """Convert polygon bbox to (x0,y0,x1,y1)."""
    if not bbox:
        return None
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _norm_token(t: str) -> str:
    """Normalize token for matching (remove punctuation, lowercase)."""
    return re.sub(r"[^A-Za-z0-9']+", "", (t or "").lower())


def _word_boundary_contains(haystack: str, needle: str) -> bool:
    """Check if needle appears in haystack with word boundaries."""
    if not needle:
        return False
    return re.search(rf"\b{re.escape(needle)}\b", haystack) is not None


def _validate_error(err: Dict[str, Any], ocr_page_text: str) -> bool:
    """Strict validation: error_text must appear in anchor_quote with word boundaries."""
    anchor = _norm_ws(err.get("anchor_quote", ""))
    bad = _norm_ws(err.get("error_text", ""))
    corr = _norm_ws(err.get("correction", ""))

    # All required fields
    if not anchor or not bad or not corr:
        return False

    # Dynamic minimum words (relax for headings)
    min_w = _min_words_for_anchor(anchor)
    if len(anchor.split()) < min_w:
        return False

    # anchor must be a substring of OCR text
    if not _anchor_is_valid(anchor, ocr_page_text, min_words=min_w):
        return False

    # error_text must appear in the anchor
    if bad not in anchor:
        return False
    
    # error_text must have word boundaries in OCR text
    if not _word_boundary_contains(_norm_ws(ocr_page_text), bad):
        return False

    # avoid trivial "corrections" (just spacing/punctuation)
    if re.sub(r"\W+", "", bad.lower()) == re.sub(r"\W+", "", corr.lower()):
        return False

    return True





def _is_visual_confusion(bad: str, corr: str) -> bool:
    """Heuristic: skip errors that are likely OCR glyph confusions (not author mistakes)."""
    if not bad or not corr:
        return False

    b = re.sub(r"[^A-Za-z0-9]", "", bad.lower())
    c = re.sub(r"[^A-Za-z0-9]", "", corr.lower())
    if not b or not c:
        return False

    # Quick length guard
    if abs(len(b) - len(c)) > 1:
        return False

    confusables = {
        ("b", "k"), ("k", "b"),
        ("a", "o"), ("o", "a"),
        ("i", "1"), ("1", "i"),
        ("l", "1"), ("1", "l"),
        ("0", "o"), ("o", "0"),
        ("s", "5"), ("5", "s"),
        ("g", "q"), ("q", "g"),
        ("u", "v"), ("v", "u"),
        ("c", "e"), ("e", "c"),
        ("t", "f"), ("f", "t"),
        ("a", "q"), ("q", "a"),
        ("p", "g"), ("g", "p"),
        ("j", "i"), ("i", "j"),
        ("f", "7"), ("7", "f"),
    }

    # Allow the classic rn/m swap
    if (b == "rn" and c == "m") or (b == "m" and c == "rn"):
        return True

    # Single-character difference
    if len(b) == len(c):
        diffs = [(x, y) for x, y in zip(b, c) if x != y]
        if len(diffs) == 1:
            return tuple(diffs[0]) in confusables

    # Length differs by one: insertion/deletion of a confusable character
    if len(b) + 1 == len(c):
        # bad missing one char
        for i in range(len(c)):
            candidate = c[:i] + c[i + 1:]
            if candidate == b and (c[i], c[i]) in confusables:
                return True
    if len(c) + 1 == len(b):
        for i in range(len(b)):
            candidate = b[:i] + b[i + 1:]
            if candidate == c and (b[i], b[i]) in confusables:
                return True

    return False


def _bbox_to_rect_float(bbox: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """Convert polygon bbox to (x0,y0,x1,y1) with float precision."""
    if not bbox:
        return None
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _word_rects_in_page_coords(page_info: Dict[str, Any]) -> List[Tuple[fitz.Rect, float, str]]:
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


def _rect_intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    """Check if two rectangles intersect."""
    return a.intersects(b)


def _find_error_word_span(wordrects: List[Tuple[fitz.Rect, float, str]], error_text: str) -> Optional[fitz.Rect]:
    """Find the bounding box for error_text by matching normalized word sequences."""
    target = _norm_token(error_text)
    if not target:
        return None

    tokens = [(_norm_token(w), r, c) for (r, c, w) in wordrects]
    
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


def _span_confidence_from_wordrects(wordrects: List[Tuple[fitz.Rect, float, str]], target_rect: fitz.Rect) -> float:
    """Estimate average confidence for words intersecting a target rectangle."""
    confs = [c for (r, c, _) in wordrects if r.intersects(target_rect)]
    return (sum(confs) / len(confs)) if confs else 0.0


def _unit_scale_to_points(unit: str) -> float:
    """Convert Azure coordinate units to PyMuPDF points (1 inch = 72 points)."""
    u = (unit or "").lower()
    if u == "inch":
        return 72.0
    # pixel or other: assume already aligned
    return 1.0





# ===========================
# Environment Setup
# ===========================

def load_environment() -> Tuple[str, DocumentAnalysisClient]:
    """Load API credentials from .env file."""
    load_dotenv()
    grok_key = os.getenv("Grok_API")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_KEY")
    
    missing = []
    if not grok_key:
        missing.append("Grok_API")
    if not azure_endpoint:
        missing.append("AZURE_ENDPOINT")
    if not azure_key:
        missing.append("AZURE_KEY")
    
    if missing:
        raise EnvironmentError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Please set them in your .env file."
        )
    
    doc_client = DocumentAnalysisClient(
        endpoint=azure_endpoint,
        credential=AzureKeyCredential(azure_key)
    )
    return grok_key, doc_client


# ===========================
# Azure OCR
# ===========================

def run_ocr_on_pdf(doc_client: DocumentAnalysisClient, pdf_path: str) -> Dict[str, Any]:
    """
    OCR the full PDF directly (best for text + consistent coordinates).
    Extract lines + words (with polygons + confidence) per page.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    poller = doc_client.begin_analyze_document("prebuilt-read", document=pdf_bytes)
    result = poller.result()

    pages_output: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for p in result.pages or []:
        page_number = int(p.page_number or 0)
        page_w = float(p.width or 0.0)
        page_h = float(p.height or 0.0)
        unit = str(getattr(p, "unit", "") or "pixel")  # often "inch" for PDFs

        page_lines: List[Dict[str, Any]] = []
        page_words: List[Dict[str, Any]] = []

        # Words (best for tight bbox + confidence)
        for w in (p.words or []):
            txt = (w.content or "").strip()
            if not txt:
                continue
            poly = []
            if w.polygon:
                poly = [(float(pt.x), float(pt.y)) for pt in w.polygon]
            page_words.append({
                "text": txt,
                "bbox": poly,
                "confidence": float(getattr(w, "confidence", 1.0) or 1.0),
            })

        # Lines (good for anchor matching / reading flow)
        for ln in (p.lines or []):
            txt = (ln.content or "").strip()
            if not txt:
                continue
            poly = []
            if ln.polygon:
                poly = [(float(pt.x), float(pt.y)) for pt in ln.polygon]
            page_lines.append({"text": txt, "bbox": poly})

        page_text = " ".join([x["text"] for x in page_lines]).strip()

        pages_output.append({
            "page_number": page_number,
            "page_width": page_w,
            "page_height": page_h,
            "unit": unit,
            "ocr_page_text": page_text,
            "lines": page_lines,
            "words": page_words,
        })
        full_text_parts.append(page_text)

    pages_output.sort(key=lambda x: x["page_number"])

    # IMPORTANT: DO NOT sort full_text_parts alphabetically
    return {"pages": pages_output, "full_text": "\n".join(full_text_parts).strip()}


# ===========================
# Annotation
# ===========================

def _find_best_line(anchor: str, fallback: str, lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find a line whose text contains the anchor (or fallback)."""
    norm_anchor = _norm_ws(anchor)
    norm_fallback = _norm_ws(fallback)
    for ln in lines:
        text = _norm_ws(ln.get("text", ""))
        if not text:
            continue
        if norm_anchor and norm_anchor in text:
            return ln
        if norm_fallback and norm_fallback in text:
            return ln
    return None


def annotate_pdf(
    ocr_data: Dict[str, Any],
    errors: List[Dict[str, Any]],
    input_pdf: str,
    output_pdf: str,
) -> None:
    """Create annotated PDF with inline corrections above highlighted errors."""
    if not errors:
        print("No errors to annotate; skipping annotated PDF creation.")
        return

    page_map = {p.get("page_number"): p for p in ocr_data.get("pages", [])}

    # Precompute drawable items per page
    drawable: Dict[int, List[Dict[str, Any]]] = {}
    src = fitz.open(input_pdf)
    try:
        for err in errors:
            page_num = err.get("page") or err.get("page_number")
            if not isinstance(page_num, int) or page_num < 1 or page_num > len(src):
                continue

            page_info = page_map.get(page_num)
            if not page_info:
                continue

            # Build word rects in page coordinates (points)
            wordrects = _word_rects_in_page_coords(page_info)
            if not wordrects:
                continue
            
            error_text = (err.get("error_text") or "").strip()
            if not error_text:
                continue
            
            # Find pixel-perfect bbox for error_text using word rects
            rect = _find_error_word_span(wordrects, error_text)
            if not rect:
                continue
            
            page = src[page_num - 1]
            page_w = float(page_info.get("page_width") or page.rect.width)
            page_h = float(page_info.get("page_height") or page.rect.height)
            if page_w <= 0 or page_h <= 0:
                continue

            # Scale from Azure coordinates to page display coordinates
            unit = page_info.get("unit", "pixel")
            azure_scale = _unit_scale_to_points(unit)
            sx = page.rect.width / (page_w * azure_scale)
            sy = page.rect.height / (page_h * azure_scale)
            rect = rect * fitz.Matrix(sx, sy)
            
            # Check confidence if available (skip if too low for handwritten docs)
            conf = _span_confidence_from_wordrects(wordrects, rect)
            if conf < 0.70:  # threshold for handwriting tolerance
                print(f"  Skipping low-confidence error (conf={conf:.2f}): {error_text}")
                continue

            entry = {
                "rect": rect,
                "correction": (err.get("correction") or "").strip() or "(review)",
            }
            drawable.setdefault(page_num, []).append(entry)

        if not drawable:
            print("No drawable anchors found; annotated PDF not created.")
            return

        out = fitz.open()
        for idx in range(len(src)):
            src_page = src[idx]
            w, h = src_page.rect.width, src_page.rect.height
            new_page = out.new_page(width=w, height=h)
            new_page.show_pdf_page(fitz.Rect(0, 0, w, h), src, idx)

            items = drawable.get(idx + 1, [])

            for it in items:
                rect = it["rect"]
                
                # Draw darker red rectangle around the error
                new_page.draw_rect(rect, color=(0.8, 0, 0), width=2.0)
                
                # Write correction text above the rectangle with white background
                correction_text = it['correction']
                text_point = fitz.Point(rect.x0, rect.y0 - 4)  # Slightly above the rectangle
                
                # Calculate text width for background rectangle
                text_width = fitz.get_text_length(correction_text, fontname="hebo", fontsize=10)
                
                # Draw white background rectangle for the correction text
                bg_rect = fitz.Rect(
                    rect.x0 - 2,
                    rect.y0 - 16,
                    rect.x0 + text_width + 4,
                    rect.y0 - 2
                )
                new_page.draw_rect(bg_rect, color=(0.8, 0, 0), fill=(1, 1, 1), width=1.0)
                
                # Insert correction text in bold dark red
                new_page.insert_text(
                    text_point,
                    correction_text,
                    fontsize=10,
                    color=(0.8, 0, 0),  # Darker red color
                    fontname="hebo"  # Helvetica Bold
                )

        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        page_count = len(out)
        out.save(output_pdf, deflate=True)
        out.close()
        print(f"✓ Annotated PDF saved to: {output_pdf} (pages: {page_count})")
    finally:
        src.close()


# ===========================
# Grok API
# ===========================

def _grok_chat(
    grok_api_key: str,
    messages: List[Dict[str, str]],
    model: str = "grok-4-1-fast-reasoning",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: int = 180,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """Call xAI Grok API with retry logic."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {grok_api_key}",
        "Content-Type": "application/json",
    }
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise RuntimeError(f"Grok API failed after {max_retries} attempts: {e}")
            wait = min(backoff ** attempt, 60.0)
            print(f"  Grok API error (attempt {attempt}/{max_retries}), retrying in {wait:.1f}s...")
            time.sleep(wait)
    
    raise RuntimeError("Grok API: max retries exceeded")


def _process_page(page: Dict[str, Any], grok_api_key: str, instructions: str, schema_hint: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single page for error detection (used for parallel processing)."""
    page_num = page.get("page_number", 0)
    ocr_page_text = page.get("ocr_page_text", "").strip()
    
    # Strip noise before processing
    ocr_page_text = _strip_noise(ocr_page_text)
    
    if not ocr_page_text or len(ocr_page_text) < 50:
        print(f"  Page {page_num}: Skipping (too little text)")
        return []
    
    print(f"  Page {page_num}: Checking for errors...")
    
    # Prepare payload
    user_payload = {
        "page_number": page_num,
        "ocr_page_text": ocr_page_text,
        "output_schema": schema_hint,
    }
    
    system_msg = {"role": "system", "content": instructions}
    user_msg = {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    
    # Reduced attempts for faster processing
    max_attempts = 2
    valid_errors = []
    
    for attempt in range(1, max_attempts + 1):
        try:
            data = _grok_chat(
                grok_api_key,
                messages=[system_msg, user_msg],
                temperature=0.1,
                max_tokens=4000,
            )
            
            content = data["choices"][0]["message"]["content"]
            content_clean = clean_json_from_llm(content)
            parsed = json.loads(content_clean)
            
            # Extract errors
            errors = parsed.get("errors", [])
            if not errors:
                print(f"    Page {page_num} [attempt {attempt}/{max_attempts}]: No errors found.")
                break
            
            # Validate errors strictly
            valid_errors = []
            for e in errors:
                if _validate_error(e, ocr_page_text):
                    valid_errors.append(e)
            
            print(f"    Page {page_num} [attempt {attempt}/{max_attempts}]: Valid: {len(valid_errors)}/{len(errors)}")
            
            # Accept if we have at least 1 valid error
            if len(valid_errors) >= 1:
                return valid_errors
            
            # Retry with stronger prompt
            if attempt < max_attempts:
                user_msg["content"] = json.dumps({
                    **user_payload,
                    "note": "Previous attempt had invalid anchor_quote values. "
                            "COPY EXACT substrings from OCR_PAGE_TEXT."
                }, ensure_ascii=False)
                print(f"    Retrying page {page_num}...")
            else:
                print(f"    ⚠ Page {page_num}: No valid errors after {max_attempts} attempts.")
        
        except Exception as e:
            print(f"    ✗ Error on page {page_num}, attempt {attempt}: {e}")
            if attempt == max_attempts:
                print(f"    ⚠ Page {page_num}: Failed after {max_attempts} attempts.")
    
    return valid_errors


def detect_spelling_grammar_errors(
    grok_api_key: str,
    ocr_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Detect spelling and grammar errors using Grok with parallel processing.
    Returns list of annotations with validated anchor_quote.
    """
    
    pages = ocr_data.get("pages", [])
    
    schema_hint = {
        "page": 1,
        "errors": [
            {
                "page": 1,
                "type": "spelling",  # or "grammar"
                "anchor_quote": "EXACT substring from OCR_PAGE_TEXT (4-20 words)",
                "error_text": "the misspelled or grammatically incorrect phrase",
                "correction": "corrected version",
                "explanation": "why this is an error",
            }
        ]
    }
    
    instructions = (
        "You are a spelling and grammar checker.\n\n"
        "TASK:\n"
        "- Find spelling errors (misspelled words, typos).\n"
        "- Find grammar errors (subject-verb agreement, tense, articles, prepositions, etc.).\n"
        "- For each error, return:\n"
        "  * type: 'spelling' or 'grammar'\n"
        "  * anchor_quote: EXACT contiguous substring from OCR_PAGE_TEXT (4-20 words containing the error)\n"
        "  * error_text: the specific word/phrase that is wrong\n"
        "  * correction: the corrected version\n"
        "  * explanation: brief reason why it's an error\n\n"
        "ANCHOR RULE (CRITICAL):\n"
        "- You are given OCR_PAGE_TEXT below.\n"
        "- anchor_quote MUST be an EXACT contiguous substring copied from OCR_PAGE_TEXT.\n"
        "- Use 4 to 20 words for anchor_quote.\n"
        "- Do NOT paraphrase. Do NOT correct spelling inside anchor_quote.\n"
        "- The anchor_quote must CONTAIN the error_text as a substring.\n"
        "- error_text MUST appear EXACTLY inside anchor_quote.\n"
        "- If you cannot find a suitable quote in OCR_PAGE_TEXT, SKIP that error.\n\n"
        "RULES:\n"
        "- Ignore OCR artifacts (misread characters are not spelling errors).\n"
        "- Focus on actual author mistakes (typos, grammar issues).\n"
        "- Return 2-8 errors per page (most significant ones).\n"
        "- Output valid JSON only matching the schema.\n"
    )
    
    # Process pages in parallel for faster results
    all_errors: List[Dict[str, Any]] = []
    max_workers = min(4, len(pages))  # Process up to 4 pages concurrently
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_page, page, grok_api_key, instructions, schema_hint): page
            for page in pages
        }
        
        for future in as_completed(futures):
            try:
                errors = future.result()
                all_errors.extend(errors)
            except Exception as e:
                page = futures[future]
                print(f"    ✗ Page {page.get('page_number', '?')} failed: {e}")
    
    return all_errors


def _filter_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop likely OCR-glyph confusions so we only keep author mistakes."""
    kept = []
    skipped = 0
    for err in errors:
        bad = (err.get("error_text") or "").strip()
        corr = (err.get("correction") or "").strip()
        # Ignore spacing-only issues
        norm_bad = re.sub(r"\s+", "", bad)
        norm_corr = re.sub(r"\s+", "", corr)
        if norm_bad and norm_bad == norm_corr:
            skipped += 1
            continue
        if _is_visual_confusion(bad, corr):
            skipped += 1
            continue
        kept.append(err)
    if skipped:
        print(f"Filtered out {skipped} OCR-like confusions")
    return kept


def run_spelling_detection(grok_api_key: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Single entrypoint for pipeline use: run spelling/grammar detection on existing OCR data.
    Does not call run_ocr_on_pdf; caller supplies ocr_data (must have pages with ocr_page_text).
    Returns a consistent payload. Exceptions (e.g. Grok/network) propagate; caller should catch.
    """
    pages = ocr_data.get("pages", [])
    errors = detect_spelling_grammar_errors(grok_api_key, ocr_data)
    errors = _filter_errors(errors)
    return {
        "errors": errors,
        "meta": {
            "total_pages": len(pages),
            "total_errors": len(errors),
        },
    }


# ===========================
# Cache helpers
# ===========================

def clear_debug_cache(cache_dir: str = "debug_cache") -> None:
    """Remove existing files from cache directory before a run."""
    if not os.path.isdir(cache_dir):
        return

    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
        except Exception as exc:
            print(f"  ⚠ Unable to remove {path}: {exc}")


# ===========================
# Main
# ===========================

def main():
    parser = argparse.ArgumentParser(
        description="Detect spelling and grammar errors in PDF using Azure OCR + xAI Grok"
    )
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--output-json", default="spelling_grammar_errors.json", help="Output JSON path")
    parser.add_argument("--output-pdf", default=None, help="Annotated PDF output path (default: <pdf>-annotated.pdf)")
    args = parser.parse_args()
    
    if not os.path.isfile(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    
    print("Cleaning cache directory...")
    clear_debug_cache()

    print("Loading environment...")
    grok_key, doc_client = load_environment()

    default_pdf = Path(args.pdf).with_name(f"{Path(args.pdf).stem}-annotated.pdf")
    output_pdf = args.output_pdf or str(default_pdf)
    
    print(f"Running OCR on {args.pdf}...")
    ocr_data = run_ocr_on_pdf(doc_client, args.pdf)
    print(f"  OCR complete: {len(ocr_data['pages'])} pages")
    
    print("\nDetecting spelling and grammar errors...")
    errors = detect_spelling_grammar_errors(grok_key, ocr_data)
    errors = _filter_errors(errors)
    print(f"  Found {len(errors)} validated errors")
    
    # Save results
    output = {
        "pdf": os.path.basename(args.pdf),
        "total_pages": len(ocr_data["pages"]),
        "total_errors": len(errors),
        "errors": errors,
    }
    
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to: {args.output_json}")
    print(f"✓ Total errors found: {len(errors)}")
    annotate_pdf(ocr_data, errors, args.pdf, output_pdf)
    
    # Summary by type
    spelling_count = sum(1 for e in errors if e.get("type") == "spelling")
    grammar_count = sum(1 for e in errors if e.get("type") == "grammar")
    print(f"  - Spelling: {spelling_count}")
    print(f"  - Grammar: {grammar_count}")


if __name__ == "__main__":
    main()