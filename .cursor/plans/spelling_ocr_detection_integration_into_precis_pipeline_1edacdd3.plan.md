---
name: Spelling/OCR detection integration into Precis pipeline
overview: ""
todos: []
isProject: false
---

# Spelling/OCR detection integration into Precis pipeline

## Current state

- **[grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py)**: Runs its own OCR (`run_ocr_on_pdf`), then grading and annotations; passes `spelling_errors=None` to `annotate_pdf_essay_pages`. No spelling detection is run. Annotator is imported with the same fallback pattern as the plan (try import, then `spec_from_file_location`).
- **[ocr-spell-correction.py](D:/Sova/Projects/Precis/precis/ocr-spell-correction.py)**: Entry is CLI `main()`: loads env, runs `run_ocr_on_pdf(doc_client, pdf_path)`, then `detect_spelling_grammar_errors(grok_key, ocr_data)`, then `_filter_errors(errors)`. Returns raw errors list; no single callable that accepts precomputed OCR. `detect_spelling_grammar_errors(grok_key, ocr_data)` expects `ocr_data["pages"]` with each page having `page_number` and `ocr_page_text` (grade_pdf_precis sets `ocr_page_text` in `split_extra_text` at line 551).
- **[annotate_pdf_with_precis.py](D:/Sova/Projects/Precis/precis/annotate_pdf_with_precis.py)**: `annotate_pdf_essay_pages(..., spelling_errors=...)` accepts spelling_errors but does not draw them; `_add_spelling_annotations_to_pdf(pil_images, ocr_data, spelling_errors, output_path)` builds a PDF from PIL and draws spelling on it. It is never called from grade_pdf_precis. Annotator expects each error to have `page` (1-based), `error_text`, `correction`, `anchor_quote` (optional).

## Strategy

1. **Single entrypoint in ocr-spell-correction**: Add a function that takes existing `ocr_data` + `grok_key`, runs detection + filter, returns a stable payload (no duplicate OCR).
2. **grade_pdf_precis**: Safely import the spelling module (same pattern as annotator)“Because the file name contains a hyphen (`ocr-spell-correction.py`), do **not** use normal `import`. Use `importlib.util.spec_from_file_location(...)` as the **primary** import method.”, run spelling after OCR when enabled, normalize and validate errors, pass to annotator; on failure log and continue with empty spelling.
3. **Normalize before annotation**: One function that maps raw errors to `{page, error_text, correction, anchor_quote, category}`, drops invalid entries, dedupes.
4. **Apply spelling to output**: Either (a) call `_add_spelling_annotations_to_pdf` on annotated PIL before merge and merge the resulting PDF with the report,Apply spelling on the **merged output PDF** (after report pages are added) and use `answer_start_page = num_report_pages` so page indices align. or (b) after merge, add spelling to the merged PDF’s answer pages. Option (b) is simpler and avoids changing `merge_report_and_annotated_answer`: add a step that opens the output PDF and draws spelling on pages `num_report_pages + (error.page - 1)` using the same PyMuPDF logic as `_add_spelling_annotations_to_pdf`.

---

## TASK A — Stable integration function in ocr-spell-correction.py

**Goal:** One callable for pipeline use that does not depend on CLI or duplicate OCR.

1. **Add** (e.g. after `_filter_errors` around line 739) a single entrypoint:
  - **Name/signature:** `run_spelling_detection(grok_api_key: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]`
  - **Behavior:** Call `detect_spelling_grammar_errors(grok_api_key, ocr_data)`, then `_filter_errors(...)`. Do not call `run_ocr_on_pdf`; caller provides `ocr_data`.
  - **Return payload (consistent):**

```python
     {
       "errors": [...],   # list from _filter_errors
       "meta": {
         "total_pages": len(ocr_data.get("pages", [])),
         "total_errors": len(errors),
       }
     }
     

```

- **Raises:** Let existing functions raise (e.g. Grok/network); caller will catch and degrade.

1. **Optional:** Add `__all__ = ["run_spelling_detection", "detect_spelling_grammar_errors", "load_environment", ...]` so pipeline can import explicitly.
2. **Do not** remove or change CLI `main()`; keep it for standalone use.

**File:** [ocr-spell-correction.py](D:/Sova/Projects/Precis/precis/ocr-spell-correction.py)

---

## TASK B — Integration step in grade_pdf_precis.py

**Goal:** Run spelling detection after OCR when enabled; never crash pipeline; save debug output.

1. **Import spelling module safely** (same style as annotator, ~lines 26–42):
  - Try `from ocr_spell_correction import run_spelling_detection` (use module name that matches file: `ocr_spell_correction` for `ocr-spell-correction.py` via `importlib` with a safe name, e.g. `ocr_spell_correction`).
  - Fallback: `spec_from_file_location("ocr_spell_correction", os.path.join(parent_dir, "ocr-spell-correction.py"))` and load; then `run_spelling_detection = getattr(mod, "run_spelling_detection", None)`.
  - If missing or import fails: set `run_spelling_detection = None` and continue.
2. **Run spelling after OCR when enabled:**
  - In `run_precis_grading`, after `split_extra_text` (and before or after annotations LLM, as you prefer; logically “after OCR is available” is enough), if `enable_spelling_annotations` and `run_spelling_detection` is not None:
    - Call `run_spelling_detection(grok_key, ocr_data)`.
    - On success: use returned `payload["errors"]` as raw errors for normalization.
    - On exception: log warning (e.g. `print("Warning: spelling detection failed: ...")` or `logging.warning(...)`), set raw errors to `[]`, do not re-raise.
  - If `enable_spelling_annotations` is False or `run_spelling_detection` is None: skip call; raw errors = `[]`.
3. **Save spelling debug output:**
  - Write raw (and optionally normalized) spelling result to a predictable path, e.g. `debug_llm/spelling_errors_debug.json` or next to `output_json_path`: `os.path.join(os.path.dirname(output_json_path), "debug_llm", "spelling_errors_debug.json")`.
  - Content: e.g. `{"raw_count": N, "normalized_count": M, "errors": normalized_errors, "meta": {...}}` so it’s clear what was passed to the annotator.
4. **Pass normalized errors into annotation:** When calling `annotate_pdf_essay_pages`, pass `spelling_errors=normalized_errors` (list from Task C). If spelling was skipped or failed, pass `[]`.

**File:** [grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py)

---

## TASK C — Normalize spelling error schema before annotation

**Goal:** One canonical shape; drop invalid entries; deduplicate so the pipeline never crashes on malformed data.

1. **Add normalizer in grade_pdf_precis.py** (e.g. near other annotation helpers):
  - **Signature:** `normalize_spelling_errors(raw_errors: List[Dict[str, Any]], num_pages: int) -> List[Dict[str, Any]]`
  - **Per entry:** Map to:
    - `page`: int (from `page` or `page_number`); must be in `[1, num_pages]`.
    - `error_text`: str (required); strip; if empty, drop.
    - `correction`: str (required); strip; if empty, drop.
    - `anchor_quote`: str or None (optional).
    - `category`: str, default `"spelling"` if type is spelling else `"grammar_presentation"` (or `"grammar"`).
  - **Drop:** missing page, page out of range, missing/empty `error_text` or `correction`.
  - **Deduplicate:** Same `(page, error_text, correction)` (and optionally similar `anchor_quote`) count as one; keep first occurrence.
  - **Return:** List of normalized dicts.
2. **Use in pipeline:** After getting raw errors (from Task B), call `normalized_errors = normalize_spelling_errors(raw_errors, len(ocr_data.get("pages", [])))`. Pass `normalized_errors` to the annotator and into the spelling debug JSON.

**File:** [grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py)

---

## TASK D — Feature toggle and safe behavior

**Goal:** Pipeline stable with spelling on/off or zero results.

1. **Add flag in grade_pdf_precis.py:**
  - In `run_precis_grading`, add parameter: `enable_spelling_annotations: bool = True`.
  - In `main()`, add CLI arg: `--no-spelling` or `--enable-spelling` (default True); map to `enable_spelling_annotations`.
2. **When disabled:** Do not call `run_spelling_detection`; set raw errors to `[]`; skip any spelling-specific file writes if desired; still pass `spelling_errors=[]` to the annotator.
3. **When enabled but no valid errors:** After normalization, if `normalized_errors` is empty, continue normally (annotator already handles empty list); do not treat as failure.

**File:** [grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py)

---

## TASK E — Logging and verification

**Goal:** Clear summary and inspectable debug output.

1. **Print summary** (in grade_pdf_precis after spelling step):
  - Spelling module called? (yes/no; or “skipped – disabled” / “skipped – module not available”).
  - Raw errors count.
  - Valid normalized errors count.
  - Pages affected (e.g. set of `page` from normalized list).
2. **Save debug file:** As in Task B, write to `debug_llm/spelling_errors_debug.json` (or chosen path) with raw_count, normalized_count, normalized list, and meta, so integration can be verified without re-running the full pipeline.

**File:** [grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py)

---

## Applying spelling to the output PDF

Currently `annotate_pdf_essay_pages` receives `spelling_errors` but never draws them; `_add_spelling_annotations_to_pdf` in the annotator builds a PDF from PIL and draws spelling there. The pipeline merges the report with annotated PIL pages, so spelling is never applied.

**Options:**

- **Option A (recommended):** After `merge_report_and_annotated_answer` in grade_pdf_precis, if `normalized_errors` is non-empty, open `output_pdf_path`, and for each error at 1-based `page`, add the red box and correction text to the merged PDF page at index `num_report_pages + (page - 1)` (report pages are first). Implement a helper (in annotate_pdf_with_precis.py or grade_pdf_precis.py) that takes `(output_pdf_path, ocr_data, spelling_errors, answer_start_page_0indexed)` and reuses the same PyMuPDF drawing logic as `_add_spelling_annotations_to_pdf` (word rect lookup, draw_rect, insert_text). This keeps merge unchanged and adds spelling in one place.
- **Option B:** Before merge, call `_add_spelling_annotations_to_pdf(annotated_pages, ocr_data, normalized_errors, temp_pdf)`, then merge the report with the pages of `temp_pdf` instead of with `annotated_pages`. This requires extending `merge_report_and_annotated_answer` to accept either a list of PIL images or a path to a PDF for the answer pages.

**Recommendation:** Implement Option A: add a function (e.g. `add_spelling_annotations_to_merged_pdf`) that opens the merged output PDF and applies spelling to the answer pages only, using the same word-rect and drawing logic as `_add_spelling_annotations_to_pdf`. Call it from `run_precis_grading` only when `normalized_errors` is non-empty.

**Files:** [annotate_pdf_with_precis.py](D:/Sova/Projects/Precis/precis/annotate_pdf_with_precis.py) (helper), [grade_pdf_precis.py](D:/Sova/Projects/Precis/precis/grade_pdf_precis.py) (call after merge).

---

## Optional: annotate_pdf_with_precis.py compatibility

- Ensure `_add_spelling_annotations_to_pdf` and any new “add to merged PDF” helper accept the normalized shape: `page` (int, 1-based), `error_text`, `correction`, `anchor_quote` (optional). No change needed if already using these keys.
- If the annotator currently expects different keys (e.g. `page_number`), the normalizer output should match what the annotator uses (or the annotator should be updated to use the normalized keys). Current code uses `error.get("page", 1)`, `error.get("error_text")`, `error.get("correction")`, `error.get("anchor_quote")`, so the normalizer shape above is compatible.

---

## Implementation order

1. **Task A** – Add `run_spelling_detection` in ocr-spell-correction.py.
2. **Task C** – Add `normalize_spelling_errors` in grade_pdf_precis.py.
3. **Task B** – Import spelling module, call it after OCR, save debug, pass normalized errors to annotator.
4. **Task D** – Add `enable_spelling_annotations` and CLI.
5. **Task E** – Add summary print and debug file content.
6. **Apply spelling to PDF** – Implement and call `add_spelling_annotations_to_merged_pdf` (or equivalent) after merge when normalized_errors is non-empty.

---

## Acceptance criteria (recap)

- Pipeline runs end-to-end with spelling enabled and disabled.
- Spelling output is normalized and passed to the annotation stage; spelling is visible on the output PDF when enabled and errors exist.
- No crashes from malformed or missing spelling entries (normalize + try/except).
- Debug file and console summary allow verification of integration.

