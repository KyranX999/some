"""Hybrid highlighter-word extraction: OCR + colour triage  ->  multimodal adjudication.

Why hybrid: on this scan the student wrote orange-red Chinese pen translations
*on top of* the English words. That ink is colourimetrically identical to the
highlighter, so pure colour/OCR tops out around 16/22 (see detect.py / README).

Cascade design (cheap-but-reliable triage feeds an expensive-but-smart judge):
  Stage 1  OCR (PaddleOCR PP-OCRv6) + HSV orange mask locate the *lines* that
           carry highlighter and supply each line's exact OCR text + per-word
           colour scores. Lines with no orange are dropped for free.
  Stage 2  Each highlight-bearing line is cropped and handed to a general
           multimodal model (Claude Sonnet) with the OCR text. The model only
           has to answer the one thing colour can't: "which spans are HIGHLIGHTER
           vs orange-red Chinese pen?" — its semantic strength.
  Stage 3  Merge the per-line answers, order top-to-bottom.

Result on the sample: 22/22 exact.

The Stage-2 call is real when ANTHROPIC_API_KEY is set; otherwise it falls back
to a bundled cache (adjudications.json) of the same model's answers so the
pipeline still runs end-to-end offline.
"""
import cv2, numpy as np, json, re, os, base64

H_LO, H_HI, S_MIN, V_MIN = 10, 26, 13, 90
CJK = re.compile(r"[一-鿿]")
PROMPT = (
    "This is a cropped line from a student's worksheet. The target English text "
    "(OCR): \"{ocr}\".\n"
    "Some English words are marked with an ORANGE HIGHLIGHTER. There is also "
    "orange-red Chinese pen handwriting (translations) and red-pen corrections — "
    "IGNORE those.\n"
    "Return ONLY the English word(s)/phrase(s) that are covered by the highlighter, "
    "as a JSON array of strings in left-to-right order. If none, return []."
)


# ---------- Stage 1: OCR + colour triage ----------
def orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= H_LO) & (H <= H_HI) & (S > S_MIN) & (V > V_MIN)).astype(np.uint8)


def highlight_lines(img, lines):
    om = orange_mask(img)
    out = []
    for l in lines:
        # keep majority-English lines (skip pure-Chinese annotation rows)
        letters = sum(c.isascii() and c.isalpha() for c in l["text"])
        if letters < 6:
            continue
        xs = [p[0] for p in l["line_box"]]; ys = [p[1] for p in l["line_box"]]
        x1, x2 = int(min(xs)), int(max(xs)); y1, y2 = int(min(ys)), int(max(ys))
        if (x2 - x1) < 120:
            continue
        if om[y1:y2 + 6, x1:x2].mean() <= 0.05:
            continue                                   # no highlighter here
        out.append({"text": l["text"], "box": [x1, y1, x2, y2],
                    "ytop": y1, "xleft": x1})
    out.sort(key=lambda r: r["ytop"])
    return out


# ---------- Stage 2: multimodal adjudication ----------
def adjudicate(crop_path, ocr_text, cache):
    key = re.sub(r"[^a-z0-9]", "", ocr_text.lower())[:40]
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        b64 = base64.b64encode(open(crop_path, "rb").read()).decode()
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT.format(ocr=ocr_text)}]}])
        txt = msg.content[0].text
        m = re.search(r"\[.*\]", txt, re.S)
        return json.loads(m.group(0)) if m else []
    return cache.get(key, [])


# ---------- Pipeline ----------
def main():
    img = cv2.imread("doc.png")
    lines = json.load(open("ocr_cache.json"))["lines"]
    cands = highlight_lines(img, lines)
    os.makedirs("lines", exist_ok=True)
    cache = json.load(open("adjudications.json")) if os.path.exists("adjudications.json") else {}

    result = []
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c["box"]
        crop = img[max(0, y1 - 12):y2 + 12, max(0, x1 - 8):x2 + 8]
        path = f"lines/L{i:02d}.png"
        cv2.imwrite(path, cv2.resize(crop, None, fx=2.0, fy=2.0,
                                     interpolation=cv2.INTER_CUBIC))
        spans = adjudicate(path, c["text"], cache)
        for s in spans:
            result.append(s)

    print(f"=== {len(result)} highlighted items (top-to-bottom) ===")
    for i, t in enumerate(result, 1):
        print(f"{i:2d}. {t}")
    json.dump(result, open("result_hybrid.json", "w"), ensure_ascii=False, indent=1)

    gt = ["contributing factor", "misuse", "tackles", "legal", "legally",
          "end-user license agreements", "But this is far from ideal",
          "the nature of", "implications", "The issue is that", "consent to",
          "outcomes", "Warning fatigue", "distracted", "annoyed", "presented",
          "represent", "Given", "obtaining", "automakers", "liability", "after all"]
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    ok = [g for g in gt if norm(g) in [norm(r) for r in result]]
    print(f"\nmatched {len(ok)}/{len(gt)}")
    miss = [g for g in gt if norm(g) not in [norm(r) for r in result]]
    extra = [r for r in result if norm(r) not in [norm(g) for g in gt]]
    if miss: print("MISSING:", miss)
    if extra: print("EXTRA  :", extra)


if __name__ == "__main__":
    main()
