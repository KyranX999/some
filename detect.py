"""Highlighted-word extraction via OCR + colour analysis (no multimodal LLM).

Pipeline:
  1. Cached PaddleOCR PP-OCRv6 word-level boxes  (run_ocr.py).
  2. HSV orange-highlighter mask  (red-pen H<10 and blue ink excluded).
  3. Per English word: find its printed-glyph rows (dark-pixel band, robust
     even when letters sit on a strong highlight) and measure the orange
     coverage of that band -> highlighted / not.
  4. Filter CJK tokens (the teacher/student handwritten annotations).
  5. Merge consecutive highlighted words into phrases, emit top-to-bottom.

A verification overlay is written to vis_result.png.
"""
import cv2, numpy as np, json, re, sys

H_LO, H_HI, S_MIN, V_MIN = 10, 26, 13, 90   # orange highlighter in HSV
BELOW = 8                                    # rows below glyphs (underline swipes)
THRESH = 0.50                                # min orange coverage of glyph band
CJK = re.compile(r"[一-鿿]")


def build_masks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    orange = ((H >= H_LO) & (H <= H_HI) & (S > S_MIN) & (V > V_MIN)).astype(np.uint8)
    dark = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 110).astype(np.uint8)
    return orange, dark


def glyph_band(dark, box, line_box):
    """Rows of the word's own printed glyphs (densest dark segment that
    overlaps the OCR line box). Works even when letters lie on a highlight,
    because we threshold on darkness, not saturation."""
    x1, y1, x2, y2 = [int(v) for v in box]
    sub = dark[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    rows = sub.sum(axis=1)
    on = rows > max(2, 0.08 * (x2 - x1))
    if not on.any():
        return None
    segs, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        if not v and s is not None:
            segs.append((s, i)); s = None
    if s is not None:
        segs.append((s, len(on)))
    lt = min(p[1] for p in line_box); lb = max(p[1] for p in line_box)
    inb = [g for g in segs if y1 + g[0] < lb + 4 and y1 + g[1] > lt - 4] or segs
    g = max(inb, key=lambda gg: rows[gg[0]:gg[1]].sum())
    return y1 + g[0], y1 + g[1]


def coverage(orange, dark, box, line_box):
    gb = glyph_band(dark, box, line_box)
    if gb is None:
        return 0.0
    gt, gb_ = gb
    x1, _, x2, _ = [int(v) for v in box]
    s = orange[max(0, gt - 2):gb_ + BELOW, x1:x2]
    return float(s.mean()) if s.size else 0.0


def is_sep(tok):
    return tok.strip(" ,.;:\"'()") == ""


def extract(img, lines):
    orange, dark = build_masks(img)
    lines = sorted(lines, key=lambda l: min(p[1] for p in l["line_box"]))
    phrases = []
    for l in lines:
        toks, boxes, lb = l["words"], l["word_boxes"], l["line_box"]
        flag = [None if (CJK.search(t) or is_sep(t))
                else coverage(orange, dark, b, lb) >= THRESH
                for t, b in zip(toks, boxes)]
        i, n = 0, len(toks)
        while i < n:
            if flag[i] is True:
                grp, last, k = [i], i, i + 1
                while k < n:
                    if flag[k] is None:
                        k += 1; continue
                    if flag[k]:
                        grp.append(k); last = k; k += 1
                    else:
                        break
                text = "".join(toks[grp[0]:last + 1]).strip()
                xs = [boxes[j][0] for j in grp]; ys = [boxes[j][1] for j in grp]
                bxs = [boxes[j][2] for j in grp]; bys = [boxes[j][3] for j in grp]
                phrases.append((min(ys), min(xs),
                                (min(xs), min(ys), max(bxs), max(bys)), text))
                i = last + 1
            else:
                i += 1
    phrases.sort(key=lambda p: (p[0], p[1]))
    return phrases


def main():
    img = cv2.imread("doc.png")
    data = json.load(open("ocr_cache.json"))
    phrases = extract(img, data["lines"])
    result = [p[3] for p in phrases]

    print(f"=== {len(result)} highlighted items (top-to-bottom) ===")
    for i, t in enumerate(result, 1):
        print(f"{i:2d}. {t}")
    json.dump(result, open("result.json", "w"), ensure_ascii=False, indent=1)

    vis = img.copy()
    for *_ , bb, _t in [(p[0], p[1], p[2], p[3]) for p in phrases]:
        x1, y1, x2, y2 = bb
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite("vis_result.png", vis)

    gt = ["contributing factor", "misuse", "tackles", "legal", "legally",
          "end-user license agreements", "But this is far from ideal",
          "the nature of", "implications", "The issue is that", "consent to",
          "outcomes", "Warning fatigue", "distracted", "annoyed", "presented",
          "represent", "Given", "obtaining", "automakers", "liability", "after all"]
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    gset, rset = [norm(g) for g in gt], [norm(r) for r in result]
    missing = [g for g in gt if norm(g) not in rset]
    extra = [r for r in result if norm(r) not in gset]
    print(f"\nmatched {len(gt) - len(missing)}/{len(gt)}")
    print("MISSING:", missing)
    print("EXTRA  :", extra)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        THRESH = float(sys.argv[1])
    main()
