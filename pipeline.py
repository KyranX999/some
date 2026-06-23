"""Production pipeline: highlighter-word extraction from a worksheet photo.

  Stage 1  on-device OCR (PP-OCRv6_tiny @ 1280) + HSV orange triage
           -> the lines that carry highlighter, with word boxes.
  Stage 2  ONE multimodal call on a labelled montage (with colour-mask hints)
           -> highlighter spans per line  (highlighter vs Chinese pen).
  Stage 3  assemble: cross-line span merge (handles line-wrapped phrases such
           as "Woodland Trust" / "to turn vacant plots of land green took off"),
           order top-to-bottom.

Usage:  python3 pipeline.py <image.jpg> [answer_cache.json]
The multimodal step calls Claude when ANTHROPIC_API_KEY is set; otherwise it
reads a bundled per-line answer cache so the pipeline runs offline.
"""
import cv2, numpy as np, json, re, os, base64, sys

H_LO, H_HI, S_MIN, V_MIN = 10, 26, 13, 90
ROW_H, LABEL_W = 80, 48
CJK = re.compile(r"[一-鿿]")


# ---------- Stage 1 ----------
def load_doc(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rm = gray.mean(axis=1); cm = gray.mean(axis=0)
    ry = np.where(rm > 110)[0]; rx = np.where(cm > 110)[0]
    y0, y1 = (max(0, ry.min() - 10), min(img.shape[0], ry.max() + 10)) if len(ry) else (0, img.shape[0])
    x0, x1 = (max(0, rx.min() - 10), min(img.shape[1], rx.max() + 10)) if len(rx) else (0, img.shape[1])
    return img[y0:y1, x0:x1]


def run_ocr(doc_path):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv6_tiny_det",
                    text_recognition_model_name="PP-OCRv6_tiny_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, lang="en", enable_mkldnn=False,
                    text_det_limit_side_len=1280, text_det_limit_type="max")
    r = ocr.predict(doc_path, return_word_box=True)[0]
    return [{"text": r["rec_texts"][i],
             "line_box": np.array(r["rec_polys"][i]).tolist(),
             "words": r["text_word"][i],
             "word_boxes": [np.array(b).tolist() for b in r["text_word_boxes"][i]]}
            for i in range(len(r["rec_texts"]))]


def merge_rows(lines):
    """Merge OCR boxes the detector split across the SAME text row back into one
    logical line (side-by-side fragments), so word order and line-wrap logic are
    correct. Conservative: only joins boxes on the same baseline that are
    horizontally adjacent and non-overlapping (tilt-safe)."""
    items = []
    for l in lines:
        ys = [p[1] for p in l["line_box"]]; xs = [p[0] for p in l["line_box"]]
        items.append({**l, "y1": min(ys), "y2": max(ys), "x1": min(xs), "x2": max(xs)})
    items.sort(key=lambda r: r["x1"])
    used = [False] * len(items); out = []
    for i, a in enumerate(items):
        if used[i]:
            continue
        grp = [a]; used[i] = True
        changed = True
        while changed:
            changed = False
            cur = grp[-1]
            hc = cur["y2"] - cur["y1"]
            for j, b in enumerate(items):
                if used[j]:
                    continue
                ycdiff = abs((b["y1"] + b["y2"]) / 2 - (cur["y1"] + cur["y2"]) / 2)
                gap = b["x1"] - cur["x2"]
                if ycdiff < 0.4 * hc and -8 <= gap <= 1.8 * hc:
                    grp.append(b); used[j] = True; changed = True; break
        grp.sort(key=lambda r: r["x1"])
        words, wb = [], []
        for g in grp:
            if words:
                words.append(" "); wb.append([g["x1"], g["y1"], g["x1"], g["y2"]])
            words += g["words"]; wb += g["word_boxes"]
        bx = [min(g["x1"] for g in grp), min(g["y1"] for g in grp),
              max(g["x2"] for g in grp), max(g["y2"] for g in grp)]
        out.append({"text": " ".join(g["text"] for g in grp),
                    "line_box": [[bx[0], bx[1]], [bx[2], bx[1]], [bx[2], bx[3]], [bx[0], bx[3]]],
                    "words": words, "word_boxes": wb})
    out.sort(key=lambda r: r["line_box"][0][1])
    return out


def orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= H_LO) & (H <= H_HI) & (S > S_MIN) & (V > V_MIN)).astype(np.uint8)


def highlight_lines(img, lines):
    lines = merge_rows(lines)
    om = orange_mask(img); out = []
    for l in lines:
        if sum(c.isascii() and c.isalpha() for c in l["text"]) < 6:
            continue
        xs = [p[0] for p in l["line_box"]]; ys = [p[1] for p in l["line_box"]]
        x1, x2, y1, y2 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        if (x2 - x1) < 120 or om[y1:y2 + 6, x1:x2].mean() <= 0.05:
            continue
        out.append({**l, "box": [x1, y1, x2, y2], "ytop": y1, "xleft": x1, "xright": x2})
    out.sort(key=lambda r: r["ytop"])
    return out


# ---------- Stage 2 ----------
def build_montage(img, cands, path):
    om = orange_mask(img); panels = []
    for i, c in enumerate(cands):
        x1, y1, x2, y2 = c["box"]; pad = 10
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - 6):x2 + 6].copy()
        m = om[max(0, y1 - pad):y2 + pad, max(0, x1 - 6):x2 + 6]
        mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
        for cnt in cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) > 60:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(crop, (bx, by), (bx + bw, by + bh), (0, 180, 0), 1)
        h, w = crop.shape[:2]
        crop = cv2.resize(crop, (int(w * ROW_H / h), ROW_H), interpolation=cv2.INTER_CUBIC)
        g = np.full((ROW_H, LABEL_W, 3), 255, np.uint8)
        cv2.putText(g, str(i), (3, ROW_H // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        panels.append(np.hstack([g, crop]))
    W = max(p.shape[1] for p in panels); rows = []
    for p in panels:
        if p.shape[1] < W:
            p = np.hstack([p, np.full((ROW_H, W - p.shape[1], 3), 255, np.uint8)])
        rows += [p, np.full((6, W, 3), 200, np.uint8)]
    cv2.imwrite(path, np.vstack(rows))


PROMPT = (
    "This montage stacks numbered English worksheet lines [0..{n}]. Green boxes mark "
    "algorithm-detected ORANGE pixels, but orange appears BOTH as highlighter AND as "
    "orange-red Chinese pen handwriting / red corrections.\nOCR text per line:\n{texts}\n\n"
    "Return, per line, only the English span(s) covered by the orange HIGHLIGHTER "
    "(ignore Chinese pen, circles, pencil underlines). JSON only: "
    "{{\"<i>\": [\"span\"]}}; [] if none.")


def adjudicate(montage, cands, answer_cache):
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        texts = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(cands))
        b64 = base64.b64encode(open(montage, "rb").read()).decode()
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": PROMPT.format(n=len(cands) - 1, texts=texts)}]}])
        mm = re.search(r"\{.*\}", msg.content[0].text, re.S)
        return json.loads(mm.group(0)) if mm else {}
    return answer_cache


# ---------- Stage 3: assemble + cross-line merge ----------
def span_xrange(line, span):
    """approx [x1,x2] of a span inside its line, via word boxes."""
    toks = [t.strip().lower() for t in line["words"]]
    want = span.lower().split()
    if not want:
        return None
    for s in range(len(toks)):
        if toks[s] == want[0]:
            j, k = s, 0; xs = []
            while j < len(toks) and k < len(want):
                if toks[j] == want[k]:
                    b = line["word_boxes"][j]; xs += [b[0], b[2]]; k += 1
                j += 1
            if k == len(want) and xs:
                return [min(xs), max(xs)]
    return None


def assemble(cands, ans):
    # flat list of (line_idx, span, x1, x2) in reading order
    items = []
    for i, c in enumerate(cands):
        for sp in ans.get(str(i), []):
            xr = span_xrange(c, sp)
            items.append({"li": i, "span": sp, "loc": xr is not None,
                          "x1": xr[0] if xr else c["xleft"],
                          "x2": xr[1] if xr else c["xright"]})
    # cross-line merge: trailing span near right edge of line i + leading span
    # near left edge of line i+1  ->  one wrapped phrase
    merged, used = [], [False] * len(items)
    for idx, it in enumerate(items):
        if used[idx]:
            continue
        cur = dict(it)
        while True:
            li = cur["li"]; line = cands[li]
            near_right = cur.get("loc", True) and \
                cur["x2"] >= line["xright"] - 0.06 * (line["xright"] - line["xleft"])
            nxt = None
            if near_right and li + 1 < len(cands):
                for j, jt in enumerate(items):
                    if used[j] or jt["li"] != li + 1 or not jt["loc"]:
                        continue
                    nl = cands[li + 1]
                    if jt["x1"] <= nl["xleft"] + 0.10 * (nl["xright"] - nl["xleft"]):
                        nxt = (j, jt); break
            if nxt:
                used[nxt[0]] = True
                cur = {"li": li + 1, "span": cur["span"] + " " + nxt[1]["span"],
                       "x1": cur["x1"], "x2": nxt[1]["x2"], "loc": nxt[1]["loc"],
                       "ytop": cands[li]["ytop"]}
            else:
                break
        cur.setdefault("ytop", cands[it["li"]]["ytop"])
        merged.append((cands[it["li"]]["ytop"], it["x1"], cur["span"]))
    merged.sort(key=lambda t: (t[0], t[1]))
    return [m[2] for m in merged]


def main(image, answer_file=None):
    doc = load_doc(image); cv2.imwrite("doc_prod.png", doc)
    cache_ocr = image + ".ocr.json"
    if os.path.exists(cache_ocr):
        lines = json.load(open(cache_ocr))
    else:
        lines = run_ocr("doc_prod.png"); json.dump(lines, open(cache_ocr, "w"), ensure_ascii=False)
    cands = highlight_lines(doc, lines)
    build_montage(doc, cands, "montage_prod.png")
    ans = json.load(open(answer_file)) if (answer_file and os.path.exists(answer_file)) else {}
    ans = adjudicate("montage_prod.png", cands, ans)
    result = assemble(cands, ans)
    json.dump(result, open("result_prod.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(cands)} candidate lines -> 1 call -> {len(result)} items")
    for i, t in enumerate(result, 1):
        print(f"{i:2d}. {t}")
    return cands, result


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
