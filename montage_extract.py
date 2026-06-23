"""Stage-2 v2: single-call montage adjudication with colour-mask hints.

Instead of one multimodal request per highlight-bearing line, this stacks every
candidate line into ONE labelled montage image (each line tagged [n], with the
algorithm's detected orange regions outlined as a hint) and sends a SINGLE
request that returns the highlighted English spans for all lines at once.

=> 1 multimodal round-trip per worksheet (vs N), much lower latency/cost.

Live call uses ANTHROPIC_API_KEY (Claude). Offline, it falls back to a cached
montage answer so the pipeline still runs end-to-end.
"""
import cv2, numpy as np, json, re, os, base64, sys

H_LO, H_HI, S_MIN, V_MIN = 10, 26, 13, 90
ROW_H = 78          # px height each line is scaled to in the montage
LABEL_W = 46        # left gutter for the [n] index


def orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= H_LO) & (H <= H_HI) & (S > S_MIN) & (V > V_MIN)).astype(np.uint8)


def highlight_lines(img, lines):
    om = orange_mask(img)
    out = []
    for l in lines:
        if sum(c.isascii() and c.isalpha() for c in l["text"]) < 6:
            continue
        xs = [p[0] for p in l["line_box"]]; ys = [p[1] for p in l["line_box"]]
        x1, x2 = int(min(xs)), int(max(xs)); y1, y2 = int(min(ys)), int(max(ys))
        if (x2 - x1) < 120 or om[y1:y2 + 6, x1:x2].mean() <= 0.05:
            continue
        out.append({"text": l["text"], "box": [x1, y1, x2, y2], "ytop": y1})
    out.sort(key=lambda r: r["ytop"])
    return out


def line_panel(img, om, box, idx):
    """One montage row: the line crop with detected-orange regions outlined
    (green) as a hint, an index label, scaled to ROW_H."""
    x1, y1, x2, y2 = box
    pad = 10
    crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - 6):x2 + 6].copy()
    m = om[max(0, y1 - pad):y2 + pad, max(0, x1 - 6):x2 + 6]
    # outline orange connected components so the model sees the algo's hint
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) > 60:
            bx, by, bw, bh = cv2.boundingRect(c)
            cv2.rectangle(crop, (bx, by), (bx + bw, by + bh), (0, 180, 0), 1)
    h, w = crop.shape[:2]
    scale = ROW_H / h
    crop = cv2.resize(crop, (int(w * scale), ROW_H), interpolation=cv2.INTER_CUBIC)
    gutter = np.full((ROW_H, LABEL_W, 3), 255, np.uint8)
    cv2.putText(gutter, str(idx), (3, ROW_H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    return np.hstack([gutter, crop])


def build_montage(img, cands, path):
    om = orange_mask(img)
    panels = [line_panel(img, om, c["box"], i) for i, c in enumerate(cands)]
    W = max(p.shape[1] for p in panels)
    rows = []
    for p in panels:
        if p.shape[1] < W:
            p = np.hstack([p, np.full((ROW_H, W - p.shape[1], 3), 255, np.uint8)])
        rows.append(p)
        rows.append(np.full((6, W, 3), 200, np.uint8))   # separator
    cv2.imwrite(path, np.vstack(rows))


PROMPT = (
    "This montage stacks numbered lines [0..{n}] cropped from a student's English "
    "worksheet. Green boxes mark where an algorithm detected ORANGE pixels — but "
    "orange appears BOTH as highlighter marks AND as orange-red Chinese pen "
    "handwriting (translations) and red corrections.\n"
    "For each line, here is the OCR text:\n{texts}\n\n"
    "Task: for every line, return which English word(s)/phrase(s) are marked by the "
    "orange HIGHLIGHTER only (ignore the Chinese pen handwriting, circles, and "
    "underlines/pencil). Respect exact span boundaries.\n"
    "Reply ONLY with JSON: {{\"<index>\": [\"span\", ...], ...}}. Empty list if none."
)


def adjudicate_montage(path, cands, cache):
    texts = "\n".join(f"[{i}] {c['text']}" for i, c in enumerate(cands))
    prompt = PROMPT.format(n=len(cands) - 1, texts=texts)
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=1500,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
                {"type": "text", "text": prompt}]}])
        m = re.search(r"\{.*\}", msg.content[0].text, re.S)
        return json.loads(m.group(0)) if m else {}
    return cache            # offline: bundled montage answer


def main(doc="doc2.png", cache_file="ocr_cache2.json",
         answer="montage_answer.json", out="result_montage.json"):
    img = cv2.imread(doc)
    lines = json.load(open(cache_file))["lines"]
    cands = highlight_lines(img, lines)
    build_montage(img, cands, "montage.png")
    cache = json.load(open(answer)) if os.path.exists(answer) else {}
    ans = adjudicate_montage("montage.png", cands, cache)
    result = []
    for i in range(len(cands)):
        result.extend(ans.get(str(i), []))
    json.dump(result, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(cands)} lines -> 1 montage call -> {len(result)} highlighted items")
    for i, t in enumerate(result, 1):
        print(f"{i:2d}. {t}")
    return result


if __name__ == "__main__":
    main(*sys.argv[1:])
