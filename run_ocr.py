"""Step 1: Run PaddleOCR (PP-OCRv6, latest 飞桨/PaddlePaddle) on the worksheet,
with word-level boxes, and cache the result to ocr_cache.json so the
highlight-matching algorithm can iterate quickly without reloading models."""
import cv2, numpy as np, json
from paddleocr import PaddleOCR


def main():
    img = cv2.imread("worksheet.jpg")
    # Crop away the dark phone-screenshot bars, keep the paper region.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rowmean = gray.mean(axis=1)
    bright = np.where(rowmean > 110)[0]
    y0, y1 = max(0, bright.min() - 10), min(img.shape[0], bright.max() + 10)
    crop = img[y0:y1, :]
    cv2.imwrite("doc.png", crop)

    ocr = PaddleOCR(lang="en", enable_mkldnn=False)
    res = ocr.predict("doc.png", return_word_box=True)
    r = res[0]

    lines = []
    for i in range(len(r["rec_texts"])):
        words = r["text_word"][i]
        wboxes = [np.array(b).tolist() for b in r["text_word_boxes"][i]]
        lines.append({
            "text": r["rec_texts"][i],
            "score": float(r["rec_scores"][i]),
            "line_box": np.array(r["rec_polys"][i]).tolist(),
            "words": words,
            "word_boxes": wboxes,
        })
    json.dump({"crop_y0": int(y0), "lines": lines},
              open("ocr_cache.json", "w"), ensure_ascii=False, indent=1)
    print(f"cached {len(lines)} lines -> ocr_cache.json")


if __name__ == "__main__":
    main()
