"""Extract card images from the Card_ID List_EN.pdf.

Extracts the EMBEDDED image from each card's PDF page (not the full page render),
which avoids "[Back to Table]" text and white borders.

Usage:
  python scripts/deck_builder/extract_card_images.py
"""
import json
import os

import pymupdf  # PyMuPDF (not the 'fitz' frontend package)

PDF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "Card_ID List_EN.pdf")
PAGES_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_pages.json")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "card_images")


def extract_card_image(pdf, page_num, out_path):
    """Extract the card image from a PDF page.

    Strategy:
    1. Try extracting embedded images (page.get_images())
    2. Fallback: render page with clip rect that excludes edges
    """
    page = pdf[page_num]

    # Strategy 1: Extract embedded image
    images = page.get_images(full=True)
    if images:
        # Get the largest image (the card, not icons)
        best_img = None
        best_size = 0
        for img_info in images:
            xref = img_info[0]
            try:
                base_image = pdf.extract_image(xref)
                size = base_image["width"] * base_image["height"]
                if size > best_size:
                    best_size = size
                    best_img = base_image
            except Exception:
                continue

        if best_img and best_img["width"] > 100:
            with open(out_path, "wb") as f:
                f.write(best_img["image"])
            return True

    # Strategy 2: Render with clip rect (center 75% of page, skip edges)
    rect = page.rect
    # Clip to center area, excluding top/bottom where links live
    clip = pymupdf.Rect(
        rect.width * 0.10,   # left: skip10%
        rect.height * 0.08,  # top: skip8%
        rect.width * 0.90,   # right: stop at90%
        rect.height * 0.92   # bottom: stop at92%
    )
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), clip=clip)
    pix.save(out_path)
    return True


def main():
    with open(PAGES_JSON) as f:
        card_pages = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = pymupdf.open(PDF_PATH)
    total = len(card_pages)

    print(f"Extracting {total} card images from {PDF_PATH}")
    print(f"Output: {OUT_DIR}/")

    extracted = 0
    for i, (card_id, page_num) in enumerate(card_pages.items()):
        out_path = os.path.join(OUT_DIR, f"{card_id}.jpg")
        if os.path.exists(out_path):
            extracted += 1
            continue

        try:
            extract_card_image(pdf, page_num, out_path)
            extracted += 1
        except Exception as e:
            print(f"  WARNING: card {card_id} page {page_num}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} processed...", flush=True)

    print(f"Done! {extracted}/{total} images in {OUT_DIR}/")


if __name__ == "__main__":
    main()
