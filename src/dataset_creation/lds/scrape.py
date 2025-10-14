#!/usr/bin/env python3
import time
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

BASE_URL = "https://www.learningdisabilityservice-leeds.nhs.uk"
START_URL = f"{BASE_URL}/easy-on-the-i/image-bank/"
OUT_DIR = (Path(__file__).resolve().parent / "../../../data/lds/").resolve()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[403,429,500,502,503])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(HEADERS)
    return s

def get_soup(session, url):
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 403:
            time.sleep(2)
            r = session.get(url, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"[WARN] Failed to GET {url}: {e}")
        return None

def find_next_page(soup):
    nxt = soup.select_one("a.pagination__next")
    return nxt["href"] if nxt else None

def extract_pairs(soup):
    pairs = []
    for item in soup.select("li.image-bank-item"):
        links = item.select("a.btn--download")
        in_box = None
        out_box = None
        for link in links:
            text = link.text.strip().lower()
            if "in the box" in text:
                in_box = link["href"]
            elif "outside the box" in text:
                out_box = link["href"]
        if in_box and out_box:
            pairs.append((in_box, out_box))
    return pairs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = make_session()
    page = START_URL
    count = 0
    visited = set()
    page_num = 1

    while page:
        print(f"[INFO] Page {page_num}: {page}")
        soup = get_soup(session, page)
        if not soup:
            break

        pairs = extract_pairs(soup)
        for in_url, out_url in pairs:
            in_name = Path(urlparse(in_url).path).name  # save OUT file as IN filename
            out_file = OUT_DIR / in_name

            if out_file.exists():
                print(f"[SKIP] {in_name} already exists")
                continue

            out_full = urljoin(BASE_URL, out_url)
            try:
                r = session.get(out_full, timeout=20)
                with open(out_file, "wb") as f:
                    f.write(r.content)
                print(f"[OK] Saved {in_name}")
                count += 1
            except Exception as e:
                print(f"[FAIL] {out_full}: {e}")

            time.sleep(0.4)

        next_page = find_next_page(soup)
        page = next_page if next_page else None
        page_num += 1

    print(f"[DONE] Downloaded {count} images to {OUT_DIR}")

if __name__ == "__main__":
    main()
