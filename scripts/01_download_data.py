"""
PHASE 1 - SCRIPT 1
Purpose : Download PHOIBLE v2.0 dataset from official GitHub source
Output  : data/raw/phoible.csv
"""

import requests
import os

# ── Configuration ────────────────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv"
)
RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAVE_PATH = os.path.join(RAW_DIR, "phoible.csv")

# ── Download ──────────────────────────────────────────────────────────────────
def download_phoible():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(SAVE_PATH):
        print(f"[INFO] File already exists at: {SAVE_PATH}")
        print("[INFO] Delete it manually if you want a fresh download.")
        return

    print("[INFO] Downloading PHOIBLE v2.0 ...")
    response = requests.get(URL, timeout=120)

    if response.status_code == 200:
        with open(SAVE_PATH, "wb") as f:
            f.write(response.content)
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"[SUCCESS] Downloaded {size_mb:.2f} MB → {SAVE_PATH}")
    else:
        raise ConnectionError(
            f"[ERROR] Failed to download. HTTP status: {response.status_code}"
        )

if __name__ == "__main__":
    download_phoible()
