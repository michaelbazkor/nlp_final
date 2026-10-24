"""Prefetch SEC ZIPs and EDGAR JSONL files with the standard library only."""

from build_sec_mda import (
    EDGAR_SPLITS,
    EDGAR_URL,
    EDGAR_DIR,
    SEC_URL,
    TEXT_YEARS,
    ZIP_DIR,
    download,
    quarters,
)
import os


def main() -> None:
    for quarter in quarters():
        download(SEC_URL.format(quarter=quarter), os.path.join(ZIP_DIR, f"{quarter}.zip"))
    for year in TEXT_YEARS:
        for split in EDGAR_SPLITS:
            dest = os.path.join(EDGAR_DIR, str(year), f"{split}.jsonl")
            download(EDGAR_URL.format(year=year, split=split), dest)
    print("prefetch complete")


if __name__ == "__main__":
    main()
