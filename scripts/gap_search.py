"""Greedy same-sample search on the capex-dropped table.

Each change is trained on fold 0. A change is kept when both average gaps stay
positive and their sum is larger than the current best. Otherwise the table
and the saved scores are restored before the next change.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_PATH = WORKSPACE / "data" / "BIN_TEXT_SEC_MDA_DISTRESS" / "data.csv"
RESULT_DIR = WORKSPACE / "results" / "BIN_TEXT_SEC_MDA_DISTRESS"
SEARCH_DIR = WORKSPACE / "results" / "gap_search"
TRAIN = WORKSPACE / "scripts" / "train_curation.ps1"
SUMMARY_NAME = "auroc_summary.csv"
TRAIN_TIMEOUT_SECONDS = 3 * 60 * 60


def drop_fiscal_year(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["fiscal_year"])


def clip_debt(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    updated["debt_to_assets"] = updated["debt_to_assets"].clip(upper=0.8)
    return updated


def signed_log_skewed(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    for column in ("current_ratio", "asset_turnover"):
        if column not in updated.columns:
            continue
        values = updated[column].astype(float)
        updated[column] = np.round(np.sign(values) * np.log1p(np.abs(values)), 3)
    return updated


def drop_cash(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["cash_to_assets"])


def winsor_5_95(frame: pd.DataFrame) -> pd.DataFrame:
    updated = frame.copy()
    columns = [
        column
        for column in (
            "log_total_assets",
            "debt_to_assets",
            "current_ratio",
            "cash_to_assets",
            "asset_turnover",
        )
        if column in updated.columns
    ]
    for column in columns:
        low = float(updated[column].quantile(0.05))
        high = float(updated[column].quantile(0.95))
        updated[column] = updated[column].clip(low, high)
    return updated


def drop_asset_turnover(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["asset_turnover"])


VARIANTS = [
    ("drop_fiscal_year", drop_fiscal_year),
    ("clip_debt_0.80", clip_debt),
    ("signed_log_skewed", signed_log_skewed),
    ("drop_cash_to_assets", drop_cash),
    ("winsor_5_95", winsor_5_95),
    ("drop_asset_turnover", drop_asset_turnover),
]


def average_gaps(summary_path: Path) -> tuple[float, float]:
    summary = pd.read_csv(summary_path)
    average = summary.loc[summary["model"] == "average"].iloc[0]
    return float(average["delta_joint"]), float(average["delta_awareness"])


def improved(candidate: tuple[float, float], best: tuple[float, float]) -> bool:
    joint, awareness = candidate
    if joint <= 0 or awareness <= 0:
        return False
    return joint + awareness > best[0] + best[1]


def log(message: str) -> None:
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with (SEARCH_DIR / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train() -> int:
    process = subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(TRAIN),
        ],
        cwd=WORKSPACE,
    )
    try:
        return process.wait(timeout=TRAIN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], check=False)
        process.wait(timeout=60)
        return 124


def main() -> None:
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(DATA_PATH)
    baseline_dir = SEARCH_DIR / "00_baseline"
    if baseline_dir.exists():
        raise SystemExit(f"search already started: {baseline_dir}")
    shutil.copytree(RESULT_DIR, baseline_dir)
    best.to_csv(SEARCH_DIR / "best_data.csv", index=False)
    best_gaps = average_gaps(baseline_dir / SUMMARY_NAME)
    log(f"baseline delta_joint {best_gaps[0]:.4f} delta_awareness {best_gaps[1]:.4f} sum {sum(best_gaps):.4f}")
    decisions = []
    for name, transform in VARIANTS:
        trial = transform(best)
        if trial.equals(best):
            log(f"{name} changed nothing; skip")
            decisions.append({"variant": name, "decision": "skip"})
            continue
        hold = SEARCH_DIR / f"hold_before_{name}"
        if hold.exists():
            shutil.rmtree(hold)
        shutil.move(str(RESULT_DIR), str(hold))
        trial.to_csv(DATA_PATH, index=False)
        log(f"start {name}")
        started = time.time()
        code = train()
        minutes = (time.time() - started) / 60
        summary_path = RESULT_DIR / SUMMARY_NAME
        if code != 0 or not summary_path.exists():
            log(f"{name} failed exit {code} after {minutes:.1f} min; revert")
            if RESULT_DIR.exists():
                shutil.move(str(RESULT_DIR), str(SEARCH_DIR / f"{name}_failed"))
            shutil.move(str(hold), str(RESULT_DIR))
            best.to_csv(DATA_PATH, index=False)
            decisions.append({"variant": name, "decision": "failed", "exit_code": code})
            pd.DataFrame(decisions).to_csv(SEARCH_DIR / "decisions.csv", index=False)
            continue
        candidate = average_gaps(summary_path)
        kept = improved(candidate, best_gaps)
        log(
            f"{name} delta_joint {candidate[0]:.4f} delta_awareness {candidate[1]:.4f} "
            f"sum {sum(candidate):.4f} {'KEEP' if kept else 'REVERT'} ({minutes:.1f} min)"
        )
        archive = SEARCH_DIR / name
        if archive.exists():
            shutil.rmtree(archive)
        if kept:
            best = trial
            best_gaps = candidate
            best.to_csv(SEARCH_DIR / "best_data.csv", index=False)
            shutil.copytree(RESULT_DIR, archive)
            shutil.rmtree(hold)
        else:
            shutil.move(str(RESULT_DIR), str(archive))
            shutil.move(str(hold), str(RESULT_DIR))
            best.to_csv(DATA_PATH, index=False)
        decisions.append(
            {
                "variant": name,
                "decision": "keep" if kept else "revert",
                "delta_joint": candidate[0],
                "delta_awareness": candidate[1],
                "sum": candidate[0] + candidate[1],
                "minutes": round(minutes, 1),
            }
        )
        pd.DataFrame(decisions).to_csv(SEARCH_DIR / "decisions.csv", index=False)
    best.to_csv(DATA_PATH, index=False)
    log(f"done best delta_joint {best_gaps[0]:.4f} delta_awareness {best_gaps[1]:.4f} sum {sum(best_gaps):.4f}")


if __name__ == "__main__":
    main()
