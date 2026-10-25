"""Five-fold text-only screen of Item 1A excerpts.

The sentence file is the only local copy of the unselected lead sentences.
It is not row-aligned with data.csv, so this screen uses the file's own
labels. A fixed lexicon picks the sentences. Nothing is fit to the label.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tabstar.preprocessing.splits import split_to_test

WORKSPACE = Path(__file__).resolve().parents[1]
SENTENCE_PATH = WORKSPACE / "material" / "item_1a_first_sentences.txt"
OUT_DIR = WORKSPACE / "results" / "gap_search" / "five_fold_loop"
LEXICON = re.compile(
    r"\b(?:debts?|indebtedness|covenants?|liquidity|defaults?|cash flows?|"
    r"margins?|customers?|suppliers?|demand|going concern|credit|"
    r"borrowings?|leverage)\b",
    re.IGNORECASE,
)
CIK_RE = re.compile(r"CIK\s+(\d+)\s+Fiscal year\s+(\d+)\s+Next-year distress label\s+(\d+)")
COUNT_RE = re.compile(r"(\d+) paragraph sentences\s+(\d+) used")
SENT_RE = re.compile(r"^\s*(\*?)\s*(\d+)\.\s+(.*)$")
MAX_SENTENCES = 6


def parse_documents(path: Path) -> list[dict]:
    documents: list[dict] = []
    current: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = CIK_RE.search(line)
        if match:
            if current is not None:
                documents.append(current)
            current = {
                "cik": match.group(1),
                "fiscal_year": int(match.group(2)),
                "distress_next_year": int(match.group(3)),
                "sentences": [],
            }
            continue
        if current is None:
            continue
        if COUNT_RE.search(line):
            continue
        match = SENT_RE.match(line)
        if match:
            current["sentences"].append((match.group(1) == "*", match.group(3).strip()))
    if current is not None:
        documents.append(current)
    return documents


def kept_excerpt(sentences: list[tuple[bool, str]]) -> str:
    chosen = [sentence for used, sentence in sentences if used]
    return " ".join(chosen)


def lexicon_excerpt(sentences: list[tuple[bool, str]], fallback: str) -> str:
    chosen = []
    seen: set[str] = set()
    for _used, sentence in sentences:
        if LEXICON.search(sentence) is None:
            continue
        key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", sentence.lower())).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) >= MAX_SENTENCES:
            break
    if len(chosen) < 2:
        return fallback
    return " ".join(chosen)


def text_auc(frame: pd.DataFrame, column: str) -> list[float]:
    scores = []
    y = frame["distress_next_year"].astype(int)
    x = frame[[column]].copy()
    for fold in range(5):
        x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=True, fold=fold)
        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
        train_matrix = vectorizer.fit_transform(x_train[column].astype(str))
        test_matrix = vectorizer.transform(x_test[column].astype(str))
        model = LogisticRegression(max_iter=400, solver="liblinear")
        model.fit(train_matrix, y_train)
        probability = model.predict_proba(test_matrix)[:, 1]
        scores.append(float(roc_auc_score(y_test, probability)))
    return scores


def firm_holdout_auc(frame: pd.DataFrame, column: str) -> list[float]:
    """Hold out 10 percent of firms so another year of the same firm cannot leak."""
    scores = []
    firms = frame["cik"].drop_duplicates().tolist()
    for fold in range(5):
        rng_firms = pd.Series(firms).sample(frac=1, random_state=42 + fold).tolist()
        holdout_n = max(1, int(round(0.1 * len(rng_firms))))
        held = set(rng_firms[:holdout_n])
        test = frame[frame["cik"].isin(held)]
        train = frame[~frame["cik"].isin(held)]
        if test["distress_next_year"].nunique() < 2 or train["distress_next_year"].nunique() < 2:
            scores.append(float("nan"))
            continue
        vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=20000)
        train_matrix = vectorizer.fit_transform(train[column].astype(str))
        test_matrix = vectorizer.transform(test[column].astype(str))
        model = LogisticRegression(max_iter=400, solver="liblinear")
        model.fit(train_matrix, train["distress_next_year"].astype(int))
        probability = model.predict_proba(test_matrix)[:, 1]
        scores.append(float(roc_auc_score(test["distress_next_year"].astype(int), probability)))
    return scores


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    documents = parse_documents(SENTENCE_PATH)
    rows = []
    finance_only_outside = 0
    for document in documents:
        kept = kept_excerpt(document["sentences"])
        lexicon = lexicon_excerpt(document["sentences"], kept)
        used_finance = any(used and LEXICON.search(sentence) for used, sentence in document["sentences"])
        other_finance = any((not used) and LEXICON.search(sentence) for used, sentence in document["sentences"])
        finance_only_outside += int(other_finance and not used_finance)
        rows.append(
            {
                "cik": document["cik"],
                "fiscal_year": document["fiscal_year"],
                "distress_next_year": document["distress_next_year"],
                "kept": kept,
                "lexicon": lexicon,
                "replaced": lexicon != kept,
            }
        )
    frame = pd.DataFrame(rows)
    firm_counts = frame.groupby("cik").size()
    diagnostic = {
        "rows": int(len(frame)),
        "unique_firms": int(frame["cik"].nunique()),
        "firms_with_multiple_rows": int((firm_counts > 1).sum()),
        "max_rows_per_firm": int(firm_counts.max()),
        "lexicon_replaced_rows": int(frame["replaced"].sum()),
        "finance_term_only_outside_kept": finance_only_outside,
        "positive_rate": float(frame["distress_next_year"].mean()),
        "note": (
            "This sentence file is not row-aligned with data.csv. "
            "The first filing, CIK 0000003116 fiscal 2017, is absent from the current excerpts."
        ),
    }
    summary_rows = []
    for column in ("kept", "lexicon"):
        row_scores = text_auc(frame, column)
        firm_scores = firm_holdout_auc(frame, column)
        summary_rows.append(
            {
                "excerpt": column,
                "split": "official_row",
                **{f"fold_{fold}": round(score, 4) for fold, score in enumerate(row_scores)},
                "mean": round(sum(row_scores) / len(row_scores), 4),
                "min": round(min(row_scores), 4),
            }
        )
        summary_rows.append(
            {
                "excerpt": column,
                "split": "firm_holdout",
                **{f"fold_{fold}": None if score != score else round(score, 4) for fold, score in enumerate(firm_scores)},
                "mean": round(float(pd.Series(firm_scores).mean()), 4),
                "min": round(float(pd.Series(firm_scores).min()), 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    frame.drop(columns=["kept", "lexicon"]).to_csv(OUT_DIR / "sentence_file_keys.csv", index=False)
    summary.to_csv(OUT_DIR / "text_screen.csv", index=False)
    (OUT_DIR / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
