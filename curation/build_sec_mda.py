"""Join SEC 10-K financial ratios to compressed Item 1A risk-factor text.

Writes data/BIN_TEXT_SEC_MDA_DISTRESS/data.csv, metadata.json, and provenance.json.
Raw SEC ZIPs and EDGAR JSONL files stay in the user cache, outside OneDrive.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

DATASET_NAME = "BIN_TEXT_SEC_MDA_DISTRESS"
USER_AGENT = "nlp-course-multabench academic research AdminContact@example.com"
SEC_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip"
EDGAR_URL = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{year}/{split}.jsonl"
EDGAR_REPO = "https://huggingface.co/datasets/eloukas/edgar-corpus"

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "nlp-sec")
ZIP_DIR = os.path.join(CACHE, "zips")
PARQUET_DIR = os.path.join(CACHE, "facts")
EDGAR_DIR = os.path.join(CACHE, "edgar")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(WORKSPACE, "data", DATASET_NAME)

SEC_YEARS = range(2015, 2022)  # 2015 supplies YoY growth; 2021 supplies the 2020 label
TEXT_YEARS = range(2016, 2021)
FEATURE_YEARS = range(2016, 2021)
EDGAR_SPLITS = ("train", "test", "validate")
APPLE_CIK = "0000320193"
APPLE_FY = 2018
MAX_ROWS = 1000
MAX_DEBT_TO_ASSETS = None
MIN_SECTION_WORDS = 40
RANDOM_SEED = 42
LEAD_MIN_WORDS = 8
LEAD_MAX_WORDS = 42
LEAD_MAX_NUMERIC_FRACTION = 0.12
MAX_LEAD_SENTENCES = 10
SENTENCE_DF_MAX = 0.15
DATE_PREFIX_RE = re.compile(
    r"^(As of|For the year ended|In|During)\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},?\s*",
    re.IGNORECASE,
)
BOILERPLATE_PHRASES = (
    "forward-looking statements",
    "item 1a",
    "investing in our",
    "high degree of risk",
    "carefully consider",
    "securities act",
    "stock price",
    "common stock",
    "cyber",
    "information technology",
    "attract and retain",
    "key personnel",
    "tax laws",
    "general economic",
    "natural disasters",
    "climate change",
    "anti-takeover",
    "delaware law",
    "sarbanes-oxley",
    "internal control over financial",
    "dividends",
    "securities analysts",
)
CONTENT_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "been",
    "have", "has", "had", "our", "ours", "their", "they", "them", "these", "those",
    "which", "who", "whom", "whose", "into", "over", "under", "than", "then", "also",
    "other", "more", "most", "its", "not", "any", "such", "about", "after", "before",
    "between", "within", "without", "during", "would", "could", "will", "may", "into",
}
RISK_TERM_RE = re.compile(
    r"\b(?:indebtedness|covenant|liquidity|credit|default|going concern|margin|litigation|customer concentration|supply)\b",
    re.IGNORECASE,
)
CHANGE_COLUMNS = ["income_change", "coverage_change", "ocf_change"]
MAX_YEARS_PER_FIRM = 2
MIN_YEAR_GAP = 2
SKEWED_RATIOS = ["current_ratio", "asset_turnover"]
POSITIVE_TAIL = 0.25
NEGATIVE_TAIL = 0.55
EXTREME_TRIM = 0.0

INSTANT_ALIASES = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalents", "Cash"],
    "receivables": ["AccountsReceivableNetCurrent", "AccountsReceivableNet", "ReceivablesNetCurrent"],
}
DURATION_ALIASES = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RegulatedOperatingRevenue",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "interest": ["InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
}

NUMERIC_COLUMNS = [
    "log_total_assets",
    "debt_to_assets",
    "current_ratio",
    "cash_to_assets",
    "asset_turnover",
    "capex_to_assets",
]
DROPPED_FEATURES = [
    "operating_margin",
    "interest_coverage",
    "quick_ratio",
    "return_on_assets",
    "gross_margin",
    "debt_to_equity",
    "capex_to_revenue",
    "capex_to_assets",
    "yoy_revenue_growth",
    "sic_division",
]
SIC_DIVISIONS = [
    (100, 999, "Agriculture"),
    (1000, 1499, "Mining"),
    (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"),
    (4000, 4999, "Transportation and Utilities"),
    (5000, 5199, "Wholesale Trade"),
    (5200, 5999, "Retail Trade"),
    (6000, 6799, "Finance"),
    (7000, 8999, "Services"),
    (9100, 9729, "Public Administration"),
]


def quarters() -> list[str]:
    return [f"{year}q{quarter}" for year in SEC_YEARS for quarter in range(1, 5)]


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    partial = dest + ".part"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            print(f"downloading {url}")
            with urlopen(request, timeout=180) as response, open(partial, "wb") as handle:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    handle.write(block)
            os.replace(partial, dest)
            return
        except Exception as exc:
            last_error = exc
            if os.path.exists(partial):
                os.remove(partial)
            print(f"retry {attempt + 1} for {url}: {exc}")
    raise RuntimeError(f"failed to download {url}") from last_error


def _blank(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.isin(["", "nan", "None", "NaN"])


def _digits(value: object) -> str | None:
    raw = "".join(ch for ch in str(value) if ch.isdigit())
    if not raw:
        return None
    return raw.zfill(10) if len(raw) <= 10 else raw


def sic_division(value: object) -> str:
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "Nonclassifiable"
    for low, high, name in SIC_DIVISIONS:
        if low <= code <= high:
            return name
    return "Nonclassifiable"


def _read_header(zf: zipfile.ZipFile, name: str) -> list[str]:
    with zf.open(name) as handle:
        header = handle.readline().decode("utf-8", errors="replace")
    return header.rstrip("\n").split("\t")


def parse_quarter(quarter: str) -> pd.DataFrame:
    cached = os.path.join(PARQUET_DIR, f"{quarter}.parquet")
    if os.path.exists(cached):
        cached_frame = pd.read_parquet(cached)
        if "ocf" in cached_frame.columns:
            return cached_frame

    zip_path = os.path.join(ZIP_DIR, f"{quarter}.zip")
    download(SEC_URL.format(quarter=quarter), zip_path)
    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise RuntimeError(f"{zip_path} is not a zip file")

    with zipfile.ZipFile(zip_path) as zf:
        sub_cols = _read_header(zf, "sub.txt")
        num_cols = _read_header(zf, "num.txt")
        sub_usecols = [c for c in ["adsh", "cik", "sic", "form", "fy", "fp", "filed", "period"] if c in sub_cols]
        num_usecols = [c for c in ["adsh", "tag", "version", "ddate", "qtrs", "uom", "coreg", "value", "segments"] if c in num_cols]
        sub = pd.read_csv(zf.open("sub.txt"), sep="\t", dtype=str, usecols=sub_usecols, low_memory=False)

        sub["form"] = sub["form"].fillna("").str.strip().str.upper()
        sub["fp"] = sub["fp"].fillna("").str.strip().str.upper()
        sub["fy_n"] = pd.to_numeric(sub["fy"], errors="coerce")
        sub = sub[sub["form"].isin(["10-K", "10-K/A"]) & sub["fp"].eq("FY")]
        sub = sub[sub["fy_n"].between(min(SEC_YEARS), max(SEC_YEARS))]
        sub["cik"] = sub["cik"].map(_digits)
        sub = sub.dropna(subset=["cik", "adsh"])
        allowed = set(sub["adsh"])
        wanted_tags = {tag for aliases in (*INSTANT_ALIASES.values(), *DURATION_ALIASES.values()) for tag in aliases}

        pieces = []
        reader = pd.read_csv(
            zf.open("num.txt"),
            sep="\t",
            dtype=str,
            usecols=num_usecols,
            chunksize=400_000,
            low_memory=False,
        )
        for chunk in reader:
            chunk = chunk[chunk["adsh"].isin(allowed) & chunk["tag"].isin(wanted_tags)]
            if chunk.empty:
                continue
            chunk = chunk[chunk["uom"].fillna("").str.upper().eq("USD")]
            if "coreg" in chunk.columns:
                chunk = chunk[_blank(chunk["coreg"])]
            if "segments" in chunk.columns:
                chunk = chunk[_blank(chunk["segments"])]
            if not chunk.empty:
                pieces.append(chunk)

    if not pieces:
        wide = sub.iloc[0:0][["adsh", "cik", "sic", "form", "fy_n", "filed"]].rename(columns={"fy_n": "fy"})
        for concept in (*INSTANT_ALIASES, *DURATION_ALIASES):
            wide[concept] = pd.Series(dtype="float64")
        os.makedirs(PARQUET_DIR, exist_ok=True)
        wide.to_parquet(cached, index=False)
        print(f"{quarter}: no matching facts")
        return wide

    facts = pd.concat(pieces, ignore_index=True)
    facts = facts.merge(sub, on="adsh", how="inner")
    facts["qtrs_n"] = pd.to_numeric(facts["qtrs"], errors="coerce")
    instant_tags = {tag for aliases in INSTANT_ALIASES.values() for tag in aliases}
    duration_tags = {tag for aliases in DURATION_ALIASES.values() for tag in aliases}
    instant = facts[facts["tag"].isin(instant_tags) & facts["qtrs_n"].eq(0)]
    duration = facts[facts["tag"].isin(duration_tags) & facts["qtrs_n"].eq(4)]
    facts = pd.concat([instant, duration], ignore_index=True)
    facts["value_n"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.dropna(subset=["value_n"])
    facts["ddate_s"] = facts["ddate"].fillna("").str.replace(r"\.0$", "", regex=True)
    facts["period_s"] = facts["period"].fillna("").str.replace(r"\.0$", "", regex=True)
    facts["exact"] = facts["ddate_s"].eq(facts["period_s"])
    facts["gaap"] = facts["version"].fillna("").str.startswith("us-gaap")
    facts = facts.sort_values(["exact", "gaap"], ascending=False)
    facts = facts.drop_duplicates(["adsh", "tag"], keep="first")

    wide_parts = []
    for concept, aliases in {**INSTANT_ALIASES, **DURATION_ALIASES}.items():
        priority = {alias: index for index, alias in enumerate(aliases)}
        part = facts[facts["tag"].isin(priority)].copy()
        if part.empty:
            continue
        part["priority"] = part["tag"].map(priority)
        part = part.sort_values("priority").drop_duplicates("adsh", keep="first")
        wide_parts.append(part.set_index("adsh")["value_n"].rename(concept))

    meta_cols = ["cik", "sic", "form", "fy_n", "filed"]
    meta = sub.drop_duplicates("adsh").set_index("adsh")[meta_cols]
    wide = meta.join(wide_parts, how="left").reset_index()
    wide = wide.rename(columns={"fy_n": "fy"})
    os.makedirs(PARQUET_DIR, exist_ok=True)
    wide.to_parquet(cached, index=False)
    print(f"{quarter}: {len(wide)} 10-K filings")
    return wide


def load_financials() -> pd.DataFrame:
    frames = [parse_quarter(quarter) for quarter in quarters()]
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["cik", "fy"])
    raw["fy"] = raw["fy"].astype(int)
    raw["filed_n"] = pd.to_numeric(raw["filed"], errors="coerce").fillna(0)
    raw["is_amendment"] = raw["form"].eq("10-K/A")
    raw = raw.sort_values(["is_amendment", "filed_n"], ascending=[True, False])
    raw = raw.drop_duplicates(["cik", "fy"], keep="first")
    return raw.reset_index(drop=True)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator.abs() > 0)
    return numerator / denom


def add_ratios(filings: pd.DataFrame) -> pd.DataFrame:
    frame = filings.copy()
    assets = frame["assets"].where(frame["assets"] > 0)
    equity = frame["equity"].where(frame["equity"] != 0)
    current_liab = frame["liabilities_current"].where(frame["liabilities_current"].abs() > 0)
    revenue = frame["revenue"].where(frame["revenue"] != 0)
    interest = frame["interest"].abs().where(frame["interest"].abs() > 1)
    quick_numerator = frame["cash"].fillna(0) + frame["receivables"].fillna(0)
    missing_both = frame["cash"].isna() & frame["receivables"].isna()
    quick_numerator = quick_numerator.mask(missing_both)

    frame["log_total_assets"] = np.log(assets)
    frame["debt_to_assets"] = _safe_div(frame["liabilities"], assets)
    frame["debt_to_equity"] = _safe_div(frame["liabilities"], equity)
    frame["current_ratio"] = _safe_div(frame["assets_current"], current_liab)
    frame["quick_ratio"] = _safe_div(quick_numerator, current_liab)
    frame["return_on_assets"] = _safe_div(frame["net_income"], assets)
    frame["gross_margin"] = _safe_div(frame["gross_profit"], revenue)
    frame["operating_margin"] = _safe_div(frame["operating_income"], revenue)
    frame["asset_turnover"] = _safe_div(frame["revenue"], assets)
    frame["interest_coverage"] = _safe_div(frame["operating_income"], interest)
    frame["cash_to_assets"] = _safe_div(frame["cash"], assets)
    frame["capex_to_assets"] = _safe_div(frame["capex"].abs(), assets)
    frame["capex_to_revenue"] = _safe_div(frame["capex"].abs(), frame["revenue"].abs())
    frame["sic_division"] = frame["sic"].map(sic_division)

    prior = frame[["cik", "fy", "revenue"]].rename(columns={"fy": "fy_next", "revenue": "revenue_prior"})
    prior["fy"] = prior["fy_next"] + 1
    frame = frame.merge(prior[["cik", "fy", "revenue_prior"]], on=["cik", "fy"], how="left")
    frame["yoy_revenue_growth"] = frame["revenue"] / frame["revenue_prior"].where(frame["revenue_prior"].abs() > 0) - 1

    nxt = frame[["cik", "fy", "operating_income", "interest", "ocf"]].rename(
        columns={
            "fy": "fy_prev",
            "operating_income": "opinc_next",
            "interest": "interest_next",
            "ocf": "ocf_next",
        }
    )
    nxt["fy"] = nxt["fy_prev"] - 1
    frame = frame.merge(
        nxt[["cik", "fy", "opinc_next", "interest_next", "ocf_next"]],
        on=["cik", "fy"],
        how="left",
    )
    income_base = frame["operating_income"]
    frame["income_change"] = (frame["opinc_next"] - income_base) / income_base.abs().where(income_base.abs() > 0)
    interest_t = frame["interest"].abs()
    interest_n = frame["interest_next"].abs()
    coverage_t = frame["operating_income"] / interest_t.where(interest_t > 1)
    coverage_n = frame["opinc_next"] / interest_n.where(interest_n > 1)
    frame["coverage_change"] = coverage_n - coverage_t
    ocf_base = frame["ocf"]
    frame["ocf_change"] = (frame["ocf_next"] - ocf_base) / ocf_base.abs().where(ocf_base.abs() > 0)
    return frame


def shock_label(frame: pd.DataFrame) -> pd.Series:
    """1 only when year t+1 is a deterioration from year t, not a firm that stays weak."""
    interest_t = frame["interest"].abs()
    interest_n = frame["interest_next"].abs()
    coverage_t = frame["operating_income"] / interest_t.where(interest_t > 1)
    coverage_n = frame["opinc_next"] / interest_n.where(interest_n > 1)
    coverage_crosses = (coverage_t >= 1) & (coverage_n < 1)
    coverage_defined = coverage_t.notna() & coverage_n.notna()

    income_base = frame["operating_income"]
    income_change = (frame["opinc_next"] - income_base) / income_base.abs().where(income_base.abs() > 0)
    income_drop = income_change <= -0.30
    income_defined = income_change.notna()

    cash_turns_negative = (frame["ocf"] >= 0) & (frame["ocf_next"] < 0)
    cash_defined = frame["ocf"].notna() & frame["ocf_next"].notna()

    fired = (
        coverage_crosses.fillna(False)
        | income_drop.fillna(False)
        | cash_turns_negative.fillna(False)
    )
    defined = coverage_defined | income_defined | cash_defined
    return fired.where(defined).astype("Float64")


def choose_label(frame: pd.DataFrame) -> tuple[pd.Series, str, float]:
    labels = shock_label(frame)
    rate = float(labels.mean())
    return labels, "deterioration_shock", rate


def word_count(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def first_sentence(paragraph: str) -> str:
    match = re.match(r".+?[.!?](?:\s|$)", paragraph)
    if match:
        return match.group(0).strip()
    return paragraph


def numeric_fraction(sentence: str) -> float:
    if not sentence:
        return 1.0
    return sum(character.isdigit() for character in sentence) / len(sentence)


def keep_lead(sentence: str) -> bool:
    word_total = len(sentence.split())
    if word_total < LEAD_MIN_WORDS or word_total > LEAD_MAX_WORDS:
        return False
    lowered = sentence.lower()
    if any(phrase in lowered for phrase in BOILERPLATE_PHRASES):
        return False
    return numeric_fraction(sentence) < LEAD_MAX_NUMERIC_FRACTION


def sentence_key(sentence: str) -> str:
    letters = re.sub(r"[^a-z0-9\s]", "", sentence.lower())
    return re.sub(r"\s+", " ", letters).strip()


def candidate_leads(text: object) -> str:
    """Lead sentences that pass the local filters, before the corpus-frequency cut."""
    if not isinstance(text, str):
        return ""
    kept = []
    for paragraph in text.split("\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        sentence = DATE_PREFIX_RE.sub("", first_sentence(paragraph)).strip()
        if keep_lead(sentence):
            kept.append(sentence)
    return "\n".join(kept)


LEVERAGE_TERM_RE = re.compile(
    r"\b(?:indebtedness|covenants?|liquidity|credit|default|borrowings?|leverage|debt)\b",
    re.IGNORECASE,
)


def apply_sentence_idf(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop generic leads, then keep the first 10 in document order.

    survivor_blob keeps every sentence that passed the frequency cut, in document
    order, so a later step can pick leverage sentences without changing this sample.
    """
    lists = frame["lead_blob"].fillna("").map(lambda text: [part for part in str(text).split("\n") if part])
    n_filings = len(frame)
    sentence_counts: dict[str, int] = {}
    word_counts: dict[str, int] = {}
    parsed_rows = []
    next_mark = 10
    print("progress 0% idf counts", flush=True)
    for index, sentences in enumerate(lists, start=1):
        row = []
        filing_sentences: set[str] = set()
        filing_words: set[str] = set()
        for sentence in sentences:
            key = sentence_key(sentence)
            tokens = [
                token for token in re.findall(r"[a-z]{4,}", sentence.lower())
                if token not in CONTENT_STOPWORDS
            ]
            row.append((sentence, key, tokens))
            if key:
                filing_sentences.add(key)
            filing_words.update(tokens)
        for key in filing_sentences:
            sentence_counts[key] = sentence_counts.get(key, 0) + 1
        for token in filing_words:
            word_counts[token] = word_counts.get(token, 0) + 1
        parsed_rows.append(row)
        percent = 100 * index // max(n_filings, 1)
        if percent >= next_mark:
            print(f"progress {next_mark}% ({index}/{n_filings}) idf counts", flush=True)
            next_mark += 10

    sentence_cutoff = SENTENCE_DF_MAX * n_filings
    frequent_sentences = {key for key, count in sentence_counts.items() if count > sentence_cutoff}
    excerpts = []
    survivor_blobs = []
    word_totals = []
    dropped_exact = 0
    dropped_generic = 0
    next_mark = 10
    print("progress 0% idf selection", flush=True)
    for index, row in enumerate(parsed_rows, start=1):
        chosen = []
        seen: set[str] = set()
        for sentence, key, tokens in row:
            if not key or key in seen:
                continue
            if key in frequent_sentences:
                dropped_exact += 1
                continue
            if not tokens:
                dropped_generic += 1
                continue
            mean_share = sum(word_counts[token] for token in tokens) / len(tokens) / n_filings
            if mean_share > SENTENCE_DF_MAX:
                dropped_generic += 1
                continue
            seen.add(key)
            chosen.append(sentence)
        excerpts.append(" ".join(chosen[:MAX_LEAD_SENTENCES]))
        survivor_blobs.append("\n".join(chosen))
        word_totals.append(word_count(excerpts[-1]))
        percent = 100 * index // max(n_filings, 1)
        if percent >= next_mark:
            print(f"progress {next_mark}% ({index}/{n_filings}) idf selection", flush=True)
            next_mark += 10
    filtered = frame.copy()
    filtered["risk_excerpt"] = excerpts
    filtered["survivor_blob"] = survivor_blobs
    filtered["risk_words"] = word_totals
    info = {
        "filings_scored": n_filings,
        "exact_sentence_types_above_15pct": len(frequent_sentences),
        "exact_sentence_occurrences_dropped": dropped_exact,
        "generic_word_occurrences_dropped": dropped_generic,
        "max_lead_sentences": MAX_LEAD_SENTENCES,
        "selection": "first 10 specific leads in document order",
    }
    print(
        f"idf kept the first {MAX_LEAD_SENTENCES} specific leads; "
        f"dropped {dropped_generic} generic sentences",
        flush=True,
    )
    return filtered, info


def leverage_excerpt(survivors: object, fallback: str) -> str:
    """Up to 10 leverage sentences that already passed the frequency cut."""
    chosen = []
    seen: set[str] = set()
    for sentence in str(survivors or "").split("\n"):
        sentence = sentence.strip()
        if not sentence or LEVERAGE_TERM_RE.search(sentence) is None:
            continue
        key = sentence_key(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) >= MAX_LEAD_SENTENCES:
            break
    if not chosen:
        return fallback
    return " ".join(chosen)


def load_edgar_year(year: int) -> pd.DataFrame:
    cached = os.path.join(EDGAR_DIR, f"{year}-section1a-idf.parquet")
    if os.path.exists(cached):
        return pd.read_parquet(cached)
    rows = []
    for split in EDGAR_SPLITS:
        dest = os.path.join(EDGAR_DIR, str(year), f"{split}.jsonl")
        download(EDGAR_URL.format(year=year, split=split), dest)
        with open(dest, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                section = item.get("section_1A")
                rows.append(
                    {
                        "cik": _digits(item.get("cik")),
                        "edgar_year": int(item["year"]) if str(item.get("year", "")).isdigit() else year,
                        "lead_blob": candidate_leads(section),
                        "section_words": word_count(section),
                    }
                )
    frame = pd.DataFrame(rows).dropna(subset=["cik"])
    frame = frame.sort_values("section_words", ascending=False).drop_duplicates(["cik", "edgar_year"], keep="first")
    os.makedirs(EDGAR_DIR, exist_ok=True)
    frame.to_parquet(cached, index=False)
    print(f"edgar {year}: {len(frame)} filings")
    return frame


def choose_year_shift(financials: pd.DataFrame, text: pd.DataFrame) -> tuple[int, dict[str, int]]:
    keys = financials.loc[financials["fy"].isin(FEATURE_YEARS), ["cik", "fy"]]
    fin_keys = set(zip(keys["cik"], keys["fy"].astype(int)))
    overlaps = {}
    for shift in (-1, 0, 1):
        shifted = set(zip(text["cik"], text["edgar_year"].astype(int) + shift))
        overlaps[str(shift)] = len(fin_keys & shifted)
    best = max((-1, 0, 1), key=lambda shift: overlaps[str(shift)])
    if best != 0 and overlaps[str(best)] < overlaps["0"] * 1.1:
        best = 0
    return best, overlaps


def winsorize(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    cuts = {}
    for column in columns:
        low = float(frame[column].quantile(0.01))
        high = float(frame[column].quantile(0.99))
        cuts[column] = {"p01": low, "p99": high}
        if low < high:
            frame[column] = frame[column].clip(low, high)
    return cuts


def stratified_sample(frame: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    counts = frame[label].value_counts().sort_index()
    raw = counts / counts.sum() * n
    alloc = np.floor(raw).astype(int)
    leftover = int(n - alloc.sum())
    order = (raw - alloc).sort_values(ascending=False)
    for cls in order.index:
        if leftover <= 0:
            break
        alloc.loc[cls] += 1
        leftover -= 1
    parts = []
    for cls, size in alloc.items():
        group = frame[frame[label] == cls]
        chosen = rng.choice(group.index.to_numpy(), size=int(min(size, len(group))), replace=False)
        parts.append(frame.loc[chosen])
    sampled = pd.concat(parts)
    order_idx = rng.permutation(len(sampled))
    return sampled.iloc[order_idx].reset_index(drop=True)


def complete_cases(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    kept = frame.dropna(subset=columns).copy()
    kept = kept[kept["section_words"] > MIN_SECTION_WORDS]
    kept = kept[kept["risk_words"] > 0]
    kept = kept[kept["fy"].isin(FEATURE_YEARS)]
    return kept


def purify_boundary(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Worst 25 percent is distress and best 55 percent is healthy, unless an extreme trim removes the outer tails."""
    parts = frame[CHANGE_COLUMNS]
    scale = parts.std(ddof=0).replace(0, np.nan)
    standardized = (parts - parts.mean()) / scale
    score = standardized.mean(axis=1, skipna=True)
    valid = frame.loc[score.notna()].copy()
    score = score.loc[valid.index]
    rank = score.rank(method="first", ascending=True)
    n_rows = len(valid)
    n_extreme = int(np.floor(n_rows * EXTREME_TRIM))
    n_positive = int(np.floor(n_rows * POSITIVE_TAIL))
    n_negative = int(np.floor(n_rows * NEGATIVE_TAIL))
    labels = pd.Series(pd.NA, index=valid.index, dtype="Float64")
    labels.loc[(rank > n_extreme) & (rank <= n_positive)] = 1
    labels.loc[(rank > n_rows - n_negative) & (rank <= n_rows - n_extreme)] = 0
    valid["distress_next_year"] = labels
    kept = valid[valid["distress_next_year"].notna()].copy()
    info = {
        "eligible_for_boundary": n_rows,
        "extreme_trim": n_extreme,
        "positive_tail": int((labels == 1).sum()),
        "negative_tail": int((labels == 0).sum()),
        "rows_after_boundary": int(len(kept)),
    }
    return kept, info


def cap_at_most(frame: pd.DataFrame, max_years: int) -> tuple[pd.DataFrame, dict]:
    """Keep at most max_years fiscal years for each firm."""
    rng = np.random.default_rng(RANDOM_SEED)
    chosen_index = []
    capped = 0
    for _, group in frame.groupby("cik", sort=False):
        years = sorted({int(year) for year in group["fy"]})
        if len(years) > max_years:
            capped += 1
            picked = rng.choice(years, size=max_years, replace=False)
            chosen = [int(year) for year in picked]
        else:
            chosen = years
        chosen_index.extend(group.index[group["fy"].isin(chosen)].tolist())
    kept = frame.loc[chosen_index].copy()
    info = {
        "max_years_per_firm": max_years,
        "firms": int(frame["cik"].nunique()),
        "firms_capped": capped,
        "rows_after_firm_cap": int(len(kept)),
    }
    return kept, info


def cap_firm_years(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep at most two fiscal years per firm, preferring years at least two apart."""
    rng = np.random.default_rng(RANDOM_SEED)
    chosen_index = []
    firms = 0
    firms_with_gap = 0
    firms_single = 0
    firms_adjacent_only = 0
    for _, group in frame.groupby("cik", sort=False):
        firms += 1
        years = sorted({int(year) for year in group["fy"]})
        if len(years) == 1:
            firms_single += 1
            chosen = years
        else:
            pairs = [
                (left, right)
                for index, left in enumerate(years)
                for right in years[index + 1 :]
                if right - left >= MIN_YEAR_GAP
            ]
            if pairs:
                firms_with_gap += 1
                left, right = pairs[int(rng.integers(0, len(pairs)))]
                chosen = [left, right]
            else:
                firms_adjacent_only += 1
                chosen = [years[int(rng.integers(0, len(years)))]]
        chosen_index.extend(group.index[group["fy"].isin(chosen)].tolist())
    kept = frame.loc[chosen_index].copy()
    info = {
        "max_years_per_firm": MAX_YEARS_PER_FIRM,
        "min_year_gap": MIN_YEAR_GAP,
        "firms": firms,
        "firms_with_two_years_at_least_two_apart": firms_with_gap,
        "firms_single_year": firms_single,
        "firms_adjacent_only_kept_one_year": firms_adjacent_only,
        "rows_after_firm_cap": int(len(kept)),
    }
    return kept, info


def signed_log_round(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame[column].astype(float)
        frame[column] = np.round(np.sign(values) * np.log1p(np.abs(values)), 3)


def longitudinal_sample(frame: pd.DataFrame, n: int) -> tuple[pd.DataFrame, dict]:
    """Fill the sample with label-flipping firms, then other multi-year firms, then single years."""
    rng = np.random.default_rng(RANDOM_SEED)
    buckets: dict[int, list[pd.DataFrame]] = {0: [], 1: [], 2: []}
    for _, group in frame.groupby("cik", sort=False):
        labels = {int(value) for value in group["distress_next_year"]}
        n_years = int(group["fy"].nunique())
        if n_years >= 2 and labels == {0, 1}:
            rank = 0
        elif n_years >= 2:
            rank = 1
        else:
            rank = 2
        buckets[rank].append(group)
    for rank, groups in buckets.items():
        order = rng.permutation(len(groups))
        buckets[rank] = [groups[index] for index in order]

    chosen: list[pd.DataFrame] = []
    taken = {0: 0, 1: 0, 2: 0}
    for rank in (0, 1, 2):
        for group in buckets[rank]:
            remaining = n - sum(len(part) for part in chosen)
            if remaining <= 0:
                break
            if len(group) <= remaining:
                chosen.append(group)
                taken[rank] += len(group)
                continue
            if rank == 0 and remaining >= 2:
                positive = group[group["distress_next_year"] == 1]
                negative = group[group["distress_next_year"] == 0]
                parts = [positive.sample(1, random_state=RANDOM_SEED), negative.sample(1, random_state=RANDOM_SEED)]
                used = pd.concat(parts).index
                rest = group.drop(index=used)
                extra = remaining - 2
                if extra > 0 and len(rest):
                    parts.append(rest.sample(min(extra, len(rest)), random_state=RANDOM_SEED))
                piece = pd.concat(parts)
            else:
                piece = group.sample(remaining, random_state=RANDOM_SEED)
            chosen.append(piece)
            taken[rank] += len(piece)
            break
        if sum(len(part) for part in chosen) >= n:
            break
    sampled = pd.concat(chosen).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    info = {
        "rows_from_label_flipping_firms": taken[0],
        "rows_from_other_multiyear_firms": taken[1],
        "rows_from_single_year_firms": taken[2],
        "flipping_firms_available": len(buckets[0]),
        "other_multiyear_firms_available": len(buckets[1]),
        "single_year_firms_available": len(buckets[2]),
    }
    return sampled, info


def random_sample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return frame.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    os.makedirs(ZIP_DIR, exist_ok=True)
    filings = add_ratios(load_financials())

    text = pd.concat([load_edgar_year(year) for year in TEXT_YEARS], ignore_index=True)
    text, idf_info = apply_sentence_idf(text)
    shift, overlaps = choose_year_shift(filings, text)
    text["fy"] = text["edgar_year"] + shift
    apple = text[(text["cik"] == APPLE_CIK) & (text["edgar_year"] == APPLE_FY)]
    apple_words = int(apple["section_words"].iloc[0]) if len(apple) else 0

    merged = filings.merge(
        text[["cik", "fy", "risk_excerpt", "risk_words", "section_words"]],
        on=["cik", "fy"],
        how="inner",
    )
    required = list(NUMERIC_COLUMNS)
    eligible = complete_cases(merged, required)
    curated, boundary_info = purify_boundary(eligible)
    before_sample = len(curated)
    pool_positive_rate = float(curated["distress_next_year"].mean()) if before_sample else float("nan")
    if before_sample > MAX_ROWS:
        curated = random_sample(curated, MAX_ROWS)
    cuts = winsorize(curated, required) if len(curated) else {}
    curated["fiscal_year"] = curated["fy"].astype(int)
    rows_before_debt_cap = len(curated)
    if MAX_DEBT_TO_ASSETS is not None and len(curated):
        curated = curated.loc[curated["debt_to_assets"] <= MAX_DEBT_TO_ASSETS].reset_index(drop=True)
    positive_rate = float(curated["distress_next_year"].mean()) if len(curated) else float("nan")

    output_columns = [column for column in required if column not in DROPPED_FEATURES]
    output_columns += ["fiscal_year", "risk_excerpt", "distress_next_year"]
    output = curated[output_columns].copy()
    output["distress_next_year"] = output["distress_next_year"].astype(int)
    os.makedirs(OUT_DIR, exist_ok=True)
    output.to_csv(os.path.join(OUT_DIR, "data.csv"), index=False)
    metadata = {"target": "distress_next_year", "task_type": "cls", "image_col": None}
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    provenance = {
        "dataset": DATASET_NAME,
        "structured_source": SEC_URL,
        "text_source": EDGAR_REPO,
        "sec_quarters": quarters(),
        "edgar_years": list(TEXT_YEARS),
        "feature_years": list(FEATURE_YEARS),
        "join_keys": ["cik", "fy"],
        "year_shift_added_to_edgar_year": shift,
        "overlap_by_shift": overlaps,
        "apple_cik": APPLE_CIK,
        "apple_fy": APPLE_FY,
        "apple_item1a_words_in_edgar_year": apple_words,
        "text_section": "section_1A",
        "text_compression": (
            "first sentence of each Item 1A paragraph, after dropping the risk-factor legend, "
            "generic boilerplate phrases, sentences outside 8 to 42 words, sentences that are "
            "at least 12 percent digits, sentences copied into more than 15 percent of filings, "
            "and sentences whose content words appear on average in more than 15 percent of filings; "
            "the first 10 remaining sentences are kept"
        ),
        "idf": idf_info,
        "boundary": boundary_info,
        "min_section_words_exclusive": MIN_SECTION_WORDS,
        "sample_size": MAX_ROWS,
        "sample_seed": RANDOM_SEED,
        "sample_method": (
            "uniform random 1000 with sample seed 42, no firm cap, original ratio scale"
            + (
                ", then drop rows with debt_to_assets above 0.8"
                if MAX_DEBT_TO_ASSETS is not None
                else ""
            )
        ),
        "max_debt_to_assets": MAX_DEBT_TO_ASSETS,
        "rows_before_debt_cap": int(rows_before_debt_cap),
        "rows_dropped_debt_cap": int(rows_before_debt_cap - len(curated)),
        "sample_seed": RANDOM_SEED,
        "e5_prefix": (
            "MulTaBench already encodes text as 'passage: risk_excerpt: ...'. "
            "The CSV does not add a second passage prefix."
        ),
        "filings_after_dedup": int(len(filings)),
        "unique_firms": int(filings["cik"].nunique()),
        "edgar_rows": int(len(text)),
        "inner_join_rows": int(len(merged)),
        "rows_before_sample": int(before_sample),
        "rows_after_long_item1a_filter": int(len(eligible)),
        "rows_written": int(len(output)),
        "label_rule": "boundary_tails",
        "label_positive_rate_before_sample": pool_positive_rate,
        "label_positive_rate": positive_rate,
        "class_counts": output["distress_next_year"].value_counts().sort_index().astype(int).to_dict() if len(output) else {},
        "rows_after_boundary": int(boundary_info["rows_after_boundary"]),
        "columns_kept": output_columns,
        "columns_removed_from_features": DROPPED_FEATURES,
        "columns_dropped_as_identifiers_or_levels": [
            "cik", "adsh", "name", "sic", "form", "filed", "assets", "liabilities", "equity",
            "revenue", "operating_income", "net_income", "interest", "capex", "ocf",
            "opinc_next", "interest_next", "ocf_next",
        ],
        "required_column_relaxations": [],
        "winsor_cutpoints": cuts,
        "target": "distress_next_year",
        "target_definition": (
            "A lower composite of the year-t to year-t+1 changes in operating income, "
            "interest coverage, and operating cash flow is worse. The worst 25 percent "
            "of that score is 1 and the best 55 percent is 0. The middle 20 percent is excluded."
        ),
    }
    with open(os.path.join(OUT_DIR, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: provenance[k] for k in [
        "rows_written", "label_rule", "label_positive_rate", "year_shift_added_to_edgar_year",
        "overlap_by_shift", "required_column_relaxations", "class_counts",
    ]}, indent=2))


if __name__ == "__main__":
    main()
