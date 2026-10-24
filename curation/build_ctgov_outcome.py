"""Build the ClinicalTrials.gov strategy-1 pilot table.

Downloads interventional phase 2/3 drug trials and writes 1,000 rows whose
structured columns and text columns do not restate each other. Each text
value is packed so the E5 passage string is at most 512 tokens.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
from transformers import AutoTokenizer

WORKSPACE = Path(__file__).resolve().parents[1]
OUT_DIR = WORKSPACE / "data" / "BIN_TEXT_CTGOV_TRIAL_OUTCOME"
CACHE_DIR = Path.home() / ".cache" / "nlp-ctgov"
API = "https://clinicaltrials.gov/api/v2/studies"
FILTER = (
    "AREA[StudyType]INTERVENTIONAL AND AREA[InterventionType]DRUG "
    "AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) "
    "AND AREA[StartDate]RANGE[2015-01-01,2025-12-31]"
)
STATUSES = "COMPLETED,TERMINATED,WITHDRAWN"
FIELDS = ",".join(
    [
        "NCTId",
        "OverallStatus",
        "Phase",
        "DesignAllocation",
        "DesignInterventionModel",
        "DesignMasking",
        "DesignPrimaryPurpose",
        "LeadSponsorClass",
        "ArmGroupType",
        "MinimumAge",
        "MaximumAge",
        "Sex",
        "HealthyVolunteers",
        "EligibilityCriteria",
        "BriefSummary",
        "InterventionName",
        "InterventionDescription",
        "LocationCountry",
    ]
)
PAGE_SIZE = 1000
SAMPLE_SIZE = 1000
SAMPLE_SEED = 42
MIN_ELIGIBILITY_WORDS = 40
MAX_TOKENS = 512
E5_NAME = "intfloat/e5-small-v2"
ELIGIBILITY_COL = "eligibility_criteria"
MECHANISM_COL = "study_mechanism"
TARGET = "completed"
STRUCTURED = [
    "phase",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
    "lead_sponsor_class",
    "n_arms",
    "n_facilities",
    "n_countries",
    "min_age_years",
    "max_age_years",
    "sex_eligibility",
    "healthy_volunteers",
]
USER_AGENT = "nlp-ctgov-curation"

CLINICAL_RE = re.compile(
    r"\b(ecog|karnofsky|nyha|anc|neutrophil|platelet|hemoglob|hgb|egfr|"
    r"biomarker|mutation|biopsy|histolog|metastatic|stage|prior|chemotherap|"
    r"radiation|\bmg\b|creatinine|bilirubin|\bast\b|\balt\b|\binr\b|pregnan|"
    r"contracept|tumor|tumour|cancer|disease|infection|\bhiv\b|hepatitis|"
    r"washout|therap|treatment|diagnos|lesion|receptor|antibody|enzyme|"
    r"protein|gene|viral|renal|hepatic|cardiac|qtc|lvef)\b",
    re.I,
)
AGE_RE = re.compile(
    r"(\byears?\s+of\s+age\b|\byears?\s+old\b|\baged\b|≥\s*\d+|\>=\s*\d+|"
    r"\d+\s*(?:to|-|–)\s*\d+\s*years?)",
    re.I,
)
SEX_RE = re.compile(r"\b(males?|females?|men|women|both sexes)\b", re.I)
HEALTHY_RE = re.compile(r"healthy volunteers?", re.I)
DESIGN_RE = re.compile(
    r"("
    r"\bphase\s*(?:[1-4]|i{1,3}v?|iv)\b"
    r"|randomi[sz]"
    r"|double[- ]blind|single[- ]blind|triple[- ]blind|quadruple[- ]blind"
    r"|open[- ]label"
    r"|placebo[- ]controlled"
    r"|multi[- ]cent(?:er|re)"
    r"|(?:approximately|up to|about|a total of)\s+\d+\s+"
    r"(?:patients|subjects|participants)"
    r"|\b\d+\s+(?:patients|subjects|participants)\s+will be\s+"
    r"(?:enrolled|recruited)"
    r"|\b\d+\s+(?:sites?|centers?|centres?)\b"
    r")",
    re.I,
)
EXCLUSION_HEADER_RE = re.compile(r"exclusion", re.I)
AGE_VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)",
    re.I,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def log(message: str) -> None:
    print(message, flush=True)


def fetch_page(page_token: str | None) -> dict:
    params = {
        "pageSize": PAGE_SIZE,
        "countTotal": "true",
        "filter.overallStatus": STATUSES,
        "filter.advanced": FILTER,
        "fields": FIELDS,
    }
    if page_token:
        params["pageToken"] = page_token
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ClinicalTrials.gov page failed: {last_error}")


def download_pages() -> list[Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"pages": [], "next_token": None, "complete": False, "total": None}
    while not manifest["complete"]:
        page_index = len(manifest["pages"])
        payload = fetch_page(manifest["next_token"])
        page_path = CACHE_DIR / f"page_{page_index:03d}.json"
        page_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest["pages"].append(page_path.name)
        manifest["next_token"] = payload.get("nextPageToken")
        manifest["complete"] = not manifest["next_token"]
        if manifest["total"] is None:
            manifest["total"] = payload.get("totalCount")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        log(
            f"cached page {page_index} studies {len(payload.get('studies', []))} "
            f"total {manifest['total']}"
        )
    return [CACHE_DIR / name for name in manifest["pages"]]


def parse_age(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    match = AGE_VALUE_RE.search(value)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("year"):
        return number
    if unit.startswith("month"):
        return number / 12
    if unit.startswith("week"):
        return number / 52
    if unit.startswith("day"):
        return number / 365
    if unit.startswith("hour"):
        return number / (365 * 24)
    return number / (365 * 24 * 60)


def is_demographic(line: str) -> bool:
    if CLINICAL_RE.search(line):
        return False
    words = line.split()
    if HEALTHY_RE.search(line) and len(words) <= 30:
        return True
    if AGE_RE.search(line) and len(words) <= 24:
        return True
    return bool(SEX_RE.search(line) and len(words) <= 16)


def clean_eligibility_lines(text: str) -> list[str]:
    kept: list[str] = []
    for raw in text.splitlines():
        line = raw.strip(" \t-*•")
        if not line:
            continue
        if is_demographic(line):
            continue
        if DESIGN_RE.search(line) and not CLINICAL_RE.search(line):
            continue
        kept.append(line)
    return kept


def split_exclusion(lines: list[str]) -> tuple[list[str], list[str]]:
    for index, line in enumerate(lines):
        if EXCLUSION_HEADER_RE.search(line) and len(line) <= 80:
            return lines[:index], lines[index:]
    return lines, []


def mechanism_sentences(brief: str, interventions: list[dict]) -> list[str]:
    pieces = [brief]
    for item in interventions:
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name:
            pieces.append(name + ".")
        if description:
            pieces.append(description)
    sentences: list[str] = []
    for piece in pieces:
        flat = re.sub(r"\s+", " ", piece).strip()
        if not flat:
            continue
        for sentence in SENTENCE_RE.split(flat):
            sentence = sentence.strip()
            if sentence and not DESIGN_RE.search(sentence):
                sentences.append(sentence)
    return sentences


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(E5_NAME, local_files_only=True)
    tokenizer.model_max_length = 10**6
    return tokenizer


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def passage(column: str, body: str) -> str:
    return f"passage: {column}: {body}"


def truncate_body(tokenizer, column: str, body: str) -> str:
    ids = tokenizer.encode(body, add_special_tokens=False)
    lo, hi = 0, len(ids)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        piece = tokenizer.decode(ids[:mid], skip_special_tokens=True).strip()
        if n_tokens(tokenizer, passage(column, piece)) <= MAX_TOKENS:
            best = piece
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def head_that_fits(tokenizer, column: str, parts: list[str], sep: str) -> str:
    if not parts:
        return ""
    lo, hi = 0, len(parts)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        body = sep.join(parts[:mid]).strip()
        if n_tokens(tokenizer, passage(column, body)) <= MAX_TOKENS:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best == 0:
        return truncate_body(tokenizer, column, parts[0])
    return sep.join(parts[:best]).strip()


def pack_eligibility(tokenizer, text: str) -> tuple[str, str]:
    lines = clean_eligibility_lines(text)
    if not lines:
        return "", "empty"
    original = "\n".join(lines)
    if n_tokens(tokenizer, passage(ELIGIBILITY_COL, original)) <= MAX_TOKENS:
        return original, "unchanged"
    inclusion, exclusion = split_exclusion(lines)
    if exclusion:
        packed = head_that_fits(tokenizer, ELIGIBILITY_COL, exclusion + inclusion, "\n")
        return packed, "exclusion_first"
    packed = head_that_fits(tokenizer, ELIGIBILITY_COL, lines, "\n")
    return packed, "tail_trim"


def pack_mechanism(tokenizer, brief: str, interventions: list[dict]) -> tuple[str, str]:
    sentences = mechanism_sentences(brief, interventions)
    if not sentences:
        return "", "empty"
    original = " ".join(sentences)
    if n_tokens(tokenizer, passage(MECHANISM_COL, original)) <= MAX_TOKENS:
        return original, "unchanged"
    return head_that_fits(tokenizer, MECHANISM_COL, sentences, " "), "tail_trim"


def study_row(study: dict, tokenizer) -> tuple[dict | None, str]:
    protocol = study.get("protocolSection") or {}
    status = (protocol.get("statusModule") or {}).get("overallStatus")
    if status not in {"COMPLETED", "TERMINATED", "WITHDRAWN"}:
        return None, "status"
    locations_module = protocol.get("contactsLocationsModule")
    if not isinstance(locations_module, dict) or "locations" not in locations_module:
        return None, "locations"
    locations = locations_module.get("locations") or []
    countries = {
        str(item.get("country")).strip()
        for item in locations
        if isinstance(item, dict) and item.get("country")
    }
    identification = protocol.get("identificationModule") or {}
    design = protocol.get("designModule") or {}
    info = design.get("designInfo") or {}
    masking = (info.get("maskingInfo") or {}).get("masking")
    phases = design.get("phases") or []
    if not any(phase in {"PHASE2", "PHASE3"} for phase in phases):
        return None, "phase"
    eligibility = protocol.get("eligibilityModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    eligibility_text, eligibility_mode = pack_eligibility(
        tokenizer, str(eligibility.get("eligibilityCriteria") or "")
    )
    if len(eligibility_text.split()) < MIN_ELIGIBILITY_WORDS:
        return None, "short_eligibility"
    mechanism_text, mechanism_mode = pack_mechanism(
        tokenizer,
        str((protocol.get("descriptionModule") or {}).get("briefSummary") or ""),
        arms.get("interventions") or [],
    )
    if n_tokens(tokenizer, passage(ELIGIBILITY_COL, eligibility_text)) > MAX_TOKENS:
        raise RuntimeError(f"eligibility over budget {identification.get('nctId')}")
    if n_tokens(tokenizer, passage(MECHANISM_COL, mechanism_text)) > MAX_TOKENS:
        raise RuntimeError(f"mechanism over budget {identification.get('nctId')}")
    arm_groups = arms.get("armGroups")
    row = {
        "nct_id": identification.get("nctId"),
        "phase": "|".join(phases),
        "allocation": info.get("allocation"),
        "intervention_model": info.get("interventionModel"),
        "masking": masking,
        "primary_purpose": info.get("primaryPurpose"),
        "lead_sponsor_class": (
            (protocol.get("sponsorCollaboratorsModule") or {}).get("leadSponsor") or {}
        ).get("class"),
        "n_arms": None if arm_groups is None else len(arm_groups),
        "n_facilities": len(locations),
        "n_countries": len(countries),
        "min_age_years": parse_age(eligibility.get("minimumAge")),
        "max_age_years": parse_age(eligibility.get("maximumAge")),
        "sex_eligibility": eligibility.get("sex"),
        "healthy_volunteers": eligibility.get("healthyVolunteers"),
        ELIGIBILITY_COL: eligibility_text,
        MECHANISM_COL: mechanism_text,
        TARGET: 1 if status == "COMPLETED" else 0,
        "eligibility_pack": eligibility_mode,
        "mechanism_pack": mechanism_mode,
    }
    return row, "kept"


def build_frame(page_paths: list[Path], tokenizer) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = {
        "studies": 0,
        "kept": 0,
        "status": 0,
        "locations": 0,
        "phase": 0,
        "short_eligibility": 0,
        "duplicate": 0,
    }
    rows: list[dict] = []
    seen: set[str] = set()
    for path in page_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for study in payload.get("studies", []):
            counts["studies"] += 1
            row, reason = study_row(study, tokenizer)
            if row is None:
                counts[reason] += 1
                continue
            nct_id = row["nct_id"]
            if nct_id in seen:
                counts["duplicate"] += 1
                continue
            seen.add(nct_id)
            rows.append(row)
            counts["kept"] += 1
        log(f"parsed {path.name} kept {counts['kept']}")
    return pd.DataFrame(rows), counts


def stratified_sample(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame[TARGET] == 1]
    stopped = frame[frame[TARGET] == 0]
    if completed.empty or stopped.empty:
        raise RuntimeError("both classes are required")
    rate = len(completed) / len(frame)
    n_completed = int(round(SAMPLE_SIZE * rate))
    n_stopped = SAMPLE_SIZE - n_completed
    if n_completed > len(completed) or n_stopped > len(stopped):
        raise RuntimeError(
            f"not enough rows for {n_completed} completed and {n_stopped} stopped"
        )
    sample = pd.concat(
        [
            completed.sample(n=n_completed, random_state=SAMPLE_SEED),
            stopped.sample(n=n_stopped, random_state=SAMPLE_SEED),
        ]
    )
    return sample.sample(frac=1, random_state=SAMPLE_SEED).reset_index(drop=True)


def main() -> None:
    page_paths = download_pages()
    tokenizer = load_tokenizer()
    frame, counts = build_frame(page_paths, tokenizer)
    sample = stratified_sample(frame)
    for column in (ELIGIBILITY_COL, MECHANISM_COL):
        lengths = [
            n_tokens(tokenizer, passage(column, str(value)))
            for value in sample[column].fillna("")
        ]
        if max(lengths) > MAX_TOKENS:
            raise RuntimeError(f"{column} exceeds {MAX_TOKENS}")
    model_frame = sample[STRUCTURED + [ELIGIBILITY_COL, MECHANISM_COL, TARGET]].copy()
    model_frame["healthy_volunteers"] = model_frame["healthy_volunteers"].map(
        {True: 1, False: 0}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(OUT_DIR / "data.csv", index=False)
    metadata = {"target": TARGET, "task_type": "cls", "image_col": None}
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "source": API,
        "filter": FILTER,
        "statuses": STATUSES.split(","),
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "min_eligibility_words": MIN_ELIGIBILITY_WORDS,
        "max_tokens": MAX_TOKENS,
        "tokenizer": E5_NAME,
        "counts": counts,
        "pool_completed": int((frame[TARGET] == 1).sum()),
        "pool_stopped": int((frame[TARGET] == 0).sum()),
        "sample_completed": int((sample[TARGET] == 1).sum()),
        "sample_stopped": int((sample[TARGET] == 0).sum()),
        "eligibility_pack": sample["eligibility_pack"].value_counts().to_dict(),
        "mechanism_pack": sample["mechanism_pack"].value_counts().to_dict(),
        "structured_columns": STRUCTURED,
        "text_columns": [ELIGIBILITY_COL, MECHANISM_COL],
        "dropped_fields": {
            "enrollment_count": "Posted count is actual accrual. Withdrawn trials are 0.",
            "enrollment_type": "ACTUAL versus ESTIMATED tracks whether the trial finished.",
            "why_stopped": "Termination reason.",
            "start_date_and_completion_dates": "Calendar year and duration are outcome-adjacent shortcuts.",
            "condition_mesh_keywords": "Coded copy of the disease, which stays in the text.",
            "intervention_name_as_column": "Drug name stays inside study_mechanism.",
            "sponsor_name": "High cardinality would be detected as a text column.",
            "nct_id": "Identifier. Stored here, not in the model table.",
            "official_title_and_brief_title": "Mostly phase and design wording.",
            "results_arm_text_masking_text": "Post-outcome text or a restatement of the design columns.",
        },
        "nct_ids": sample["nct_id"].tolist(),
    }
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    log(
        f"wrote {len(model_frame)} rows "
        f"completed {provenance['sample_completed']} stopped {provenance['sample_stopped']} "
        f"eligibility packs {provenance['eligibility_pack']}"
    )


if __name__ == "__main__":
    main()
