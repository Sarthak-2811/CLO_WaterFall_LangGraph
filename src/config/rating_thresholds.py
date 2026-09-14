"""
Rating-agency idealized loss thresholds.

These are simplified, illustrative approximations of the kind of "idealized
cumulative loss rate" tables rating agencies (S&P/Moody's/Fitch) publish for
CLO liabilities. They are NOT the real published tables -- they exist so the
Critic agent has a concrete, per-tranche bar for EVERY rated class, instead
of only ever checking the single most-senior tranche (the bug in the
original implementation).
"""
import re
from typing import Optional, Tuple

RATING_LOSS_THRESHOLDS = {
    "AAA": 0.0001,
    "AA":  0.0005,
    "A":   0.0015,
    "BBB": 0.0075,
    "BB":  0.03,
    "B":   0.08,
}

CLASS_NAME_RATING_FALLBACK = [
    (re.compile(r"class\s*a[-\s]?1", re.I), "AAA"),
    (re.compile(r"class\s*a[-\s]?2", re.I), "AAA"),
    (re.compile(r"class\s*a\b", re.I), "AAA"),
    (re.compile(r"class\s*b\b", re.I), "AA"),
    (re.compile(r"class\s*c\b", re.I), "A"),
    (re.compile(r"class\s*d\b", re.I), "BBB"),
    (re.compile(r"class\s*e\b", re.I), "BB"),
    (re.compile(r"class\s*f\b", re.I), "B"),
]


def normalize_rating(raw: Optional[str]) -> Optional[str]:
    """'Aaa (sf)' / 'AAA+' / 'Baa2' -> a bucket key in RATING_LOSS_THRESHOLDS."""
    if not raw:
        return None
    r = raw.strip().upper().replace("(SF)", "").strip()
    r = re.split(r"[\s+\-\d]", r)[0]
    ordered = [
        ("AAA", "AAA"), ("AA", "AA"),
        ("BAA", "BBB"), ("BBB", "BBB"),
        ("BA", "BB"), ("BB", "BB"),
        ("A", "A"), ("B", "B"),
    ]
    for prefix, bucket in ordered:
        if r.startswith(prefix):
            return bucket
    return None


def threshold_for_tranche(tranche: dict) -> Tuple[Optional[float], Optional[str], bool]:
    """Returns (loss_threshold, rating_bucket, was_inferred)."""
    if tranche.get("is_equity"):
        return None, "NR/Equity", False

    stated = normalize_rating(tranche.get("target_rating"))
    if stated:
        return RATING_LOSS_THRESHOLDS.get(stated), stated, False

    class_name = tranche.get("class_name", "")
    for pattern, bucket in CLASS_NAME_RATING_FALLBACK:
        if pattern.search(class_name):
            return RATING_LOSS_THRESHOLDS.get(bucket), bucket, True

    return None, None, False
