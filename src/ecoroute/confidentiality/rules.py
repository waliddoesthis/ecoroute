"""L1: deterministic detectors for structured secrets and identifiers.

Each pattern is paired with a validator where one exists (Luhn for cards, mod-97 for
IBANs), so random digit runs are not flagged. Secrets and payment or national IDs are
restricted; contact details are confidential; network identifiers are internal.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ecoroute.confidentiality.levels import Level


@dataclass(frozen=True)
class Finding:
    kind: str
    level: Level
    start: int
    end: int
    layer: str = "rules"
    # A weak finding (a first name, a city) is only sensitive in combination: alone it
    # counts as internal; with a second kind of personal data it keeps its level.
    weak: bool = False

    def describe(self) -> str:
        return f"{self.kind} at {self.start}-{self.end} ({self.level})"


def luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_ok(text: str) -> bool:
    s = re.sub(r"\s", "", text).upper()
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


def ssn_ok(text: str) -> bool:
    area, group, serial = re.findall(r"\d+", text)
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _always(_: str) -> bool:
    return True


@dataclass(frozen=True)
class Rule:
    kind: str
    level: Level
    pattern: re.Pattern
    valid: Callable[[str], bool] = _always


R = Level.RESTRICTED
C = Level.CONFIDENTIAL
I = Level.INTERNAL  # noqa: E741

RULES: list[Rule] = [
    # Secrets (patterns follow gitleaks / detect-secrets).
    Rule("private_key", R, re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    Rule("aws_access_key", R, re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    Rule("github_token", R, re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_\w{22,})")),
    Rule("slack_token", R, re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    Rule("google_api_key", R, re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    Rule("stripe_key", R, re.compile(r"\b[rs]k_(?:live|test)_[0-9A-Za-z]{16,}")),
    Rule("llm_api_key", R, re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}")),
    Rule("jwt", R, re.compile(r"\beyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}")),
    Rule(
        "password_assignment",
        R,
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token)\b"
            r"\s*[:=]\s*['\"]?[^\s'\"]{6,}"
        ),
    ),
    Rule("db_connection_string", R, re.compile(r"\b\w+://[^\s:/@]+:[^\s@/]+@[\w.\-]+")),
    # Payment and national identifiers.
    Rule("credit_card", R, re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), luhn_ok),
    Rule("iban", R, re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"), iban_ok),
    Rule("us_ssn", R, re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ssn_ok),
    # Contact details.
    Rule(
        "email", C, re.compile(r"\b[\w.+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b")
    ),
    Rule(
        "phone",
        C,
        re.compile(r"(?<![\w.])\+?\d{1,3}[ .\-]?\(?\d{2,4}\)?(?:[ .\-]?\d{2,4}){2,3}(?![\w.])"),
        # A bare digit run is more often an ID or timestamp than a phone number.
        lambda s: 9 <= sum(c.isdigit() for c in s) <= 15 and any(c in s for c in "+ .-("),
    ),
    # Network identifiers.
    Rule(
        "ipv4",
        I,
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        lambda s: all(int(p) <= 255 for p in s.split(".")),
    ),
]

_TOKEN = re.compile(r"[A-Za-z0-9+/_\-=]{32,}")


class RulesLayer:
    """Regex + validator rules, high-entropy strings, and a company dictionary.

    dictionary maps terms (client names, project codenames, internal domains) to the
    level they imply; matching is case-insensitive on word boundaries.
    """

    name = "rules"

    def __init__(
        self,
        dictionary: Mapping[str, Level | str] | None = None,
        rules: Iterable[Rule] = RULES,
        entropy_threshold: float = 4.0,
    ) -> None:
        self.rules = list(rules)
        self.entropy_threshold = entropy_threshold
        self.dictionary = []
        for term, level in (dictionary or {}).items():
            pat = re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
            self.dictionary.append((term, Level.parse(level), pat))

    def scan(self, text: str, context: Mapping | None = None) -> list[Finding]:
        found: list[Finding] = []
        for rule in self.rules:
            for m in rule.pattern.finditer(text):
                if rule.valid(m.group()):
                    found.append(Finding(rule.kind, rule.level, m.start(), m.end()))
        for m in _TOKEN.finditer(text):
            tok = m.group()
            mixed = any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)
            if mixed and shannon_entropy(tok) >= self.entropy_threshold:
                found.append(Finding("high_entropy_string", Level.RESTRICTED, m.start(), m.end()))
        for term, level, pat in self.dictionary:
            for m in pat.finditer(text):
                found.append(Finding(f"dictionary:{term}", level, m.start(), m.end()))
        return _drop_nested(found)


def _drop_nested(found: list[Finding]) -> list[Finding]:
    """Keep the strictest finding where spans overlap (a card number is not also a phone)."""
    found.sort(key=lambda f: (-f.level, f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in found:
        if not any(f.start < k.end and k.start < f.end for k in kept):
            kept.append(f)
    return sorted(kept, key=lambda f: f.start)
