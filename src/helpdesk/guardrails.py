"""Lightweight, fully local guardrails (no external service required).

Input side : prompt-injection / jailbreak detection, harmful-intent detection,
             PII masking (phone, email, Aadhaar-like numbers, register numbers), length limit.
Output side: PII scrubbing so the bot never echoes personal data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts|rules)",
    r"disregard (all |the )?(previous|prior|above)",
    r"(print|reveal|show|repeat|leak).{0,30}(system prompt|instructions|hidden prompt)",
    r"\byou are now\b", r"\bDAN\b", r"developer mode", r"jailbreak",
    r"pretend (you are|to be) .{0,40}(no|without) (rules|restrictions)",
    r"act as .{0,30}(no|without) (rules|restrictions|filters)",
]
HARMFUL_PATTERNS = [
    r"\bhack(ing)?\b", r"\bbypass\b.{0,30}(portal|login|attendance|security)",
    r"change my (attendance|marks|grades)", r"(fake|forge|forged).{0,20}(certificate|medical|leave|signature)",
    r"\bsteal\b", r"\bmalware\b", r"\bpassword of\b",
]
PII_PATTERNS = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PHONE": r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)",
    "REGISTER_NO": r"\b2\d{6}\b",
}
# University contact details that are public and must NOT be masked
ALLOWLIST = re.compile(r"christuniversity\.in|1800-123-3212", re.I)
MAX_CHARS = 1000


@dataclass
class InputVerdict:
    blocked: bool
    clean_text: str
    reason: str = ""
    message: str = ""
    pii_found: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return re.sub(r"(?<=\d)[\s-](?=\d)", "", text.lower())


def mask_pii(text: str, allowed_text: str = "") -> tuple[str, list[str]]:
    """Replace personal identifiers with <LABEL>. Anything that also appears in
    `allowed_text` (e.g. public university phone numbers in the knowledge base) is kept."""
    found = []
    allowed = _norm(allowed_text)
    for label, pat in PII_PATTERNS.items():
        def repl(m: re.Match) -> str:
            if ALLOWLIST.search(m.group(0)) or (allowed and _norm(m.group(0)) in allowed):
                return m.group(0)
            found.append(label)
            return f"<{label}>"
        text = re.sub(pat, repl, text)
    return text, found


def check_input(text: str) -> InputVerdict:
    t = text.strip()
    if not t:
        return InputVerdict(True, t, "empty", "Please type a question.")
    if len(t) > MAX_CHARS:
        return InputVerdict(True, t[:MAX_CHARS], "too_long",
                            f"Your message is too long. Please keep it under {MAX_CHARS} characters.")
    clean, pii = mask_pii(t)
    low = clean.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, clean, flags=re.I):
            return InputVerdict(True, clean, "prompt_injection",
                                "I can only help with questions about CHRIST University student services, "
                                "academics and administration. I can't change my instructions.", pii)
    for pat in HARMFUL_PATTERNS:
        if re.search(pat, low, flags=re.I):
            return InputVerdict(True, clean, "harmful_request",
                                "I can't help with that. If you have an attendance or marks concern, please "
                                "speak to your class teacher or the Office of Examinations.", pii)
    return InputVerdict(False, clean, pii_found=pii)


def scrub_output(text: str, allowed_text: str = "") -> str:
    return mask_pii(text, allowed_text)[0]
