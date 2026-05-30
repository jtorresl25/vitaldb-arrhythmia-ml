"""Binary label helpers for consistent Spanish display in the UI.

Internal logic always uses 'normal' / 'abnormal' (lowercase English).
The UI always shows 'Normal' / 'Anormal' (Spanish).
"""


def normalize_binary_label(x) -> str:
    """Normalize any binary-label form to 'normal' or 'abnormal'.

    Accepted inputs: "N", "normal", "Normal", "0",
                     "abnormal", "Anormal", "anormal", "1", "anomal",
                     "no N", "not N", etc.
    Returns: always "normal" or "abnormal".
    """
    s = str(x).strip().lower()
    if s in ("n", "normal", "0"):
        return "normal"
    return "abnormal"


def display_binary_label(x) -> str:
    """Return the Spanish capitalized label for UI display.

    "normal" / any normal form  →  "Normal"
    "abnormal" / any abnormal form  →  "Anormal"
    """
    return "Normal" if normalize_binary_label(x) == "normal" else "Anormal"


def display_binary_label_lower(x) -> str:
    """Return the lowercase Spanish label for inline body text.

    "normal" form  →  "normal"
    "abnormal" form  →  "anormal"
    """
    return "normal" if normalize_binary_label(x) == "normal" else "anormal"


def binary_label_badge_kind(x) -> str:
    """Return the badge/accent kind for a binary label.

    normal   →  "ok"   (teal / green)
    abnormal →  "err"  (red / warning)
    """
    return "ok" if normalize_binary_label(x) == "normal" else "err"
