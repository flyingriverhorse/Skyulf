"""Shared constants for cleaning nodes."""

import string
from typing import Dict

ALIAS_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
COMMON_BOOLEAN_ALIASES: Dict[str, str] = {
    "y": "Yes",
    "yes": "Yes",
    "true": "Yes",
    "1": "Yes",
    "on": "Yes",
    "t": "Yes",
    "affirmative": "Yes",
    "n": "No",
    "no": "No",
    "false": "No",
    "0": "No",
    "off": "No",
    "f": "No",
    "negative": "No",
}
COUNTRY_ALIAS_MAP: Dict[str, str] = {
    "usa": "USA",
    "us": "USA",
    "unitedstates": "USA",
    "unitedstatesofamerica": "USA",
    "states": "USA",
    "america": "USA",
    "unitedkingdom": "United Kingdom",
    "uk": "United Kingdom",
    "greatbritain": "United Kingdom",
    "england": "United Kingdom",
    "uae": "United Arab Emirates",
    "unitedarabemirates": "United Arab Emirates",
    "prc": "China",
    "peoplesrepublicofchina": "China",
    "southkorea": "South Korea",
    "republicofkorea": "South Korea",
    "sk": "South Korea",
}
TWO_DIGIT_YEAR_PIVOT = 50

_REMOVE_SPECIAL_PATTERNS: Dict[str, str] = {
    "keep_alphanumeric": r"[^a-zA-Z0-9]",
    "keep_alphanumeric_space": r"[^a-zA-Z0-9\s]",
    "letters_only": r"[^a-zA-Z]",
    "digits_only": r"[^0-9]",
}
