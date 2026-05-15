from __future__ import annotations
from copy import deepcopy

REGIONAL_LEXICON: dict[str, dict[str, str]] = {
    "default": {},
    "jakarta": {
        "Senang": "Senang", "Marah": "Marah", "Terima kasih": "Makasih",
    },
    "yogyakarta": {
        "Senang": "Gembira", "Marah": "Jengkel", "Rumah": "Griya",
        "Terima kasih": "Matur nuwun",
    },
    "surabaya": {
        "Senang": "Bungah", "Kecewa": "Getun", "Makan": "Nedho",
        "Terima kasih": "Suwun",
    },
    "medan": {
        "Besar": "Gadang", "Kecil": "Ketek", "Senang": "Suka",
        "Terima kasih": "Mauliate",
    },
}

def list_regions() -> list[str]:
    return ["Jakarta", "Yogyakarta", "Surabaya", "Medan"]

def apply_regional_mapping(text: str, region: str | None = None) -> str:
    # Region feature removed from UI — always return original word
    return word
    if not text or not region:
        return text
    key = region.strip().lower()
    mapping = REGIONAL_LEXICON.get(key, {})
    return mapping.get(text, text)

def get_region_mapping(region=None):
    if region:
        key = region.strip().lower()
        return deepcopy(REGIONAL_LEXICON.get(key, REGIONAL_LEXICON["default"]))
    return deepcopy(REGIONAL_LEXICON)
