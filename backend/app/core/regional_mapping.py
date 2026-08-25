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
    """Fitur region sudah dihapus dari UI — selalu kembalikan teks asli tanpa modifikasi.

    CATATAN PERBAIKAN: versi sebelumnya melakukan `return word` padahal
    parameter fungsi ini bernama `text` — `word` tidak pernah didefinisikan,
    sehingga setiap pemanggilan fungsi ini melempar NameError. Karena fungsi
    ini dipanggil setiap kali prediksi mencapai ambang stabilitas, exception
    tersebut merambat ke `ws_endpoint` dan memutus koneksi WebSocket setiap
    kali translasi *hampir* berhasil — inilah penyebab utama gejala
    "tidak bisa mendeteksi".
    """
    return text

def get_region_mapping(region=None):
    if region:
        key = region.strip().lower()
        return deepcopy(REGIONAL_LEXICON.get(key, REGIONAL_LEXICON["default"]))
    return deepcopy(REGIONAL_LEXICON)
