"""
Sarvam AI Bulbul v2 Text-to-Speech (TTS) Client.

Features:
- Native Indic Text-to-Speech via Sarvam AI API (`model="bulbul:v2"`).
- Supports Hindi (`hi-IN`), Tamil (`ta-IN`), English (`en-IN`), and other Indic languages.
- Direct base64 WAV audio output for seamless browser audio streaming.
- Fast-path fallback when API key is missing or in offline test mode.
"""

import logging
import requests
import re
from typing import Any, Dict, Optional
import config
from stt.sarvam_client import get_sarvam_language_code

logger = logging.getLogger(__name__)


def normalize_phonetic_acronyms(text: str) -> str:
    """
    Expands common uppercase technical acronyms (e.g. HTTP, API, JSON, SQL)
    into hyphenated letter sequences so Indic/English TTS pronounces them clearly.
    """
    known_acronyms = {
        'HTTP': 'H-T-T-P',
        'HTTPS': 'H-T-T-P-S',
        'API': 'A-P-I',
        'APIs': 'A-P-Is',
        'URL': 'U-R-L',
        'URLs': 'U-R-Ls',
        'HTML': 'H-T-M-L',
        'CSS': 'C-S-S',
        'JSON': 'J-S-O-N',
        'SQL': 'S-Q-L',
        'XML': 'X-M-L',
        'SDK': 'S-D-K',
        'SDKs': 'S-D-Ks',
        'TCP': 'T-C-P',
        'IP': 'I-P',
        'CPU': 'C-P-U',
        'GPU': 'G-P-U',
        'RAM': 'R-A-M',
        'STT': 'S-T-T',
        'TTS': 'T-T-S',
        'UI': 'U-I',
        'UX': 'U-X',
        'HNSW': 'H-N-S-W',
        'SLA': 'S-L-A',
        'LLM': 'L-L-M',
        'LLMs': 'L-L-Ms',
        'SLM': 'S-L-M',
        'SVD': 'S-V-D',
        'BM25': 'B-M 25',
        'GraphQL': 'Graph Q L',
        'FastAPI': 'Fast A-P-I',
        'Node.js': 'Node J S',
        'Node.JS': 'Node J S',
    }
    t = text
    for k, v in known_acronyms.items():
        t = re.sub(r'\b' + re.escape(k) + r'\b', v, t)
    
    # Generic uppercase 2-5 letter abbreviations without vowels
    def expand_abbr(m):
        w = m.group(1)
        if not re.search(r'[AEIOUaeiou]', w):
            return '-'.join(list(w))
        return w
        
    t = re.sub(r'\b([A-Z]{2,5})\b', expand_abbr, t)
    return t


def synthesize_speech_sarvam(
    text: str,
    language_code: str = "hi",
    speaker: Optional[str] = None,
    pace: float = 1.0,
) -> Dict[str, Any]:
    """
    Synthesizes Indic text into base64 WAV audio using Sarvam AI Bulbul v2.
    """
    clean_text = text.strip()
    if not clean_text:
        return {"audio_base64": None, "format": "audio/wav", "sarvam": False, "error": "Empty text"}

    # Strip English acronym expansions inside parentheses when speaking in Hindi / Tamil
    lang_norm = (language_code or "hi").lower().strip()
    if lang_norm.startswith("hi") or lang_norm == "hi-in":
        clean_text = re.sub(r'\([a-zA-Z\s\-–—,.]+\)', '', clean_text)
    elif lang_norm.startswith("ta") or lang_norm == "ta-in":
        clean_text = re.sub(r'\([a-zA-Z\s\-–—,.]+\)', '', clean_text)

    # Normalize whitespace & remove markdown
    clean_text = re.sub(r'[#*`_~\[\]\(\)>]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Apply phonetic acronym normalization for crisp English/technical word pronunciation
    clean_text = normalize_phonetic_acronyms(clean_text)

    sarvam_lang = get_sarvam_language_code(language_code)
    if sarvam_lang == "unknown":
        if re.search(r'[\u0900-\u097F]', clean_text):
            sarvam_lang = "hi-IN"
        elif re.search(r'[\u0B80-\u0BFF]', clean_text):
            sarvam_lang = "ta-IN"
        else:
            sarvam_lang = "en-IN"

    api_key = (config.SARVAM_API_KEY or "").strip()
    if not api_key:
        logger.info("No SARVAM_API_KEY configured for TTS. Client will use browser Web TTS.")
        return {
            "audio_base64": None,
            "format": "audio/wav",
            "sarvam": False,
            "language": sarvam_lang,
            "message": "SARVAM_API_KEY not configured, fallback to client synthesis"
        }

    chosen_speaker = speaker or config.SARVAM_TTS_DEFAULT_SPEAKER
    payload = {
        "inputs": [clean_text],
        "target_language_code": sarvam_lang,
        "speaker": chosen_speaker,
        "pace": pace,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,
        "model": config.SARVAM_TTS_MODEL,
    }

    # Add pitch & loudness only for legacy v2 model
    if config.SARVAM_TTS_MODEL == "bulbul:v2":
        payload["pitch"] = 0
        payload["loudness"] = 1.5

    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=headers,
            json=payload,
            timeout=config.SARVAM_TTS_TIMEOUT_SECONDS,
        )
        if res.status_code == 200:
            data = res.json()
            audios = data.get("audios", [])
            if audios and len(audios) > 0:
                return {
                    "audio_base64": audios[0],
                    "format": "audio/wav",
                    "sarvam": True,
                    "language": sarvam_lang,
                    "speaker": chosen_speaker,
                }
            else:
                logger.warning("Sarvam TTS returned empty audios array.")
        else:
            logger.warning(f"Sarvam TTS API failed with status {res.status_code}: {res.text}")
    except Exception as e:
        logger.error(f"Error calling Sarvam TTS API: {e}")

    return {
        "audio_base64": None,
        "format": "audio/wav",
        "sarvam": False,
        "language": sarvam_lang,
        "error": "Sarvam TTS call failed, fallback to client synthesis"
    }


# Convenience alias for API endpoints
synthesize_speech = synthesize_speech_sarvam
