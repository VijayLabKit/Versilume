# 127 supported source languages for poem translation

INDIC_LANGUAGES: dict[str, str] = {
    "hi": "Hindi", "bn": "Bengali", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "or": "Odia",
    "as": "Assamese", "ur": "Urdu", "sa": "Sanskrit", "sd": "Sindhi", "ks": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mni": "Manipuri", "brx": "Bodo",
    "doi": "Dogri", "sat": "Santali",
}

WORLD_LANGUAGES: dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish", "no": "Norwegian", "da": "Danish",
    "fi": "Finnish", "is": "Icelandic", "pl": "Polish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "sr": "Serbian",
    "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian", "el": "Greek",
    "tr": "Turkish", "ru": "Russian", "uk": "Ukrainian", "be": "Belarusian", "mk": "Macedonian",
    "sq": "Albanian", "hy": "Armenian", "ka": "Georgian", "az": "Azerbaijani", "kk": "Kazakh",
    "uz": "Uzbek", "ky": "Kyrgyz", "tg": "Tajik", "tk": "Turkmen", "mn": "Mongolian",
    "ar": "Arabic", "he": "Hebrew", "fa": "Persian", "ps": "Pashto", "ku": "Kurdish",
    "am": "Amharic", "ti": "Tigrinya", "so": "Somali", "sw": "Swahili", "yo": "Yoruba",
    "ig": "Igbo", "ha": "Hausa", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
    "mg": "Malagasy", "rw": "Kinyarwanda", "sn": "Shona", "st": "Sesotho", "tn": "Tswana",
    "ny": "Chichewa", "ln": "Lingala", "wo": "Wolof", "om": "Oromo", "lg": "Luganda",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "vi": "Vietnamese", "th": "Thai",
    "my": "Burmese", "km": "Khmer", "lo": "Lao", "id": "Indonesian", "ms": "Malay",
    "tl": "Tagalog", "jv": "Javanese", "su": "Sundanese", "ceb": "Cebuano", "ne": "Nepali",
    "si": "Sinhala", "dz": "Dzongkha", "bo": "Tibetan", "dv": "Divehi", "cy": "Welsh",
    "gd": "Scottish Gaelic", "ga": "Irish", "gv": "Manx", "kw": "Cornish", "br": "Breton",
    "eu": "Basque", "ca": "Catalan", "gl": "Galician", "oc": "Occitan", "co": "Corsican",
    "sc": "Sardinian", "mt": "Maltese", "eo": "Esperanto", "la": "Latin", "yi": "Yiddish",
    "fy": "Frisian", "lb": "Luxembourgish", "fo": "Faroese", "sm": "Samoan", "to": "Tongan",
    "haw": "Hawaiian", "mi": "Maori", "fj": "Fijian", "ty": "Tahitian",
}

SUPPORTED_LANGUAGES: dict[str, str] = {**INDIC_LANGUAGES, **WORLD_LANGUAGES}


def is_supported(code: str) -> bool:
    return code.lower().strip() in SUPPORTED_LANGUAGES


def language_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code.lower().strip(), "Unknown")
