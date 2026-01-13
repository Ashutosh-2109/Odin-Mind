import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Keep math-related symbols + words
    text = re.sub(r"[^a-z0-9+\-*/^=().\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
