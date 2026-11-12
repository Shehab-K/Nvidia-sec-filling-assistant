"""
Purpose: Text cleaning utility for Text-to-Speech (TTS) preprocessing.
Cleans markdown, URLs, emails, emojis, and whitespace for English responses.
"""
import re


def remove_image_markdown(text: str) -> str:
    """Remove markdown image syntax: ![alt text](url)"""
    return re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)


def replace_email_and_urls(text: str) -> str:
    """Remove email addresses and web URLs from TTS input."""
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://[^\s]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'www\.[^\s]+', '', text, flags=re.IGNORECASE)
    return text


def clean_from_emojis(text: str) -> str:
    """Remove emojis and special symbols from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\ufe0f"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text).strip()


def clean_whitespace(text: str) -> str:
    """Clean up extra whitespace, newlines, and punctuation spacing."""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +([.,!?])', r'\1', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line).strip()


def clean_text_for_tts(text: str) -> str:
    """
    Main function to clean text for TTS output.
    """
    if not text:
        return text

    text = remove_image_markdown(text)
    text = replace_email_and_urls(text)
    text = clean_from_emojis(text)
    text = clean_whitespace(text)

    return text