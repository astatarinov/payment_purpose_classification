def preprocess_text(text: str) -> str:
    text = text.replace("ндс не облагается", "").strip()
    return text
