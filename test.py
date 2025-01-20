import re

def normalize_repeated_characters(text):
    """Normalize words with excessive character repetition while preserving natural double letters."""
    return re.sub(r'(.)\1{2,}', r'\1', text)  # Reduces sequences of 3+ to just 1

# Testing the function
words = ["loooooove", "suuuper", "heyyyyy", "really", "successfully", "feed", "food"]
normalized_words = [normalize_repeated_characters(word) for word in words]
print(normalized_words)