import torch
import warnings
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

def translate(text, tgt_lang="ja"):
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    # Detect the source language
    src_lang = detect(text)
    print(f"Detected source language: {src_lang}")

    # Map langdetect codes to Helsinki-NLP model codes
    lang_map = {
        'en': 'en',
        'ja': 'jap'
    }

    if src_lang not in lang_map or tgt_lang not in lang_map:
        raise ValueError(f"Translation from {src_lang} to {tgt_lang} is not supported.")

    # Load the pre-trained model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{lang_map[src_lang]}-{lang_map[tgt_lang]}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text=[text], return_tensors="pt", padding=True, truncation=True)

    # Generate translation
    translated = model.generate(**inputs)

    # Decode the translation
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    # Example usage
    src_text = "English to Japanese testing"
    translated_text = translate(src_text, tgt_lang="ja")
    print(f"Translated text: {translated_text}")