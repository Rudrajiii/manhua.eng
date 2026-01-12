import aiohttp
from urllib.parse import urlencode


class TranslateError(Exception):
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text
        super().__init__(f"Google responded with HTTP Status Code {status_code}")


async def translate_text(text: str, src: str = 'auto', dest: str = 'en'):
    """
    Translate text using Google Translate API
    
    Args:
        text: Text to translate
        src: Source language code (default: 'auto' for auto-detect)
        dest: Target language code (default: 'en' for English)
    
    Returns:
        Dictionary with translation result containing:
        - original: Original text
        - translated: Translated text
        - source_language: Detected or specified source language
        - target_language: Target language
    """
    query = {
        'dj': '1',
        'dt': ['sp', 't', 'ld', 'bd'],
        'client': 'dict-chrome-ex',
        'sl': src,
        'tl': dest,
        'q': text,
    }

    headers = {
        'User-Agent': 'Mozilla/5.0'
    }

    query_string = urlencode(query, doseq=True)
    url = f'https://clients5.google.com/translate_a/single?{query_string}'

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    text_response = await response.text()
                    raise TranslateError(response.status, text_response)

                data = await response.json()

                src_lang = data.get('src', 'auto')
                source_language = 'Detected' if src_lang == 'auto' else src_lang

                sentences = data.get('sentences', [])
                if len(sentences) == 0:
                    raise RuntimeError('Google Translate returned no information')

                original = ''.join(sentence.get('orig', '') for sentence in sentences)
                translated = ''.join(sentence.get('trans', '') for sentence in sentences)

                return {
                    'original': original,
                    'translated': translated,
                    'source_language': source_language,
                    'target_language': dest,
                }
    except Exception as e:
        print(f'Translation error: {e}')
        return {
            'original': text,
            'translated': text,
            'source_language': 'Unknown',
            'target_language': 'Unknown',
        }
