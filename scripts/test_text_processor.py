from app.services.text_processor import TextProcessor

def test_text_processor():
    processor = TextProcessor()
    text = "COVID vaccines <b>contain microchips</b> at https://fake.com"
    cleaned = processor.clean_text(text)
    keywords = processor.extract_keywords(text)
    print(f"Cleaned text: {cleaned}")
    print(f"Keywords: {keywords}")

if __name__ == "__main__":
    test_text_processor()