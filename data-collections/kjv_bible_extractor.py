import json
import os
from dotenv import load_dotenv
import logging
import requests
from typing import List, Dict
from tqdm import tqdm


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class KJVBibleExtractor:
    def __init__(self, output_dir: str = os.getenv("EN_BIBLE_OUTPUT_DIR")):
        self.url = os.getenv("EN_BIBLE_URL")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Book mappings with English and Kalenjin names
        self.book_mappings = {
            "Genesis": {"abbrev": "gn", "kln": "Taunet"},
            "Exodus": {"abbrev": "ex", "kln": "Komong'u"},
            "Leviticus": {"abbrev": "lv", "kln": "Atebetab Lawik"},
            "Numbers": {"abbrev": "nm", "kln": "Koitosiek"},
            "Deuteronomy": {"abbrev": "dt", "kln": "Ng'atutik"},
            "Joshua": {"abbrev": "js", "kln": "Josua"},
            "Judges": {"abbrev": "jud", "kln": "Kiruogik"},
            "Ruth": {"abbrev": "rt", "kln": "Ruth"},
            "1 Samuel": {"abbrev": "1sm", "kln": "Samuel 1"},
            "2 Samuel": {"abbrev": "2sm", "kln": "Samuel 2"},
            "1 Kings": {"abbrev": "1kgs", "kln": "Laitorinik 1"},
            "2 Kings": {"abbrev": "2kgs", "kln": "Laitorinik 2"},
            "1 Chronicles": {"abbrev": "1ch", "kln": "Ateptab Betusiek 1"},
            "2 Chronicles": {"abbrev": "2ch", "kln": "Ateptab Betusiek 2"},
            "Ezra": {"abbrev": "ezr", "kln": "Esra"},
            "Nehemiah": {"abbrev": "ne", "kln": "Neemia"},
            "Esther": {"abbrev": "et", "kln": "Esta"},
            "Job": {"abbrev": "jb", "kln": "Job"},
            "Psalms": {"abbrev": "ps", "kln": "Tienwogikal"},
            "Proverbs": {"abbrev": "prv", "kln": "Kalewenaik"},
            "Ecclesiastes": {"abbrev": "ec", "kln": "Amdoindet"},
            "Song of Solomon": {"abbrev": "so", "kln": "Tiendo Ne Siirei Tugul"},
            "Isaiah": {"abbrev": "is", "kln": "Isaia"},
            "Jeremiah": {"abbrev": "jr", "kln": "Jeremia"},
            "Lamentations": {"abbrev": "lm", "kln": "Rirekab Jeremia"},
            "Ezekiel": {"abbrev": "ez", "kln": "Esekiel"},
            "Daniel": {"abbrev": "dn", "kln": "Daniel"},
            "Hosea": {"abbrev": "ho", "kln": "Hosea"},
            "Joel": {"abbrev": "jl", "kln": "Joel"},
            "Amos": {"abbrev": "am", "kln": "Amos"},
            "Obadiah": {"abbrev": "ob", "kln": "Obadia"},
            "Jonah": {"abbrev": "jn", "kln": "Jona"},
            "Micah": {"abbrev": "mi", "kln": "Mika"},
            "Nahum": {"abbrev": "na", "kln": "Nahum"},
            "Habakkuk": {"abbrev": "hk", "kln": "Habakuk"},
            "Zephaniah": {"abbrev": "zp", "kln": "Sepania"},
            "Haggai": {"abbrev": "hg", "kln": "Hagai"},
            "Zechariah": {"abbrev": "zc", "kln": "Sekaria"},
            "Malachi": {"abbrev": "ml", "kln": "Malaki"},
            "Matthew": {"abbrev": "mt", "kln": "Matayo"},
            # New Testament
            "Mathew": {"abbrev": "mt", "kln": "Matayo"},
            "Mark": {"abbrev": "mk", "kln": "Mariko"},
            "Luke": {"abbrev": "lk", "kln": "Luka"},
            "John": {"abbrev": "jo", "kln": "Johana"},
            "Acts": {"abbrev": "ac", "kln": "Boisionikab Che Kiyoktoi"},
            "Romans": {"abbrev": "rm", "kln": "Romanik"},
            "1 Corinthians": {"abbrev": "1co", "kln": "Korintoek 1"},
            "2 Corinthians": {"abbrev": "2co", "kln": "Korintoek 2"},
            "Galatians": {"abbrev": "gl", "kln": "Galatiaek"},
            "Ephesians": {"abbrev": "ep", "kln": "Epesoek"},
            "Philippians": {"abbrev": "ph", "kln": "Pilipiik"},
            "Colossians": {"abbrev": "cl", "kln": "Kolosiek"},
            "1 Thessalonians": {"abbrev": "1ts", "kln": "Tesalonikek 1"},
            "2 Thessalonians": {"abbrev": "2ts", "kln": "Tesalonikek 2"},
            "1 Timothy": {"abbrev": "1tm", "kln": "Timoteo 1"},
            "2 Timothy": {"abbrev": "2tm", "kln": "Timoteo 2"},
            "Titus": {"abbrev": "tt", "kln": "Tito"},
            "Philemon": {"abbrev": "phm", "kln": "Pilemon"},
            "Hebrews": {"abbrev": "hb", "kln": "Hebranik"},
            "James": {"abbrev": "jm", "kln": "Jakobo"},
            "1 Peter": {"abbrev": "1pe", "kln": "Petero 1"},
            "2 Peter": {"abbrev": "2pe", "kln": "Petero 2"},
            "1 John": {"abbrev": "1jo", "kln": "Johana 1"},
            "2 John": {"abbrev": "2jo", "kln": "Johana 2"},
            "3 John": {"abbrev": "3jo", "kln": "Johana 3"},
            "Jude": {"abbrev": "jd", "kln": "Juda"},
            "Revelation": {"abbrev": "re", "kln": "Ng’and’utiet"}
        }

    # The source JSON uses Portuguese abbreviations for some books
    ALT_ABBREVS = {
        "jud": "jz", "1kgs": "1rs", "2kgs": "2rs", "1ch": "1cr", "2ch": "2cr",
        "ezr": "ed", "jb": "jó", "ps": "sl", "prv": "pv", "so": "ct",
        "ho": "os", "mi": "mq", "hk": "hc", "zp": "sf", "hg": "ag",
        "mk": "mc", "lk": "lc", "ac": "atos", "ep": "ef", "ph": "fp",
        "phm": "fm", "jm": "tg", "re": "ap",
    }

    def fetch_bible_data(self) -> List[Dict]:
        """Fetch Bible data from the URL."""
        logging.info(f"Fetching Bible data from {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            # Handle UTF-8 BOM
            text = response.text.encode().decode('utf-8-sig')
            return json.loads(text)
        except requests.RequestException as e:
            logging.error(f"Failed to fetch Bible data: {e}")
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Bible data: {e}")
            return []

    def process_book(self, book_data: Dict, book_name: str) -> List[Dict]:
        """Process a single book's verses."""
        verses = []
        
        # Get chapters
        chapters = book_data.get("chapters", [])
        logging.info(f"Processing {book_name} ({len(chapters)} chapters)")
        
        # Process each chapter
        for chapter_idx, chapter in enumerate(chapters, 1):
            # Process each verse
            for verse_idx, verse_text in enumerate(chapter, 1):
                # Clean up verse text
                verse_text = verse_text.strip()
                
                # Remove curly braces annotations
                while "{" in verse_text and "}" in verse_text:
                    start = verse_text.find("{")
                    end = verse_text.find("}") + 1
                    verse_text = verse_text[:start] + verse_text[end:]
                
                verse_entry = {
                    "id": f"{book_name}#{chapter_idx}#{verse_idx}",
                    "book_kln": self.book_mappings[book_name]["kln"],
                    "book_en": book_name,
                    "chapter": chapter_idx,
                    "verse": verse_idx,
                    "text_en": verse_text.strip(),
                    "text_kln": ""
                }
                verses.append(verse_entry)
        
        return verses

    def save_verses(self, verses: List[Dict], book_name: str):
        """Save verses to a JSON file."""
        if verses:
            output_file = os.path.join(self.output_dir, f"{book_name.lower()}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(verses, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(verses)} verses for {book_name}")
            
            # Log chapter statistics
            chapters = sorted(list(set(v['chapter'] for v in verses)))
            logging.info(f"Found chapters for {book_name}: {chapters}")
        else:
            logging.warning(f"No verses found for {book_name}")

    def process_all_books(self):
        """Process all configured books."""
        # Clean output directory
        for file in os.listdir(self.output_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.output_dir, file))
                logging.info(f"Removed old file: {file}")

        # Fetch Bible data
        bible_data = self.fetch_bible_data()
        if not bible_data:
            logging.error("No Bible data found")
            return

        # Find and process target books
        for book_name, book_info in self.book_mappings.items():
            # Find the book in the data
            book_data = None
            for book in bible_data:
                wanted = {book_info["abbrev"], self.ALT_ABBREVS.get(book_info["abbrev"], "")}
                if book.get("abbrev", "").lower() in wanted:
                    book_data = book
                    break

            if book_data:
                # Process and save verses
                verses = self.process_book(book_data, book_name)
                self.save_verses(verses, book_name)
            else:
                logging.warning(f"Book {book_name} not found in the data")

if __name__ == "__main__":
    extractor = KJVBibleExtractor()
    extractor.process_all_books() 