import pdfplumber
import json
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import re
import logging
from kln_book_pages_ranges import BOOK_PAGE_RANGES
from book_mappings import BOOK_MAPPINGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('kalenjin_extraction.log'),
              logging.StreamHandler()]
)

class KalenjinBibleExtractor:
    def __init__(self, pdf_path: str = "kln_bible.pdf", output_dir: str = "extracts"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def is_chapter_header(self, line: str, book_name: str) -> Optional[int]:
        """Check if line is a chapter header and return chapter number if it is."""
        # Check for patterns like "Taunet 1" or just "1"
        patterns = [
            rf"^{book_name}\s+(\d+)\s*$",  # e.g., "Taunet 1"
            r"^\s*(\d+)\s*$"  # Just a number
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def extract_verse(self, line: str) -> Optional[Tuple[int, str]]:
        """Extract verse number and text from a line."""
        # Match lines starting with a number followed by text
        match = re.match(r'^(\d+)(.+)$', line.strip())
        if match:
            try:
                verse_num = int(match.group(1))
                verse_text = match.group(2).strip()
                return verse_num, verse_text
            except ValueError:
                return None
        return None

    def process_book(self, book: str, start_page: int, end_page: int):
        """Process a single book and save its verses."""
        logging.info(f"Processing {book} from page {start_page} to {end_page}")
        
        verses = []
        current_chapter = None
        current_verse = None
        verse_text_lines = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num in tqdm(range(start_page - 1, end_page), desc=f"Processing {book}"):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if not text:
                    continue

                lines = text.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Skip empty lines
                    if not line:
                        i += 1
                        continue

                    # Check for chapter header
                    chapter_num = self.is_chapter_header(line, book)
                    if chapter_num is not None:
                        # Save previous verse if exists
                        if current_chapter and current_verse and verse_text_lines:
                            verses.append({
                                "id": f"{book}#{current_chapter}#{current_verse}",
                                "book_kln": book,
                                "book_en": BOOK_MAPPINGS.get(book, ""),
                                "chapter": current_chapter,
                                "verse": current_verse,
                                "text_en": "",
                                "text_kln": " ".join(verse_text_lines).strip()
                            })
                            verse_text_lines = []

                        current_chapter = chapter_num
                        current_verse = None
                        i += 1
                        continue

                    # Try to extract verse
                    verse_info = self.extract_verse(line)
                    if verse_info and current_chapter:
                        # Save previous verse if exists
                        if current_verse and verse_text_lines:
                            verses.append({
                                "id": f"{book}#{current_chapter}#{current_verse}",
                                "book_kln": book,
                                "book_en": BOOK_MAPPINGS.get(book, ""),
                                "chapter": current_chapter,
                                "verse": current_verse,
                                "text_en": "",
                                "text_kln": " ".join(verse_text_lines).strip()
                            })
                            verse_text_lines = []

                        verse_num, verse_text = verse_info
                        current_verse = verse_num
                        verse_text_lines = [verse_text]
                    elif current_verse:  # Continuation of previous verse
                        # Only add line if it doesn't look like a new verse
                        if not re.match(r'^\d+\s+[A-Za-z]', line):
                            verse_text_lines.append(line)

                    i += 1

        # Save last verse if pending
        if current_chapter and current_verse and verse_text_lines:
            verses.append({
                "id": f"{book}#{current_chapter}#{current_verse}",
                "book_kln": book,
                "book_en": BOOK_MAPPINGS.get(book, ""),
                "chapter": current_chapter,
                "verse": current_verse,
                "text_en": "",
                "text_kln": " ".join(verse_text_lines).strip()
            })

        if verses:
            output_file = os.path.join(self.output_dir, f"{book.lower().replace(' ', '_')}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(verses, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(verses)} verses to {output_file}")
        else:
            logging.warning(f"No verses found for {book}")

    def process_all_books(self):
        """Process all books from the PDF using predefined page ranges."""
        # First, clean up the extracts directory
        for file in os.listdir(self.output_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.output_dir, file))
                logging.info(f"Removed old file: {file}")

        # Process all books
        for book, (start_page, end_page) in BOOK_PAGE_RANGES.items():
            # Skip page 1 and any pages beyond our defined ranges
            if start_page > 1:
                self.process_book(book, start_page, end_page)
                logging.info(f"Completed processing {book}")

if __name__ == "__main__":
    extractor = KalenjinBibleExtractor()
    extractor.process_all_books() 
