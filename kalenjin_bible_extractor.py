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
    def __init__(self, pdf_path: str = "kln_bible.pdf", output_dir: str = "."):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def is_book_header(self, text: str, page_text: str) -> bool:
        """Check if the given text is a book header.
        Book headers appear centered on a blue background."""
        # Book headers are typically centered and on their own line
        lines = page_text.split('\n')
        for i, line in enumerate(lines):
            if text.strip() == line.strip():
                # Check if next line contains chapter 1 or book name with 1
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if next_line == "1" or re.match(rf"{text.strip()}\s+1", next_line):
                    return True
        return False

    def is_chapter_start(self, line: str, next_line: str = "") -> bool:
        """Check if the line indicates the start of a chapter."""
        # Chapter starts with just a number and might be followed by a description line
        return bool(re.match(r'^\s*\d+\s*$', line.strip()) and 
                   (not next_line.strip() or not re.match(r'^\d+', next_line.strip())))

    def process_book(self, book: str, start_page: int, end_page: int):
        """Process a single book and save its verses."""
        verses = []
        current_chapter = None
        current_verse = None
        verse_text = []
        
        logging.info(f"Processing {book} from page {start_page} to {end_page}")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num in tqdm(range(start_page - 1, end_page), desc=f"Processing {book}"):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                # Skip if page is empty
                if not text:
                    continue

                lines = text.split('\n')
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    
                    # Skip empty lines, copyright notices, and headers
                    if not line or '©' in line or book in line:
                        i += 1
                        continue

                    # Check for chapter start
                    if self.is_chapter_start(line, next_line):
                        # Save any pending verse
                        if current_chapter and current_verse and verse_text:
                            verses.append({
                                "id": f"{book}#{current_chapter}#{current_verse}",
                                "book_kln": book,
                                "book_en": BOOK_MAPPINGS.get(book, ""),
                                "chapter": current_chapter,
                                "verse": current_verse,
                                "text_en": "",
                                "text_kln": " ".join(verse_text).strip()
                            })
                            verse_text = []
                        
                        current_chapter = int(line)
                        current_verse = None
                        i += 1
                        # Skip description line if present
                        if next_line and not re.match(r'^\d+', next_line):
                            i += 1
                        continue

                    # Try to match verse pattern (number at start of line)
                    verse_match = re.match(r'^(\d+)(.+)$', line)
                    if verse_match and current_chapter:
                        # Save any pending verse
                        if current_verse and verse_text:
                            verses.append({
                                "id": f"{book}#{current_chapter}#{current_verse}",
                                "book_kln": book,
                                "book_en": BOOK_MAPPINGS.get(book, ""),
                                "chapter": current_chapter,
                                "verse": current_verse,
                                "text_en": "",
                                "text_kln": " ".join(verse_text).strip()
                            })
                            verse_text = []
                        
                        current_verse = int(verse_match.group(1))
                        verse_text = [verse_match.group(2).strip()]
                    elif current_verse and not re.match(r'^\d+\s*[A-Za-z]', line):
                        # Line is continuation of previous verse if it doesn't start with a new verse number
                        verse_text.append(line)
                    
                    i += 1

        # Save last verse if pending
        if current_chapter and current_verse and verse_text:
            verses.append({
                "id": f"{book}#{current_chapter}#{current_verse}",
                "book_kln": book,
                "book_en": BOOK_MAPPINGS.get(book, ""),
                "chapter": current_chapter,
                "verse": current_verse,
                "text_en": "",
                "text_kln": " ".join(verse_text).strip()
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
        for book, (start_page, end_page) in BOOK_PAGE_RANGES.items():
            # Skip page 1 and any pages beyond our defined ranges
            if start_page > 1:
                self.process_book(book, start_page, end_page)

if __name__ == "__main__":
    extractor = KalenjinBibleExtractor("kln_bible.pdf")
    extractor.process_all_books() 
