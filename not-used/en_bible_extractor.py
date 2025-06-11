import json
import os
import re
import logging
from typing import List, Dict
from tqdm import tqdm
import pdfplumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# NOT working as expected
class EnBibleExtractor:
    def __init__(self, pdf_path: str = "en_bible.pdf", output_dir: str = "extracts"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Book ranges (Genesis and Exodus)
        self.book_ranges = {
            "Genesis": (1, 46),
            "Exodus": (47, 86)
        }

        # Book headers to detect
        self.book_headers = {
            "Genesis": ["The First Book of Moses Called", "GENESIS", "Genesis"],
            "Exodus": ["The Second Book of Moses Called", "EXODUS", "Exodus"]
        }

    def clean_text(self, text: str) -> str:
        """Clean verse text by removing references and formatting."""
        # Remove cross-references in brackets
        text = re.sub(r'\[.*?\]', '', text)
        # Remove superscript letters and references
        text = re.sub(r'[a-z](?=\s|$|\d)', '', text)
        # Remove parenthetical references
        text = re.sub(r'\([^)]*\)', '', text)
        # Remove verse references
        text = re.sub(r'[A-Z][a-z]+\s+\d+:\d+', '', text)
        # Remove page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text)
        # Remove table of contents entries
        text = re.sub(r'.*\.\s*\.\s*\.\s*\d+\s*$', '', text)
        # Remove scattered numbers
        text = re.sub(r'\s+\d+\s+', ' ', text)
        # Remove reference patterns
        text = re.sub(r'[a-z]\s*[A-Z][a-z]+\.', '', text)
        text = re.sub(r'\s+[A-Z][a-z]+\.\s*\d+:\d+', '', text)
        # Clean up whitespace
        text = ' '.join(text.split())
        # Remove any remaining reference patterns
        text = re.sub(r'\s*[a-z]\s*(?=[A-Z]|\d|$)', '', text)
        text = re.sub(r'\s*\d+:\d+\s*', '', text)
        # Remove any remaining scattered letters
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
        # Clean up quotes and punctuation
        text = re.sub(r'"([^"]*)"', r'\1', text)
        text = re.sub(r'([.,;?!])\s*([.,;?!])', r'\1', text)
        # Final whitespace cleanup
        text = ' '.join(text.split())
        return text.strip()

    def is_chapter_header(self, line: str) -> bool:
        """Check if line is a chapter header."""
        return bool(re.match(r'^\s*(?:CHAPTER|Chapter)\s+\d+\s*$', line, re.IGNORECASE))

    def get_chapter_number(self, line: str) -> int:
        """Extract chapter number from header line."""
        match = re.search(r'(?:CHAPTER|Chapter)\s+(\d+)', line, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def extract_verses(self, text: str) -> List[Dict]:
        """Extract verses from text."""
        verses = []
        current_verse = None
        verse_text = []
        current_chapter = None
        
        # Split text into lines and process each line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for chapter header
            if self.is_chapter_header(line):
                # Save previous verse if exists
                if current_verse and verse_text:
                    clean_text = self.clean_text(' '.join(verse_text))
                    if clean_text and len(clean_text) > 10:  # Minimum length check
                        verses.append({
                            'chapter': current_chapter,
                            'verse': current_verse,
                            'text': clean_text
                        })
                # Update chapter
                current_chapter = self.get_chapter_number(line)
                current_verse = None
                verse_text = []
                continue

            # Skip book headers
            if any(header in line for header in self.book_headers.get(self.current_book, [])):
                continue

            # Try to match verse number at start of line
            verse_match = re.match(r'^\s*(\d+)\s+(.+)$', line)
            if verse_match and current_chapter:
                verse_num = int(verse_match.group(1))
                # Only accept reasonable verse numbers (1-200)
                if 1 <= verse_num <= 200:
                    # Save previous verse if exists
                    if current_verse and verse_text:
                        clean_text = self.clean_text(' '.join(verse_text))
                        if clean_text and len(clean_text) > 10:  # Minimum length check
                            verses.append({
                                'chapter': current_chapter,
                                'verse': current_verse,
                                'text': clean_text
                            })

                    # Start new verse
                    current_verse = verse_num
                    verse_text = [verse_match.group(2)]
            elif current_verse:
                # Continue previous verse if not a new verse or chapter header
                # Skip lines that look like references
                if not re.match(r'^[a-z]\s*[A-Z][a-z]+\.', line) and not re.match(r'^\d+:\d+', line):
                    verse_text.append(line)

        # Don't forget the last verse
        if current_verse and verse_text and current_chapter:
            clean_text = self.clean_text(' '.join(verse_text))
            if clean_text and len(clean_text) > 10:  # Minimum length check
                verses.append({
                    'chapter': current_chapter,
                    'verse': current_verse,
                    'text': clean_text
                })

        return verses

    def process_book(self, book_name: str, start_page: int, end_page: int):
        """Process a single book."""
        logging.info(f"Processing {book_name} (pages {start_page}-{end_page})")
        
        all_verses = []
        found_book_header = False
        self.current_book = book_name
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num in tqdm(range(start_page - 1, end_page), desc=f"Processing {book_name}"):
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if not text:
                    continue

                # Look for book header
                if not found_book_header:
                    if any(header in text for header in self.book_headers.get(book_name, [])):
                        found_book_header = True
                        logging.info(f"Found {book_name} header")
                    else:
                        continue

                # Extract verses from this page
                verses = self.extract_verses(text)
                for verse in verses:
                    verse_entry = {
                        "id": f"{book_name}#{verse['chapter']}#{verse['verse']}",
                        "book_kln": "",
                        "book_en": book_name,
                        "chapter": verse['chapter'],
                        "verse": verse['verse'],
                        "text_en": verse['text'],
                        "text_kln": ""
                    }
                    all_verses.append(verse_entry)

        # Sort verses by chapter and verse number
        all_verses.sort(key=lambda x: (x['chapter'], x['verse']))
        
        # Remove duplicates while keeping the best version of each verse
        unique_verses = {}
        for verse in all_verses:
            verse_id = verse['id']
            # Keep the longest version that's not too long (to avoid merged verses)
            if verse_id not in unique_verses or (
                len(verse['text_en']) > len(unique_verses[verse_id]['text_en']) and 
                len(verse['text_en']) < 500  # Maximum length check
            ):
                unique_verses[verse_id] = verse
        
        # Convert back to list and sort
        final_verses = list(unique_verses.values())
        final_verses.sort(key=lambda x: (x['chapter'], x['verse']))
        
        # Save to file
        if final_verses:
            output_file = os.path.join(self.output_dir, f"{book_name.lower()}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_verses, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved {len(final_verses)} verses for {book_name}")
            
            # Log chapter statistics
            chapters = sorted(list(set(v['chapter'] for v in final_verses)))
            logging.info(f"Found chapters for {book_name}: {chapters}")
            
            # Validate Genesis chapters
            if book_name == "Genesis" and len(chapters) != 50:
                logging.warning(f"Expected 50 chapters for Genesis, but found {len(chapters)}")
        else:
            logging.warning(f"No verses found for {book_name}")

    def process_all_books(self):
        """Process all configured books."""
        # Clean output directory
        for file in os.listdir(self.output_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.output_dir, file))
                logging.info(f"Removed old file: {file}")

        # Process each book
        for book_name, (start_page, end_page) in self.book_ranges.items():
            self.process_book(book_name, start_page, end_page)

if __name__ == "__main__":
    extractor = EnBibleExtractor()
    extractor.process_all_books() 