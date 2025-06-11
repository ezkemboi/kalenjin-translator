import json
import os
import re
import logging
import traceback
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import pdfplumber
from en_book_pages_ranges import BOOK_PAGE_RANGES # not working as expected and not used at the moment
from book_mappings import EN_BOOK_MAPPING

# Load environment variables
load_dotenv()

EN_BIBLE_PATH = os.getenv("EN_BIBLE_PATH")
EN_BIBLE_OUTPUT_DIR = os.getenv("EN_BIBLE_OUTPUT_DIR", "extracts")
EN_BIBLE_LOG_FILE = os.getenv("EN_BIBLE_LOG_FILE", "english_bible_extraction.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(EN_BIBLE_LOG_FILE),
        logging.StreamHandler()
    ]
)

class EnglishBibleExtractor:
    def __init__(self, pdf_path: str = EN_BIBLE_PATH, output_dir: str = EN_BIBLE_OUTPUT_DIR):
        if not pdf_path:
            raise ValueError("PDF path not provided. Please set EN_BIBLE_PATH in .env file")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Patterns for text cleaning and extraction
        self.chapter_pattern = re.compile(r'(?:CHAPTER|Chapter)\s+(\d+)', re.IGNORECASE)
        self.verse_pattern = re.compile(r'^\s*(\d+)\s+(.+)$')
        self.book_header_pattern = re.compile(r'(?:GENESIS|The First Book of Moses Called GENESIS)', re.IGNORECASE)
        self.cross_ref_pattern = re.compile(r'\[.*?\]|\d+:\d+|[a-z](?=\s*[A-Z])|[a-z]$')
        self.page_number_pattern = re.compile(r'^\s*\d+\s*$')
        self.toc_pattern = re.compile(r'.*\.\s*\.\s*\.\s*\d+\s*$')
        
        self.current_book = None
        self.current_chapter = None
        self.current_verse_text = []
        self.current_verse_num = None
        self.found_genesis_header = False
        
        logging.info(f"Initialized extractor with PDF: {pdf_path}")
        logging.info(f"Output directory: {output_dir}")

    def clean_verse_text(self, text: str) -> str:
        try:
            # Remove cross references and superscript letters
            text = self.cross_ref_pattern.sub('', text)
            
            # Remove page numbers
            if self.page_number_pattern.match(text):
                return ''
                
            # Remove table of contents entries
            if self.toc_pattern.match(text):
                return ''
                
            # Clean up whitespace
            text = ' '.join(text.split())
            
            # Remove remaining superscript letters and references
            text = re.sub(r'\s+[a-z]\s*(?=[A-Z]|\d|$)', ' ', text)
            text = re.sub(r'\s*\([^)]*\)', '', text)  # Remove parenthetical references
            
            return text.strip()
        except Exception as e:
            logging.error(f"Error cleaning text: {text}")
            logging.error(traceback.format_exc())
            return text

    def process_text_line(self, line: str, verses: List[Dict]) -> None:
        try:
            line = line.strip()
            if not line:
                return

            # Debug line content
            logging.debug(f"Processing line: {line}")

            if not self.found_genesis_header:
                if self.book_header_pattern.search(line):
                    self.found_genesis_header = True
                    logging.info(f"Found {self.current_book} header")
                return

            # Check for chapter header
            chapter_match = self.chapter_pattern.search(line)
            if chapter_match:
                self.current_chapter = int(chapter_match.group(1))
                logging.info(f"Found chapter {self.current_chapter}")
                self.current_verse_text = []
                self.current_verse_num = None
                return

            # Check for verse
            verse_match = self.verse_pattern.match(line)
            if verse_match and self.current_chapter:
                # If we have accumulated text from a previous verse, save it
                if self.current_verse_text and self.current_verse_num:
                    verse_text = self.clean_verse_text(' '.join(self.current_verse_text))
                    if verse_text and not verse_text.isdigit():
                        verse = {
                            "id": f"{self.current_book}#{self.current_chapter}#{self.current_verse_num}",
                            "book_kln": "",
                            "book_en": self.current_book,
                            "chapter": self.current_chapter,
                            "verse": self.current_verse_num,
                            "text_en": verse_text,
                            "text_kln": ""
                        }
                        verses.append(verse)
                        logging.debug(f"Added verse {verse['id']}: {verse_text}")

                # Start new verse
                self.current_verse_num = int(verse_match.group(1))
                self.current_verse_text = [verse_match.group(2)]
            elif self.current_verse_num and self.current_verse_text:
                # Continue accumulating text for the current verse
                self.current_verse_text.append(line)
        except Exception as e:
            logging.error(f"Error processing line: {line}")
            logging.error(traceback.format_exc())

    def process_book(self, book: str, start_page: int, end_page: int) -> None:
        try:
            logging.info(f"Processing {book} from page {start_page} to {end_page}")
            
            self.current_book = book
            self.current_chapter = None
            self.current_verse_text = []
            self.current_verse_num = None
            self.found_genesis_header = False if book == "Genesis" else True
            verses = []
            
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num in tqdm(range(start_page - 1, end_page), desc=f"Processing {book}"):
                    try:
                        page = pdf.pages[page_num]
                        text = page.extract_text()
                        if not text:
                            logging.warning(f"Empty text on page {page_num + 1}")
                            continue

                        logging.debug(f"Page {page_num + 1} content:\n{text}")

                        # Process each line
                        for line in text.split('\n'):
                            self.process_text_line(line, verses)

                        # If we have accumulated text at the end of the page, save it
                        if self.current_verse_text and self.current_verse_num:
                            verse_text = self.clean_verse_text(' '.join(self.current_verse_text))
                            if verse_text and not verse_text.isdigit():
                                verse = {
                                    "id": f"{self.current_book}#{self.current_chapter}#{self.current_verse_num}",
                                    "book_kln": "",
                                    "book_en": self.current_book,
                                    "chapter": self.current_chapter,
                                    "verse": self.current_verse_num,
                                    "text_en": verse_text,
                                    "text_kln": ""
                                }
                                verses.append(verse)
                                logging.debug(f"Added verse {verse['id']} at end of page: {verse_text}")
                            self.current_verse_text = []
                            self.current_verse_num = None
                            
                    except Exception as e:
                        logging.error(f"Error processing page {page_num + 1}")
                        logging.error(traceback.format_exc())

            # Validate chapters for Genesis
            if book == "Genesis":
                chapters = sorted(list(set(v["chapter"] for v in verses)))
                if len(chapters) != 50:
                    logging.warning(f"Expected 50 chapters for Genesis, but found {len(chapters)}: {chapters}")

            if verses:
                # Sort verses by chapter and verse number
                verses.sort(key=lambda x: (x["chapter"], x["verse"]))
                
                output_file = os.path.join(self.output_dir, f"{book.lower()}.json")
                with open(output_file, 'w') as f:
                    json.dump(verses, f, indent=2)
                logging.info(f"Saved {len(verses)} verses to {output_file}")
            else:
                logging.warning(f"No verses found for {book}")
        except Exception as e:
            logging.error(f"Error processing book {book}")
            logging.error(traceback.format_exc())

    def process_all_books(self):
        """Process all books from the PDF using predefined page ranges."""
        try:
            # First, clean up the extracts directory
            for file in os.listdir(self.output_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.output_dir, file))
                    logging.info(f"Removed old file: {file}")

            # Process all books
            processed_count = 0
            for book, (start_page, end_page) in BOOK_PAGE_RANGES.items():
                self.process_book(book, start_page, end_page)
                processed_count += 1
                logging.info(f"Completed processing {book} ({processed_count} books processed so far)")
        except Exception as e:
            logging.error("Error in process_all_books")
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    try:
        extractor = EnglishBibleExtractor()
        extractor.process_all_books()
    except Exception as e:
        logging.error("Fatal error in main")
        logging.error(traceback.format_exc()) 