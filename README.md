# Marakwet AI - Kalenjin Language

This project processes English and Kalenjin Bible PDFs and converts them into structured JSON files, organized by book.
It will come and process also other files and folders for better analysis of Kalenjin extracts

## Features

- Extracts text from both English and Kalenjin Bible PDFs
- Aligns verses between the two languages
- Organizes verses by book into separate JSON files
- Includes progress tracking and error handling
- Maintains proper Bible book ordering

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your PDF files in the project directory:
   - `english_bible.pdf` - English Bible PDF
   - `kalenjin_bible.pdf` - Kalenjin Bible PDF

3. Run the processor:
```bash
python bible_processor.py
```

## Output

The script will create a `bible_json` directory containing separate JSON files for each book of the Bible. Each JSON file will contain an array of verse objects with the following structure:

```json
{
  "id": "Genesis#1#1",
  "book": "Genesis",
  "chapter": 1,
  "verse": 1,
  "text_en": "English verse text",
  "text_kln": "Kalenjin verse text"
}
```

## Error Handling

- The script includes progress bars to track PDF processing
- Warnings are displayed for line count mismatches between the two versions
- Invalid verse formats are automatically skipped
- The output directory is created if it doesn't exist

## Requirements

- Python 3.7+
- pdfplumber
- tqdm
- typing-extensions

## Missing Books

- Komong'u (Exodus)
- Ng'atutik (Deuteronomy)
- Tesalonikaek 2 (2 Thessalonians)

## Failed to Combine en and kln

- Failed cases when combining in English
- Acts (boisionikab_che_kiyoktoi.json) - Missing English file
- Job (job.json) - Missing both English and Kalenjin files
- Ephesians (epesoek.json) - Missing English file
