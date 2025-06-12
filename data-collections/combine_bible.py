import json
import os
from pathlib import Path


def load_book_mappings(mapping_file):
    with open(mapping_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the dictionary content
        dict_content = content[content.find('{'):content.rfind('}')+1]
        # Replace single quotes with double quotes for valid JSON
        dict_content = dict_content.replace("'", '"')
        return json.loads(dict_content)

def get_english_filename(book_name):
    # Manual mapping for special cases
    special_cases = {
        'pilipik': 'philippians',
        'ateptab_betusiek_1_': '1 chronicles',
        'boisionikab_che_kiyoktoi': 'acts',
        'galatiek': 'galatians',
        'kolosaik': 'colossians',
        'job': 'job',
        'epesoek': 'ephesians'
    }
    
    # Check if it's a special case
    base_name = book_name.lower().replace('.json', '')
    if base_name in special_cases:
        return special_cases[base_name] + '.json'
    return None

def combine_bible_books(en_file, kln_file, output_file):
    # Skip if either file doesn't exist
    if not os.path.exists(en_file) or not os.path.exists(kln_file):
        print(f"Skipping - Missing files:\nEN: {en_file}\nKLN: {kln_file}")
        return False

    try:
        # Read English file
        with open(en_file, 'r', encoding='utf-8') as f:
            en_data = json.load(f)
        
        # Read Kalenjin file
        with open(kln_file, 'r', encoding='utf-8') as f:
            kln_data = json.load(f)
        
        # Create a dictionary to store Kalenjin verses by chapter and verse
        kln_verses = {}
        for verse in kln_data:
            key = f"{verse['chapter']}#{verse['verse']}"
            kln_verses[key] = verse['text_kln']
        
        # Update English verses with Kalenjin text
        for verse in en_data:
            key = f"{verse['chapter']}#{verse['verse']}"
            verse['text_kln'] = kln_verses.get(key, "")
        
        # Write combined data to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(en_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Error processing {en_file} and {kln_file}: {str(e)}")
        return False

def process_all_books():
    # Get list of all Kalenjin files
    kln_dir = Path('marakwet-ai/extracts')
    en_dir = Path('marakwet-ai/en_extracts')
    output_dir = Path('marakwet-ai/cleaned_extracts')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process each Kalenjin file
    successful = 0
    failed = 0
    
    for kln_file in kln_dir.glob('*.json'):
        try:
            # First try to get English filename from special cases
            special_en_filename = get_english_filename(kln_file.name)
            
            if special_en_filename:
                en_file = en_dir / special_en_filename
                output_file = output_dir / special_en_filename
            else:
                # Read Kalenjin file to get book name
                with open(kln_file, 'r', encoding='utf-8') as f:
                    kln_data = json.load(f)
                    if not kln_data:
                        continue
                    # Get English book name from the first verse
                    en_book_name = kln_data[0]['book_en'].lower()
                
                # Find corresponding English file
                en_file = en_dir / f"{en_book_name}.json"
                output_file = output_dir / f"{en_book_name}.json"
            
            if combine_bible_books(str(en_file), str(kln_file), str(output_file)):
                successful += 1
                print(f"Successfully combined {output_file.stem}")
            else:
                failed += 1
                
        except Exception as e:
            print(f"Error processing {kln_file}: {str(e)}")
            failed += 1
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully combined: {successful} books")
    print(f"Failed to combine: {failed} books")

if __name__ == "__main__":
    process_all_books() 