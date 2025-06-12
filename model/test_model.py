from transformers import MarianTokenizer, MarianMTModel

model_dir = "kalenjin-translator"
tokenizer = MarianTokenizer.from_pretrained(model_dir)
model = MarianMTModel.from_pretrained(model_dir)

def translate_kln_to_en(text):
    print('Original Text:', text)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print("Tokenized input IDs:", inputs['input_ids'])

    translated = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,                # Try beam search for better output
        early_stopping=True,
        repetition_penalty=2.0,     # Helps reduce repeated words
        no_repeat_ngram_size=2,     # Prevents repeating bigrams
    )

    print("Generated token IDs:", translated[0])
    decoded_output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return decoded_output

# Interactive input
user_input = input("Enter a Kalenjin word or sentence to translate to English: ")
translation = translate_kln_to_en(user_input)
print('Finalised Translation:', translation)
