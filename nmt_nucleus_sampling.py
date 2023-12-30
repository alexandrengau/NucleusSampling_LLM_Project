from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
import json
import random
import numpy as np


def load_random_paragraph(json_path):
    # Open the JSON file containing paragraphs
    with open(json_path, 'r', encoding='utf-8') as json_file:
        # Load JSON data
        data = json.load(json_file)

        # Extract paragraphs from the loaded data
        paragraphs = data['paragraphs']

        # Choose a random paragraph from the list
        random_paragraph = random.choice(paragraphs)

        # Return the randomly selected paragraph
        return random_paragraph


def translate(model, tokenizer, input_text, max_length=512, top_p=0.5):
    # Encode the input text using the tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate translation with top-p sampling
    translation_top_p = model.generate(
        input_ids,
        max_length=max_length,
        top_p=top_p,
        top_k=0,
        num_beams=1,  # No beam search
        do_sample=True,
        decoder_start_token_id=model.config.pad_token_id,
    )

    # Generate translation using default generation (beam search)
    translation_default = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        decoder_start_token_id=model.config.pad_token_id,
    )

    # Decode the generated sequences and remove special tokens
    decoded_translation_top_p = tokenizer.decode(translation_top_p[0], skip_special_tokens=True)
    decoded_translation_default = tokenizer.decode(translation_default[0], skip_special_tokens=True)

    # Return the decoded translations
    return decoded_translation_top_p, decoded_translation_default


def compute_perplexity(model, tokenizer, input_text, generated_text):
    # Encode the input and generated text using the tokenizer
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt')

    # Disable gradient computation during inference
    with torch.no_grad():
        # Generate logits using the language model
        logits = model(
            input_ids=input_ids,
            decoder_input_ids=generated_ids
        ).logits

    # Flatten the logits and generated_ids for calculating cross-entropy
    flat_logits = logits.view(-1, logits.size(-1))

    # Calculate negative log-likelihoods (nlls) for each token
    nlls = F.cross_entropy(flat_logits, generated_ids[0].view(-1), reduction='none').tolist()

    # Calculate perplexity as the exponential of the mean of nlls
    ppl = np.exp(np.mean(nlls))

    return ppl


# Load the pre-trained English to French translation model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a random paragraph from the JSON file
json_path = './book_metadata/parsed_book.json'
input_text = load_random_paragraph(json_path)

# Generate translations using both top-p sampling and default generation (beam search) without progress bars
translation_top_p, translation_default = translate(model, tokenizer, input_text)

# Compute perplexities for each generated sequence
perplexity_default = compute_perplexity(model, tokenizer, input_text, translation_default)
perplexity_top_p = compute_perplexity(model, tokenizer, input_text, translation_top_p)

"""
When you compute compute_perplexity(model, tokenizer, input_text, generated_text) for a human-written input_text 
and its NMT-generated translation (generated_text), you are essentially assessing how well the language model (NMT 
model) predicts the generated translation given the original text. Perplexity measures how well the model's predicted 
distribution aligns with the actual tokens in the generated sequence.
"""

# Print translations
print("Input Text:\n", input_text, "\n")
print("Translation (Default Generation - Beam Search):\n", translation_default, "\n")
print("Translation (Top-p Sampling):\n", translation_top_p, "\n")

# Print and compare perplexities
print(f"Perplexity for Translation (Default Generation): {perplexity_default:.2f}")
print(f"Perplexity for Translation (Top-p Sampling): {perplexity_top_p:.2f}")


"""
--- GREAT EXAMPLE BELOW ---

Input Text:
 Mrs. Bradshaw's 'Well! Well!' seemed to sum up the general feeling; Mr. Kemp, shaking his head, eyed him with gentle reproach. 

Translation (Default Generation - Beam Search (b=4)):
 Mme Bradshaw 'Eh bien! Eh bien!' semblait résumer le sentiment général; M. Kemp, secouant la tête, le regardait avec un doux reproche. 

Translation (Top-p Sampling (p=0.5)):
 L'expression «Eh bien! Eh bien!» de Mme Bradshaw semblait résumer la sensation générale ; M. Kemp, agitant la tête, le regardait avec une douce opprobre. 

Perplexity for Translation (Default Generation): 1197.26
Perplexity for Translation (Top-p Sampling): 2373.95
"""
