from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
import json
import random
import numpy as np


def load_random_sentence_pairs(json_path):
    # Open the JSON file containing sentence pairs
    with open(json_path, 'r', encoding='utf-8') as json_file:
        # Load JSON data
        data = json.load(json_file)

        # Choose a random sentence pair from the list
        random_sentence_pair = random.choice(data)

        # Extract source and target sentences
        source_sentence = random_sentence_pair["source"]
        target_sentence = random_sentence_pair["target"]

        # Return the randomly selected sentence pair
        return source_sentence, target_sentence


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


def compute_perplexity(model, tokenizer, reference_text, generated_text):
    # Encode the reference (ground truth) and generated text using the tokenizer
    reference_ids = tokenizer.encode(reference_text, return_tensors='pt')
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt')

    # Disable gradient computation during inference
    with torch.no_grad():
        # Generate logits using the language model
        logits = model(
            input_ids=reference_ids,
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

# Load a random sentence pair from the JSON file
json_path = './en-fr_sentence_pairs/en-fr_sentence_pairs.json'
source_text, target_text = load_random_sentence_pairs(json_path)

# Generate translations using both top-p sampling and default generation (beam search)
translation_top_p, translation_default = translate(model, tokenizer, source_text)

# Compute reference perplexity for the model on the given dataset
reference_perplexity = compute_perplexity(model, tokenizer, target_text, target_text)
# Compute perplexities for each generated sequence
perplexity_default = compute_perplexity(model, tokenizer, target_text, translation_default)
perplexity_top_p = compute_perplexity(model, tokenizer, target_text, translation_top_p)

# Calculate absolute differences
abs_diff_default = abs(reference_perplexity - perplexity_default)
abs_diff_top_p = abs(reference_perplexity - perplexity_top_p)

# Print sentences
print("Source Text:\n", source_text)
print("Target Text:\n", target_text, "\n")

# Print translations
print("Translation (Default Generation - Beam Search):\n", translation_default)
print("Translation (Top-p Sampling):\n", translation_top_p, "\n")

# Print perplexities
print(f"Reference Perplexity for the Model: {reference_perplexity:.2f}")
print(f"Perplexity for Translation (Default Generation): {perplexity_default:.2f}")
print(f"Perplexity for Translation (Top-p Sampling): {perplexity_top_p:.2f}")

# Print absolute differences
print(f"\nAbsolute Difference (Reference vs. Default Generation): {abs_diff_default:.2f}")
print(f"Absolute Difference (Reference vs. Top-p Sampling): {abs_diff_top_p:.2f}")
