from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset


def translate(model, tokenizer, input_text, max_length=200, top_p=0.5):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate translation with top-p sampling
    translation_top_p = model.generate(
        input_ids,
        max_length=max_length,
        top_p=top_p,
        top_k=0,
        num_beams=1,
        do_sample=True,
    )

    # Generate translation using default generation (beam search)
    translation_default = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
    )

    return (
        tokenizer.decode(translation_top_p[0], skip_special_tokens=True),
        tokenizer.decode(translation_default[0], skip_special_tokens=True)
    )


def compute_perplexity(model, tokenizer, generated_text, input_text, reference_text):
    # Tokenize the input, generated, and reference texts using the appropriate tokenizers
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt')
    reference_ids = tokenizer.encode(reference_text, return_tensors='pt')

    # Ensure that generated_ids are used as decoder_input_ids
    with torch.no_grad():
        logits_generated = model(input_ids, decoder_input_ids=generated_ids).logits
        logits_reference = model(input_ids, decoder_input_ids=reference_ids).logits

    # Flatten the logits and the target labels
    logits_generated = logits_generated.view(-1, logits_generated.size(-1))
    logits_reference = logits_reference.view(-1, logits_reference.size(-1))

    target_labels = reference_ids.view(-1)

    # Compute the cross-entropy loss for both generated and reference sequences
    loss_generated = F.cross_entropy(logits_generated, target_labels)
    loss_reference = F.cross_entropy(logits_reference, target_labels)

    # Compute perplexities as the exponentials of the losses
    perplexity_generated = torch.exp(loss_generated).item()
    perplexity_reference = torch.exp(loss_reference).item()

    return perplexity_generated, perplexity_reference


# Load the pre-trained English to French translation model and tokenizer using AutoTokenizer and AutoModelForSeq2SeqLM
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Specify the desired language pair
language_pair = "fr-en"

# Load the WMT14 dataset for the specified language pair without storing it locally
wmt14_dataset = load_dataset("wmt14", language_pair)

# Randomly sample an index from the available translations
sample_index = np.random.randint(len(wmt14_dataset["train"]))

# Extract English and French translations for the sampled index
input_text = wmt14_dataset["train"][sample_index]["translation"]["en"][:200]
reference_french_text = wmt14_dataset["train"][sample_index]["translation"]["fr"][:200]

# Generate translations using both top-p sampling and default generation (beam search) without progress bars
translation_top_p, translation_default = translate(model, tokenizer, input_text)

# Compute perplexities for each generated sequence
perplexity_top_p, baseline_perplexity = compute_perplexity(model, tokenizer, translation_top_p, input_text, reference_french_text)
perplexity_default, _ = compute_perplexity(model, tokenizer, translation_default, input_text, reference_french_text)

# Print translations
print("Input Text:\n", input_text, "\n")
print("Translation (Default Generation - Beam Search):\n", translation_default, "\n")
print("Translation (Top-p Sampling):\n", translation_top_p, "\n")

# Print and compare perplexities to reference perplexity
print(f"Baseline Perplexity: {baseline_perplexity:.2f}")
print(f"Perplexity for Translation (Default Generation): {perplexity_default:.2f}")
print(f"Perplexity for Translation (Top-p Sampling): {perplexity_top_p:.2f}")
