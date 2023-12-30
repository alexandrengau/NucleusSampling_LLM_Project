# https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/text_generation#transformers.GenerationConfig

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F


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
        decoder_start_token_id=model.config.pad_token_id,
    )

    # Generate translation using default generation (beam search)
    translation_default = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        decoder_start_token_id=model.config.pad_token_id,
    )

    return (
        tokenizer.decode(translation_top_p[0], skip_special_tokens=True),
        tokenizer.decode(translation_default[0], skip_special_tokens=True)
    )


def compute_perplexity(model, tokenizer, input_text, generated_text):
    # Encode the input and generated text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    generated_ids = tokenizer.encode(generated_text, return_tensors='pt')

    # Ensure that decoder_input_ids or decoder_inputs_embeds is specified
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            decoder_input_ids=generated_ids
        ).logits

    # Flatten the logits and generated_ids for calculating cross-entropy
    flat_logits = logits.view(-1, logits.size(-1))

    # Compute perplexity
    perplexity = torch.exp(F.cross_entropy(flat_logits, generated_ids[0].view(-1)))

    return perplexity.item()


# Load the pre-trained English to French translation model and tokenizer using AutoTokenizer and AutoModelForSeq2SeqLM
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input an English text to translate into French
input_text = "In the bustling city of New York, where skyscrapers touch the clouds and yellow taxis weave through "\
             "crowded streets, life is a vibrant tapestry of diversity and opportunity. From the iconic Central "\
             "Park, where locals find solace in nature's embrace, to the dazzling lights of Times Square that never "\
             "sleep, the city pulses with energy. As the sun sets over the Hudson River, casting a warm glow on the "\
             "city skyline, the aroma of diverse cuisines wafts through the air. In the heart of Manhattan, "\
             "Broadway theaters showcase the finest talents in captivating performances, while museums like the "\
             "Metropolitan Museum of Art house treasures from centuries past."

# Generate translations using both top-p sampling and default generation (beam search) without progress bars
translation_top_p, translation_default = translate(model, tokenizer, input_text)

# Compute perplexities for each generated sequence
perplexity_default = compute_perplexity(model, tokenizer, input_text, translation_default)
perplexity_top_p = compute_perplexity(model, tokenizer, input_text, translation_top_p)

# Print translations
print("Input Text:\n", input_text, "\n")
print("Translation (Default Generation - Beam Search):\n", translation_default, "\n")
print("Translation (Top-p Sampling):\n", translation_top_p, "\n")

# Print and compare perplexities
print(f"Perplexity for Translation (Default Generation): {perplexity_default:.2f}")
print(f"Perplexity for Translation (Top-p Sampling): {perplexity_top_p:.2f}")