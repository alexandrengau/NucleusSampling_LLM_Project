from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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


# Load the pre-trained English to French translation model and tokenizer using AutoTokenizer and AutoModelForSeq2SeqLM
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input text
input_text = ("Poor nutrition has led to a rise in the number of stranded humpback whales on the West Australian "
              "coast, veterinary researchers have said. Carly Holyoake, from Murdoch University, at the Australian "
              "Veterinary Association's annual conference in Perth on Wednesday, said an unprecedented number of "
              "mostly young whales had become stranded on the coast since 2008.")

# Generate translations using both top-p sampling and default generation (beam search) without progress bars
translation_top_p, translation_default = translate(model, tokenizer, input_text)

# Print translations
print("Input Text:\n", input_text, "\n")
print("Translation (Default Generation - Beam Search):\n", translation_default, "\n")
print("Translation (Top-p Sampling):\n", translation_top_p)
