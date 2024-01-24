# THE CURIOUS CASE OF NEURAL TEXT *De*GENERATION - Paper-based LLM Project

## IASD Master Program 2023/2024 - PSL Research University

### About this project

This project is the final homework assignment for the Large Language Model class of the IASD (Artificial Intelligence, Systems and Data) Master Program 2023/2024, at PSL Research University (Université PSL).

*The project achieved the following objectives:*
- Implemented and evaluated the nucleus sampling method for open-ended text generation, replicating the approach outlined in the "The Curious Case of Neural Text *De*Generation" article by Holtzman et al. This included assessing nucleus sampling in comparison to decoding methods such as beam search, top-k sampling, and greedy search.
- Explored the extension of nucleus sampling to Neural Machine Translation (NMT) for directed text generation. Adapted nucleus sampling to enhance the generation of contextually relevant translated sentences using the Helsinki-NLP/opus-mt-en-fr model. Conducted evaluations based on perplexity and initiated a protocol for human-based assessments.

## General Information

The report can be viewed in the [report.pdf](report.pdf) file, and the intermediary presentation can be viewed in the [intermediary_presentation.pdf](intermediary_presentation.pdf). It answers to the instructions given in the [project_guidelines.pdf](project_guidelines.pdf) file provided by the professors.

The rest of the instructions can be found below. If you want to copy and recreate this project, or test it for yourself, some important information to know.

- **requirements.txt** Among the good practice of datascience, we encourage you to use conda or virtualenv to create python 
environment. To test your code on our platform, you are required to update the [requirements.txt](requirements.txt) file,
with the different libraries you might use. 

When your code will be tested, we will execute : 
  > pip install -r requirements.txt


- **data**

- **implementation.py**

- **utils.py**

- **config.json**

- **en-fr_sentence_pairs** The [en-fr_sentence_pairs](en-fr_sentence_pairs) subdirectory contains English-French sentence pairs extracted from the
[News-Commentary v16](https://opus.nlpl.eu/News-Commentary-v16.php) parallel corpus provided by WMT (Workshop on Machine
Translation), as well as some code to preprocess and create a JSON file hosting the dataset. For more information, please see 
[en-fr_sentence_pairs/README.md](en-fr_sentence_pairs/README.md).


- **nmt_nucleus_sampling.py** The [nmt_nucleus_sampling.py](nmt_nucleus_sampling.py) script utilizes the transformers library
to implement a neural machine translation model ([Helsinki-NLP/opus-mt-en-fr](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr))
for English to French translation. If the configuration is : 
  - *main(random_sentence_translation=True,
         dataset_translation=False,
         bleu_score_evaluation=False,
         top_p=0.7,
         num_beams=4,
         device=device)*, the script will select randomly a sentence in the  
  [en-fr_sentence_pairs/en-fr_sentence_pairs.json](en-fr_sentence_pairs/en-fr_sentence_pairs.json) dataset and translate it using beam 
  search and nucleus sampling (following the parameters entered), compute the perplexities of the translations, and then print the results.
  - *main(random_sentence_translation=False,
         dataset_translation=True,
         bleu_score_evaluation=False,
         top_p=0.7,
         num_beams=4,
         device=device)*, the script will translate the first 1000 sentences from the 
  [en-fr_sentence_pairs/en-fr_sentence_pairs.json](en-fr_sentence_pairs/en-fr_sentence_pairs.json) dataset using beam 
  search and nucleus sampling (following the parameters entered), and then overwrite the json file with the translations.
  - *main(random_sentence_translation=False,
         dataset_translation=False,
         bleu_score_evaluation=True,
         top_p=0.7,
         num_beams=4,
         device=device)*, the script will compute the BLEU score of the 1000 sentences translated using the previous configuration,
  and print the scores of both the beam-search-generated sentences and the nucleus-sampling-generated sentences.

To reproduce the results concerning NMT and shown in the report, re-extract the dataset following the instructions in the [en-fr_sentence_pairs/README.md](en-fr_sentence_pairs/README.md)
file, then run the three configurations listed above from top to bottom. To run the script, execute :
  > python nmt_nucleus_presentation.py

---

### Acknowledgement

This project was made possible with the guidance and support of the following :

- **Prof. Alexandre Allauzen**
  - Professor at *ESPCI, PSL*
  - Researcher in the *MILES Team* at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*

- **Florian Le Bronnec**
  - PhD candidate at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
 
This project was a group project, and was made possible thanks to the collaboration of :

- **Mathilde Kretz**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Thomas Boudras**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Alexandre Ngau**, *IASD Master Program 2023/2024 student, at PSL Research University*

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

**Note:**

This project is part of ongoing research and is subject to certain restrictions. Please refer to the license section and the [LICENSE.md](LICENSE.md) file for details on how to use this code for research purposes.
