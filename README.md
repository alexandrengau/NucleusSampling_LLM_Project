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


- **config.json** The JSON file *config.json* contains the initialization of all variables parametrizing the *utils.py* and *implementation.py* files. These include the extraction paths for the data and write files, as well as the parameters for the compared techniques. For the top-k sampling set *k* to te desired value, *p* for the cumulated probability threshold of nucleus sampling, *w* for the width of the beam-search and finaly set *t* to a value between 0 and 1 if you wish to add a temperature parameter for top-k or nucleus sampling.


- **data** Into the folder *data* will automatically will be automatically downloaded the 'small' databases proposed by OpenAI for the gpt-2 model. The tokenized and filtered files will then be written there, enabling generation from prepared contexts. It will also contain the texts generated after calling the model and decoding.


- **implementation.py** The main script *implementation.py* contains the two decoding functions for the studied methods as well as the the *main()*. 
  - *main()*, the script will download the OpenAI databases if there are not locally present in the folder *data*. It will then proceed to the preprocessing (tokenization and filtering) of the file that will be used for the context creation (paths to be set in *config.json*). This chosen context will be loaded to serve as the input of the LLM model (we recommand gpt2 model) to then generate a  distribution over each word of its vocabulary for a lenght *gen_len* set in *config.json*. The decoding functions are called there.
  - *decode(model=model,
            prompt=batch,
            batch_size=config["batch_size"],
            gen_len=gen_length,
            sep=SEP,
            temp=config["t"],
            k=config["k"],
            p=config["p"],
            device=device)* , the script will decode the distribution with top-k sampling or nucleus sampling techniques, with the possible addition of a temperature parameter. Note that if you want to decode using only the top-k sampling for example, parameters *p* and *t* has to be set to *false* in the *config.json*. The greedy search is obtained by setting *k* to one.
  - *beam_search_decode(model=model,
                        prompt=batch,
                        gen_length=gen_length,
                        sep=SEP,
                        w=config["w"],
                        device=device)*, the function is called when the *w* parameter is activated and will decode the distribution with beam-search of width *w*.


- **utils.py** In the *utils.py* script, you will find all the auxiliary functions required to operate the main file *implementation.py*. These include functions such as *download_datasets()* directly taken from [OpenAI](https://github.com/openai/gpt-2-output-dataset) that download the dataset as described previously. The *encode_json()* and *filter_for_conditional()* functions transform raw files, such as those downloaded above, into files containing the tokens associated with the texts (.tokenized), as well as the texts that can be used for generation (.filtered) (i.e. containing at least one complete sentence).
You'll also find the *load_dataset(dataset_path, batch_size, device, bs=False)* function, which acts as a Loader in the main script, as well as a perplexity calculation in *perplexity()* for generated texts.


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
