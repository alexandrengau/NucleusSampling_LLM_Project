# English-French Sentence Pairs Subdirectory

## Overview

This subdirectory contains English-French sentence pairs extracted from the [News-Commentary v16](https://opus.nlpl.eu/News-Commentary-v16.php) parallel corpus provided by WMT (Workshop on Machine Translation). The data includes parallel text in 15 languages, and we have specifically focused on English-French language pairs.

### Contents

- **[en-fr.tmx](en-fr.tmx)**: Translation Memory eXchange (TMX) file containing parallel sentence pairs in English and French.
- **[en-fr_sentence_pairs.json](en-fr_sentence_pairs.json)**: JSON file containing the sentence pairs extracted from en-fr.tmx.
- **[extract_tmx.py](extract_tmx.py)**: Python script for extracting sentence pairs from en-fr.tmx and creating en-fr_sentence_pairs.json.

## Data Source

The parallel corpus is sourced from News-Commentary v16, a collection of News Commentaries provided by WMT for training Statistical Machine Translation (SMT) models. The source is taken from WMT 19. If you use this corpus in your work, please cite the following article:

**Title:** Parallel Data, Tools and Interfaces in OPUS  
**Author:** J. Tiedemann  
**Year:** 2012  
**Conference:** Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)

## Usage - Extract Sentence Pairs

To extract sentence pairs from the TMX file, use the provided Python script:

```bash
python ./en-fr_sentence_pairs/extract_tmx.py
