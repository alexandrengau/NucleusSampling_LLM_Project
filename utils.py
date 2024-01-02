import json
import os
import requests
from tqdm import tqdm

import torch
import torch.nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer


def load_json(file_path="./config.json"):
    """load the configuration file"""
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config 


def load_dataset(dataset_path, batch_size, device, bs=False):
    """ Loads data from a jsonl file with "tokens" attribute """
    dataset, count, tokens, ends, last_len = [], 0, [], [], None
    with open(dataset_path, encoding='utf_8') as f:
        for line in tqdm(f):
            j = json.loads(line.strip())
            cur_len = len(j['tokens'])
            # beam search batches must only contain contexts of the same length
            if not bs:
                tokens.append(j['tokens'])
                end = cur_len-1
                ends.append(end)
                count += 1
                if count == batch_size:
                    max_len = max(ends)
                    data = torch.zeros(batch_size, max_len+1).long()
                    for b, (toks, end) in enumerate(zip(tokens, ends)):
                        data[b, :end+1] = torch.Tensor(toks)
                    data = data.to(device)
                    dataset.append((data, ends))
                    tokens, ends = [], []
                    count = 0
            else:
                if last_len is None:
                    last_len = cur_len
                elif last_len != cur_len  or count == batch_size:
                    data = torch.zeros(count, last_len).long()
                    for b, (toks, end) in enumerate(zip(tokens, ends)):
                        data[b, :last_len] = torch.Tensor(toks)
                    data = data.to(device)
                    dataset.append((data, ends))
                    tokens, ends = [], []
                    count = 0
                    last_len = cur_len
                tokens.append(j['tokens'])
                ends.append(cur_len-1)
                count += 1
    if bs and len(tokens) > 0:
        data = torch.zeros(count, last_len).long()
        for b, (toks, end) in enumerate(zip(tokens, ends)):
            data[b, :last_len] = torch.Tensor(toks)
        data = data.to(device)
        dataset.append((data, ends))

    return dataset

def encode_json():
    config = load_json() #input and output files are set in it
    subdir = 'data'
    subdir = subdir.replace('\\','/')

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"], do_lower_case=True)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    filename = config["context_output_path"]
    if os.path.exists(os.path.join(subdir, filename)):
        print(f"The dataset {filename} is already tokenized")
    else : 
        print(f"The dataset {filename} is being tokenized")
        with open(os.path.join(subdir, config["context_input_path"]), 'r') as input_file, open(os.path.join(subdir, config["context_output_path"]), 'w') as output_file:
            total_lines = sum(1 for _ in input_file)
            input_file.seek(0)
            for json_str in tqdm(input_file, total=total_lines, desc="Tokenization"):
                j = json.loads(json_str.strip())
                j['tokens'] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(j['text']))

                output_file.write(json.dumps(j) + '\n')


# For some reason HuggingFace only represent newlines inbetween non-whitespace tokens.
# So we hardcode this in to avoid strange, uninterpretable workarounds
NEWLINE = 198

def sublist_end_index(list1, list2):
    s1, s2 = ' '.join(map(str, list1)), ' '.join(map(str, list2))
    if s1 in s2:
        return s2[:s2.index(s1)].count(' ') + s1.count(' ') + 1
    else:
        return None

def filter_for_conditional():
    config = load_json()
    subdir = 'data'
    subdir = subdir.replace('\\','/')

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"], do_lower_case=True)

    filename = config["conditional_output_path"]
    if os.path.exists(os.path.join(subdir, filename)):
        print(f"The dataset {filename} is already filtered")
    else : 
        print(f"The dataset {filename} is being filtered")
        with open(os.path.join(subdir, config["conditional_input_path"]), 'r') as input_file, open(os.path.join(subdir, config["conditional_output_path"]), 'w') as output_file:
            total_lines = sum(1 for _ in input_file)
            input_file.seek(0)
            num = 0
            for json_str in tqdm(input_file, total=total_lines, desc="Filtering"):
                j = json.loads(json_str.strip())
                idx = sublist_end_index([NEWLINE, NEWLINE], j['tokens'])
                if idx is not None and idx < config["conditional_m"]:
                    j['tokens'] = j['tokens'][:idx]
                    output_file.write(json.dumps(j) + '\n')
                    num += 1
                    if num >= config["conditional_n"]:
                        break

def download_datasets():
    subdir = 'data'
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\','/') #for Windows just in case

    for split in ['train', 'valid', 'test']:
        filename = 'small-117M' + "." + split + '.jsonl'
        if os.path.exists(os.path.join(subdir, filename)):
            print(f"The dataset {filename} is already downloaded")
        else : 
            print(f"The dataset {filename} is being downloaded")
            r = requests.get("https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/" + filename, stream=True)

            with open(os.path.join(subdir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)