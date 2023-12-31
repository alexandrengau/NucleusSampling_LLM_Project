import json
from tqdm import tqdm

import torch
import torch.nn
import torch.nn.functional as F


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