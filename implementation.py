import argparse
import os
import logging
import json

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions.gumbel import Gumbel


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import utils

data_path=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def decode(model, batch_size, max_len, sep, prompt, temp=None, k=None, p=None, greedy=None):
    output=[
        {
            'ended' : False,
            "tokens" : [],
            'len' : 0,
            "context" : prompt[i].to(device).numpy().tolist
        }
    
        for i in range(batch_size)
    ]


    for _ in range(max_len) :

        #Récupère les score pour le dernier mot du contexte (pas sûr que les 4 prochaines lignes soient corrects)
        output_model = model(prompt)[0]
        scores = torch.zeros(output_model.size(0), output_model.size(2))
        for i in range(len(output_model)) :
            scores[i,:] = output_model[i, len(output[i]["context"]),:]
        

        #Mise en place de la tempéerature 
        if temp is not None :
            probs = F.softmax(scores, dim=-1)
        else :
            probs =  F.softmax(scores.div_(temp), dim=-1)

        if k :
            indices_to_remove = probs < torch.topk(probs, k)[0][..., -1, None]
            probs[indices_to_remove] = 0
            tokens = probs.multinomial(1)

        elif p :
            probs, indices = torch.sort(probs, descending=True)
            probs_cummu = torch.cumsum(probs, dim =-1)
            indice_to_remove = probs_cummu > p
            probs_cummu[indice_to_remove] = 0
            indice_tokens = probs_cummu.multinomial(1).view(-1,1)
            tokens = indices.gather(1,indice_tokens)
        
        else :
            _, tokens = probs.topk(1)
        
        #rajout d'une nouvelle ligne vide dans le prompt qu'on va remplir
        filler = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        prompt = torch.cat([prompt, filler], dim=1)
        for i in range(batch_size) :
            out = output[i] 
            if out["ended"] :
                continue

            token = tokens[i].items()
            if  token == sep :
                out["ended"] = True

            out['tokens'].append(token)
            out['len'] += 1

            #Remplissage de la ligne vide
            prompt[i, prompt[i].size] = token
    
    return output 



def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    # CHECK_VALIDITY = True
    # if CHECK_VALIDITY:
    #     g_inv = _shift_gumbel_maximum(g, Z, dim)
    #     assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
    return g, argmax

def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    u = T.unsqueeze(dim) - g_phi + torch.log1p(-torch.exp(g_phi - Z.unsqueeze(dim)))
    return T.unsqueeze(dim) - F.relu(u) - torch.log1p(torch.exp(-u.abs()))

def gumbel_sbs_decode(model, init, w, max_len, sep, device, batch_size):
    if init is None:
        context = torch.full((batch_size, 1), sep, dtype=torch.long, device=device)
        ends = torch.zeros_like(context[:, 0])
    else:
        context, ends = init
    assert (all([ends[i] == ends[0] for i in range(len(ends))]))
    cur_len = ends[0] + 1
    batch_size = context.size(0)
    context_cpu = context.cpu()

    beam = context.repeat(w, 1, 1).transpose(0, 1).reshape(batch_size * w, cur_len)
    beam_offset = (torch.arange(batch_size) * w).repeat(w, 1).t().reshape(-1).to(device)
    beam_nlls = torch.zeros(batch_size * w, 1, device=device)
    beam_ll = torch.zeros(batch_size, w, device=device)
    best_outputs, best_ll, best_gumbel, best_nlls = [[None for _ in range(batch_size)] for _ in range(4)]

    for i in trange(max_len):
        if i == 0:
            logits = model(context)[0][:, -1, :]  # (batch_size, V)
            logprobs = F.log_softmax(logits, -1)  # (batch_size, V)
            gumbel = gumbel_like(logprobs) + logprobs  # (batch_size, V)
            z, _ = gumbel.max(dim=-1, keepdims=True)  # (batch_size, 1)
            gumbel_tilde = -(-torch.exp(-z)+torch.exp(-gumbel)+1.0).log()  # (batch_size, V), +1.0 is exp(-G_phi_N)

            beam_gumbel, w_tokens = torch.topk(gumbel_tilde, w)  # (batch_size, w)
            cur_tokens = w_tokens.view(-1)  # (batch_size*w,)
            cur_lls = logprobs.gather(-1, w_tokens).view(-1)  # (batch_size*w,)

        else:
            logits = model(beam)[0][:, -1, :]  # (batch_size*w, V)
            V = logits.size(-1)
            logprobs = F.log_softmax(logits, -1)  # (batch_size*w, V)
            gumbel = Gumbel(loc=logprobs+beam_ll.view(batch_size*w, 1), scale=1.0).sample()  # (batch_size*w, V)
            z, _ = gumbel.max(dim=-1, keepdims=True)  # (batch_size*w, 1)
            gumbel_tilde, _ = gumbel_with_maximum(
                logprobs+beam_ll.view(batch_size*w, 1),  # (batch_size*w, V)
                beam_gumbel.view(batch_size*w), # (batch_size*w,)
                dim=-1)  # gumbel_tilde: (batch_size*w, V)

            beam_gumbel, beam_idxs = torch.topk(gumbel_tilde.view(batch_size, w*V), w)  # (batch_size, w), beam_idxs in [0,w*V)
            beam = beam[beam_idxs.view(-1)//V + beam_offset]  # (batch_size*w, cur_len)
            cur_tokens = (beam_idxs % V).view(-1)  # (batch_size*w,)
            cur_lls = logprobs.view(batch_size, w*V).gather(-1, beam_idxs).view(-1)  # (batch_size*w,)

        beam = torch.cat([beam, cur_tokens.unsqueeze(-1)], -1)  # (batch_size*w, cur_len+1)
        beam_nlls = torch.cat([beam_nlls, cur_lls.unsqueeze(-1)], -1)
        beam_ll += cur_lls.view(batch_size, w)
        cur_len += 1

        if cur_tokens.eq(sep).sum() > 0:
            for b in range(batch_size):
                offset = b * w
                toks = cur_tokens[offset:offset + w].tolist()
                for idx, tok in enumerate(toks):
                    if tok == sep and (best_outputs[b] is None or beam_gumbel[b, idx] > best_gumbel[b]):
                        best_outputs[b] = beam[offset + idx]
                        best_nlls[b] = beam_nlls[offset + idx]
                        best_ll[b] = beam_ll[b, idx]
                        best_gumbel[b] = beam_gumbel[b, idx]
        if all(best_ll[b] is not None and best_ll[b] > beam_ll[b, 0] for b in range(batch_size)):
            break

    outputs = [{} for _ in range(batch_size)]
    for b, output in enumerate(outputs):
        output['context'] = context_cpu[b].tolist()
        output['ended'] = best_outputs[b] is not None
        output['tokens'] = (best_outputs[b] if best_outputs[b] is not None else beam[w * b]).tolist()
        output['tokens'] = output['tokens'][len(output['context']):]
        output['nll4tok'] = (best_nlls[b] if best_nlls[b] is not None else beam_nlls[w * b]).tolist()
        output['nll4tok'] = [-x for x in output['nll4tok'][1:]]
        output['ppl4tok'] = [np.exp(nll) for nll in output['nll4tok']]
        output['ppl'] = np.exp(sum(output['nll4tok']) / len(output['nll4tok']))
        output['len'] = len(output['tokens'])

    return outputs


def main():
    config = utils.load_json()

    if config["seed"] is None:
        import time
        millis = int(round(time.time() * 1000))
        config["seed"] = millis


    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Load dataset for conditional generation
    utils.download_datasets()
    # Tokenization of dataset with the model
    utils.encode_json()
    #Filtering of dataset
    utils.filter_for_conditional()

    assert(not (config["k"] and config["p"]))
    with open(config["output_path"], 'w'):
        pass

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"], do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(config["model_name"])
    model.to(device)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    # Compute the max input length for the Transformer
    max_length = config["max_len"]

    if config["context_path"] is not None:
        if config["cache_path"] is not None and os.path.exists(config["cache_path"]):
            dataset = torch.load(config["cache_path"], map_location=device)
        else:
            dataset = utils.load_dataset(config["context_path"], config["batch_size"], device, bs=config["w"] is not None)

        if config["cache_path"] is not None and not os.path.exists(config["cache_path"]):
            torch.save(dataset, config["cache_path"])
    else:
        dataset = [ None for _ in range(config["n"] // config["batch_size"]) ]

    model.eval()
    outputs = []
    writer = open(config["output_path"], "w")
    try:
        for b, batch in enumerate(tqdm(dataset[config["skip"]:config["fin"]], desc="Generating")):
            with torch.no_grad():
                if config["w"] is None:
                    output = decode(model, config["batch_size"], max_length, SEP, device, 
                                temp=config["t"], k=config["k"], p=config["p"], greedy=config["greedy"],
                                m=config["m"], init=batch)
                else:
                    output = gumbel_sbs_decode(model, batch, config["w"], max_length, SEP, device, config["batch_size"])
                outputs.extend(output)
                for o in output:
                    o['string'] = tokenizer.decode(o['tokens'])
                    print(json.dumps(o), file=writer, flush=True)
    except (KeyboardInterrupt, SystemExit):
        pass

    writer.close()

if __name__ == '__main__':
    main()