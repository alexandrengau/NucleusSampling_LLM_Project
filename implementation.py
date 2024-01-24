import os
import json
from tqdm import tqdm, trange
import torch
import torch.nn
import torch.nn.functional as F


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE : ",device)


def decode(model, prompt, batch_size, gen_len, sep, temp=False, k=False, p=False, device=device):
    context, ends = prompt
    output = [
        {
            'ended': False,
            "tokens": [],
            'len': 0,
            'nll4tok': [],
            "context": context[i].to(device).cpu().numpy().tolist()
        }
        for i in range(batch_size)
    ]

    for _ in tqdm(range(gen_len)):

        # Get scores for the last word of the context
        output_model = model(context.to(device))[0]
        scores = torch.zeros(output_model.size(0), output_model.size(2), device=device)
        for i in range(len(output_model)):
            scores[i, :] = output_model[i, ends[i], :]

        # Set up temperature
        if temp is False:
            probs = F.softmax(scores, dim=-1)
        else:
            probs = F.softmax(scores.div_(temp), dim=-1)

        logprobs = F.log_softmax(scores, dim=-1)

        if k is not False:
            probs, tokens = torch.topk(probs, k)
            selected_indices = probs.multinomial(1)
            token_selected = tokens.gather(1, selected_indices.view(-1, 1))
            logprobs_selected = probs.gather(1, selected_indices.view(-1, 1)).log()

        elif p is not False:
            probs_sorted, _ = torch.sort(probs, descending=True)
            probs_cummu = torch.cumsum(probs_sorted, dim=-1)
            probs_inf_to_p = probs_cummu <= p
            nb_indice_to_kept = torch.sum(probs_inf_to_p, dim=-1, dtype=torch.int64)

            token_selected = torch.zeros(batch_size, 1, device=device)
            logprobs_selected = torch.zeros(batch_size, 1, device=device)
            for i in range(batch_size):
                if nb_indice_to_kept[i].item() == 0:
                    nb_indice_to_kept[i] = 1
                probs_batch, tokens = torch.topk(probs[i, :], nb_indice_to_kept[i].item())
                selected_indices = probs_batch.multinomial(1)
                token_selected[i] = tokens.gather(0, selected_indices.view(-1))
                logprobs_selected[i] = probs_batch.gather(0, selected_indices.view(-1)).log()

        else:
            _, token_selected = probs.topk(1)
            logprobs_selected = logprobs.gather(1, token_selected.view(-1, 1))

        # Add a new empty line in the prompt to fill
        filler = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        context = torch.cat([context, filler], dim=1)
        for i in range(batch_size):
            out = output[i]
            if out["ended"]:
                continue

            token = token_selected[i].item()
            logprob = logprobs_selected[i].item()

            if token == sep:
                out["ended"] = True

            out['tokens'].append(token)
            out['nll4tok'].append(-logprob)
            out['len'] += 1

            # Fill the empty line
            ends[i] += 1
            context[i, ends[i]] = token

    return output



def beam_search_decode(model, prompt, gen_length, sep, w, device=device):
    context, ends = prompt
    current_len = ends[0] + 1

    beam = context.repeat(w, 1).to(device)

    current_outputs, current_logprobs, current_Negalogprobs = [None for _ in range(3)]
    for i in trange(gen_length):
        if i == 0:
            # Get log prob for the context
            scores = model(context.to(device))[0][:, -1, :]
            logprobs = F.log_softmax(scores, -1)
            w_logprobs, w_tokens = torch.topk(logprobs, w)
            tokens = w_tokens.view(-1)
            logprobs = w_logprobs.view(-1)

            # Add for each line
            beam = torch.cat([beam, tokens.unsqueeze(-1)], -1)
            beam_NegaLogprobs = -logprobs.unsqueeze(-1)
            beam_logprobs = logprobs.view(1, w)

        else:
            # Get log prob for the beams
            scores = model(beam.to(device))[0][:, -1, :]
            logprobs = F.log_softmax(scores, -1)
            w_logprobs, w_tokens = torch.topk(logprobs, w)

            # Calculate accumulated logprobs
            accumulated_logprobs = (w_logprobs + beam_logprobs.repeat(w, 1).t())
            beam_logprobs, beam_idxs = accumulated_logprobs.view(-1).topk(w)

            tokens = w_tokens.view(-1)[beam_idxs]
            logprobs = w_logprobs.view(-1)[beam_idxs]

            # Update the beam
            beam_ww = beam.repeat(1, w).view(w*w, current_len)
            beam_NegaLogprobs_ww = beam_NegaLogprobs.repeat(1, w).view(w*w, current_len - ends[0] - 1)
            beam = torch.cat((beam, torch.zeros(w, 1, dtype=torch.int64).to(device)), dim=1)
            beam_NegaLogprobs = torch.cat((beam_NegaLogprobs, torch.zeros(w, 1).to(device)), dim=1)
            for i in range(w):
                beam[i] = torch.cat([beam_ww[beam_idxs[i]], tokens[i].view(1)])
                beam_NegaLogprobs[i] = torch.cat([beam_NegaLogprobs_ww[beam_idxs[i]], -logprobs[i].view(1)])

        current_len += 1

        for idx, tok in enumerate(tokens.tolist()):
            if tok == sep and (current_outputs is None or beam_logprobs[idx] > current_logprobs):
                current_outputs = beam[idx]
                current_Negalogprobs = beam_NegaLogprobs[idx]
                current_logprobs = beam_logprobs[idx]

    output = [{}]

    output[0]['context'] = context[0].tolist()
    output[0]['ended'] = current_outputs is not None
    output[0]['tokens'] = (current_outputs if current_outputs is not None else beam[w-1]).tolist()
    output[0]['tokens'] = output[0]['tokens'][len(output[0]['context']):]
    output[0]['nll4tok'] = (current_Negalogprobs if current_Negalogprobs is not None else beam_NegaLogprobs[w-1]).tolist()
    output[0]['len'] = len(output[0]['tokens'])

    return output


def main():
    config = utils.load_json()
    subdir = 'data'

    if config["seed"] is False:
        import time
        millis = int(round(time.time() * 1000))
        config["seed"] = millis

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # Load dataset for conditional generation
    utils.download_datasets()
    # Tokenization of dataset with the model
    utils.encode_json()
    # Filtering of dataset
    utils.filter_for_conditional()

    tokenizer = GPT2Tokenizer.from_pretrained(config["model_name"], do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(config["model_name"])
    model.to(device)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    # Compute the max input length for the Transformer
    gen_length = config["gen_len"]

    if config["context_path"] is not False:
        if config["cache_path"] is not False and os.path.exists(config["cache_path"]):
            dataset = torch.load(os.path.join(subdir, config["cache_path"]), map_location=device)
        else:
            dataset = utils.load_dataset(dataset_path=os.path.join(subdir, config["context_path"]),
                                         batch_size=config["batch_size"], device=device, bs=config["w"] is not False)

        if config["cache_path"] is not False and not os.path.exists(config["cache_path"]):
            torch.save(dataset, config["cache_path"])
    else:
        dataset = [None for _ in range(config["n"] // config["batch_size"])]

    model.eval()
    outputs = []
    writer = open(os.path.join(subdir, config["output_path"]), "w")

    for _, batch in enumerate(tqdm(dataset, desc="Generating")):
        with torch.no_grad():
            if config["w"] is False:
                output = decode(model=model,
                                prompt=batch,
                                batch_size=config["batch_size"],
                                gen_len=gen_length,
                                sep=SEP,
                                temp=config["t"],
                                k=config["k"],
                                p=config["p"],
                                device=device)
            else:
                output = beam_search_decode(model=model,
                                            prompt=batch,
                                            gen_length=gen_length,
                                            sep=SEP,
                                            w=config["w"],
                                            device=device)
            outputs.extend(output)
            for o in output:
                o['cond'] = tokenizer.decode(o['context'])
                o['gen'] = tokenizer.decode(o['tokens'])
                print(json.dumps(o), file=writer, flush=True)
    writer.close()

    ppl = utils.perplexity()
    print('ceci devrait être le resultat du ppl:', ppl)


if __name__ == '__main__':
    main()