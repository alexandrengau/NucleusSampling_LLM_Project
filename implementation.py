import os
import json
from tqdm import tqdm, trange
import torch
import torch.nn
import torch.nn.functional as F


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(model, prompt, batch_size, gen_len, sep, temp=False, k=False, p=False):
    context, ends = prompt
    output=[
        {
            'ended' : False,
            "tokens" : [],
            'len' : 0,
            'nll4tok'    : [],
            "context" : context[i].to(device).numpy().tolist()
        }
    
        for i in range(batch_size)
    ]

    for _ in tqdm(range(gen_len)) :

        #Récupère les score pour le dernier mot du contexte (pas sûr que les 4 prochaines lignes soient correctes)
        output_model = model(context)[0]
        scores = torch.zeros(output_model.size(0), output_model.size(2))
        for i in range(len(output_model)) :
            scores[i,:] = output_model[i, ends[i],:]
        

        #Mise en place de la tempéerature 
        if temp is False:
            probs = F.softmax(scores, dim=-1)
        else :
            probs =  F.softmax(scores.div_(temp), dim=-1)

        logprobs = F.log_softmax(scores, dim=-1)

        if k is not False:
            probs, tokens = torch.topk(probs, k)
            selected_indices = probs.multinomial(1)
            token_selected = tokens.gather(1, selected_indices.view(-1, 1))
            logprobs_selected = probs.gather(1, selected_indices.view(-1, 1)).log()

        elif p is not False:
            probs_sorted, _ = torch.sort(probs, descending=True)
            probs_cummu = torch.cumsum(probs_sorted, dim =-1)
            probs_inf_to_p = probs_cummu <= p
            nb_indice_to_kept = torch.sum(probs_inf_to_p, dim=-1, dtype=torch.int64)
            
            token_selected = torch.zeros(batch_size, 1)
            logprobs_selected = torch.zeros(batch_size, 1)
            for i in range(batch_size) :
                if nb_indice_to_kept[i].item() == 0:
                    nb_indice_to_kept[i] = 1
                probs_batch, tokens = torch.topk(probs[i,:], nb_indice_to_kept[i].item())
                selected_indices = probs_batch.multinomial(1)
                token_selected[i] = tokens.gather(0, selected_indices.view(-1))
                logprobs_selected[i] = probs_batch.gather(0, selected_indices.view(-1)).log()
        
        else :
            _, token_selected = probs.topk(1)
            logprobs_selected = logprobs.gather(1, tokens.view(-1, 1))
        
        #rajout d'une nouvelle ligne vide dans le prompt qu'on va remplir
        filler = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        context = torch.cat([context, filler], dim=1)
        for i in range(batch_size) :
            out = output[i]
            if out["ended"] :
                continue

            token = token_selected[i].item()
            logprob = logprobs_selected[i].item()

            if  token == sep :
                out["ended"] = True

            out['tokens'].append(token)
            out['nll4tok'].append(-logprob)
            out['len'] += 1

            #Remplissage de la ligne vide
            ends[i] +=1
            context[i, ends[i]] = token
    
    return output 



def beam_search_decode(model, prompt, gen_length, sep, w):
    context, ends = prompt
    current_len = ends[0] + 1
    batch_size = context.size(0)

    # Crée le tensor avec w fois le contexte répété
    beam = context.repeat(w, 1, 1).transpose(0,1).reshape(batch_size*w, current_len)
    
    # Création de la matrice offset, elle est utilisée comme décalage pour indexer correctement la matrice beam
    beam_offset = (torch.arange(batch_size)*w*w).repeat(w, 1).t().reshape(-1).to(device)
    
    current_outputs, current_logprobs, current_Negalogprobs = [[None for _ in range(batch_size)] for _ in range(3)]


    for i in trange(gen_length):
        if i == 0:
            #récupere log prob pour le contexte
            scores = model(context)[0][:, -1, :] 
            logprobs = F.log_softmax(scores, -1)  
            w_logprobs, w_tokens = torch.topk(logprobs, w)
            tokens = w_tokens.view(-1)  
            logprobs = w_logprobs.view(-1)

            #Ajout pour chaque ligne du
            beam = torch.cat([beam, tokens.unsqueeze(-1)], -1) 
            beam_NegaLogprobs = -logprobs.unsqueeze(-1)
            beam_logprobs = logprobs.view(batch_size, w)

        else:
            #récupere les logprob pour les faisceaux
            scores = model(beam)[0][:, -1, :] 
            logprobs = F.log_softmax(scores, -1)  
            w_logprobs, w_tokens = torch.topk(logprobs, w)

            #On calcul les logprob accumulés
            accumulated_logprobs = (w_logprobs + beam_logprobs.view(-1).repeat(w, 1).t()).reshape(batch_size, w*w)
            #On va chercher pour chaque context les w suites possibles 
            beam_logprobs, beam_idxs = accumulated_logprobs.topk(w)  # both: (batch_size, w)
            beam_idxs = beam_idxs.view(-1) + beam_offset  # (batch_size*w,)
            tokens = w_tokens.view(-1)[beam_idxs]  # (batch_sze*w,)
            logprobs = w_logprobs.view(-1)[beam_idxs]  # (batch_size*w,)
            
            #Actualisation du faisceau
            beam = beam.repeat(1, w).view(batch_size*w*w, current_len)[beam_idxs]  # (batch_size*w, current_len)
            beam = torch.cat([beam, tokens.unsqueeze(-1)], -1)  # (batch_size*w, current_len+1)
            beam_NegaLogprobs = beam_NegaLogprobs.repeat(1, w).view(batch_size*w*w, current_len - ends[0] - 1)[beam_idxs]
            beam_NegaLogprobs = torch.cat([beam_NegaLogprobs, -logprobs.unsqueeze(-1)], -1)

        current_len += 1

        for b in range(batch_size):
            offset = b*w
            current_tokens = tokens[offset:offset+w].tolist()
            for idx, tok in enumerate(current_tokens):
                if tok == sep and (current_outputs[b] is None or beam_logprobs[b, idx] > current_logprobs[b]):
                    current_outputs[b] = beam[offset+idx]
                    current_Negalogprobs[b] = beam_NegaLogprobs[offset+idx]
                    current_logprobs[b] = beam_logprobs[b, idx]

    outputs = [{} for _ in range(batch_size)]
    for b, output in enumerate(outputs):
        output['context'] = context[b].tolist()
        output['ended'] = current_outputs[b] is not None
        output['tokens'] = (current_outputs[b] if current_outputs[b] is not None else beam[w*b]).tolist()
        output['tokens'] = output['tokens'][len(output['context']):]
        output['nll4tok'] = (current_Negalogprobs[b] if current_Negalogprobs[b] is not None else beam_NegaLogprobs[w*b]).tolist()
        output['len'] = len(output['tokens'])

    return outputs


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
    #Filtering of dataset
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
            dataset = utils.load_dataset(dataset_path=os.path.join(subdir, config["context_path"]), batch_size = config["batch_size"], device = device, bs=config["w"] is not False)

        if config["cache_path"] is not False and not os.path.exists(config["cache_path"]):
            torch.save(dataset, config["cache_path"])
    else:
        dataset = [ None for _ in range(config["n"] // config["batch_size"]) ]

    model.eval()
    outputs = []
    writer = open(os.path.join(subdir, config["output_path"]), "w")

    for _, batch in enumerate(tqdm(dataset, desc="Generating")):
        with torch.no_grad():

            if config["w"] is False:
                output = decode(model = model,
                                prompt = batch,
                                batch_size = config["batch_size"],
                                gen_len = gen_length,
                                sep = SEP, 
                                temp = config["t"], 
                                k = config["k"], 
                                p = config["p"])
            else :
                output = beam_search_decode(model = model,
                                       prompt = batch,
                                       gen_length = gen_length,
                                       sep = SEP, 
                                       w = config["w"])
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