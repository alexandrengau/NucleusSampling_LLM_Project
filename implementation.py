import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer

data_path=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def decode(model, batch_size, max_len, sep, prompt, temp=None, k=None, p=None, greedy=None):
    output=[
        {
            'ended' : False,
            "tokens" : [],
            'len' : 0,
            "context" : prompt[i].cpu().numpy().tolist
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


        