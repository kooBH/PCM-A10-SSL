import torch
import torch.nn as nn


def get_model(hp):

    return

def run(data,model,criterion,ret_output=False): 
    input = data['input'].to(device)
    target = data['target'].to(device)
    output = model(input)

    loss = criterion(output,target).to(device)

    if ret_output :
        return output, loss
    else : 
        return loss