import torch
import torch.nn as nn


def get_model(hp):

    return

def run(data,label,model,criterion,ret_output=False): 
    input = data
    output = model(input)

    loss = criterion(output,label)

    if ret_output :
        return output, loss
    else : 
        return loss