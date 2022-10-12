
import torch
import torch.nn as nn
import  numpy_tes as np

inputs = torch.FloatTensor([0,1,0,0,0,1])
outputs = torch.LongTensor([0,1])
inputs = inputs.view((1,3,2))
outputs = outputs.view((1,2))
weight_CE = torch.FloatTensor([1,2,3])
ce = nn.CrossEntropyLoss(ignore_index=255,weight=weight_CE)
loss = ce(inputs,outputs)
print(loss)