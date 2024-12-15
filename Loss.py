import torch
import torch.nn as nn

# class Loss:
#     def __init__(self, output, labels):
#         self.output = output
#         self.labels = labels
    
#     def CrossEntropyLoss(self):
#         loss = nn.CrossEntropyLoss()
#         ls_CrossEntropy = loss(self.output,self.labels)
#         return ls_CrossEntropy

def CrossEntropyLoss(output,target):
    loss = nn.CrossEntropyLoss()
    ls_CrossEntropy = loss(output,target)
    acc = compute_accuracy(output,target)
    return acc,ls_CrossEntropy

def compute_accuracy(output,target):
    batch_size = output.shape[0]
    _, pred = torch.max(output, dim=-1)
    #print(pred)
    correct = pred.eq(target).sum() * 1.0
    acc = correct / batch_size 
    return acc  