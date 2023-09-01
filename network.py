
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from pdb import set_trace as breakpoint





    

class AlexNetHashGAT(nn.Module):
    """docstring for AlexNetHash"""
    def __init__(self, codeNum,alpha=0.2):
        super (AlexNetHashGAT, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        del alexnet.classifier[6]
        self.Alexnet = alexnet
        self.hash = nn.Sequential(nn.Linear(4096,codeNum),nn.Sigmoid())
        
        self.embedding = nn.Linear(codeNum,900)
        
    def forward(self,x):
        x=self.Alexnet(x)
        code = self.hash(x)
        emb = self.embedding(code)
        return code, emb  






