import torch
import numpy as np
import torch.nn as nn
import scipy.io as sio



def encoding_onehot(target,device, nclasses=20):
    target_onehot = torch.FloatTensor(target.size(0), nclasses).to(device)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

class ReadTxt(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(ReadTxt, self).__init__()
    def forward(self,path):
        fh = open(path, 'r')
        imgs = []
        labels = []
        imgs_lab = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            label = words[1:]
            # label = np.array(map(int,label))
            #print(label)
            label = np.array(label).astype(int)
            imgs_lab.append((words[0], label))
            labels.append(label)
            imgs.append(words[0])
        #labels = torch.Tensor(labels).int()
        return imgs, labels, imgs_lab


        
        
        
		


			
		
