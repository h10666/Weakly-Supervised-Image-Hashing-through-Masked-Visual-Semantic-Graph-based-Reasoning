import argparse
import numpy as np
#from tqdm import tqdm

import torch

import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from utils import ReadTxt
import scipy.io as sio

from network import AlexNetHashGAT
from dataset_Alexnet import dataset
import pickle
# import sys
# import importlib
# importlib.reload(sys)


from calc_hr_new import calc_map


parser = argparse.ArgumentParser()




## dataset arguments

parser.add_argument('--dataroot', default='/home/jinlu/108.00.000/dataset/nus/ImageData/Flickr/', help='path to dataset')
parser.add_argument('--trainFile',default = './dataset/nus/mm2020_setting/nus_im_tags_sample.plkt', help='file of training set')
parser.add_argument('--databaseFile',default = './dataset/nus/mm2020_setting/nus_train_10concept.txt', help='file of database set')
parser.add_argument('--testFile',default = './dataset/nus/mm2020_setting/nus_test_10concept.txt', help='file of database set')


parser.add_argument('--codeNum', type=int, default=64, help='hash code numbers')


## GPU and parameters ( e.g., lr, batchsize,) argument
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')


parser.add_argument('--outf', default='./models/nus/hash/model_64bit_new_lr', help='folder to output images and model checkpoints')



opt = parser.parse_args()
opt.cuda = True


### imgage and tags for training 
# im_tags_tr = pickle.load(open('./dataset/nus/mm2020_setting/nus_im_tags_sample.plk','rb'), encoding="bytes")

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# construct the network
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
net = AlexNetHashGAT(codeNum=opt.codeNum).to(device)


print('The network structure as follows.')
print(net)
if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=range(ngpu))





# eval
print('loading database and test file...')
read_txt = ReadTxt()

_, _, db_ims_lab = read_txt(path=opt.databaseFile)
_, _, te_ims_lab = read_txt(path=opt.testFile)


database_dataset = dataset(dir_path=opt.dataroot ,img_lab=db_ims_lab, 
                           transform_pre=transforms.Compose([
                               transforms.Resize((224,224))
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                           ]))


test_dataset = dataset(dir_path=opt.dataroot ,img_lab=te_ims_lab, 
                           transform_pre=transforms.Compose([
                               transforms.Resize((224,224))
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                           ]))







database_dataloader = torch.utils.data.DataLoader(database_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
net.eval()
eval_data = []
weight_path = '%s/net_final.pth' % (opt.outf)
net.load_state_dict(torch.load(weight_path))

rB = []
    
rL = []
    


qB = []
qL = []
with torch.no_grad():
    for i, data in enumerate(database_dataloader, 0):
        net.zero_grad()
        img = data[0].to(device)
        lb = data[1].to(device)
            
        batch_size = img.size()[0]
            
        code,_ = net(img)
             
        b1 = (code<0.5).float()*-1
        b2 = (code>=0.5).float()*1
        b = b1 + b2
            
        rB.extend(b.cpu().data.numpy())
        rL.extend(lb.cpu().data.numpy())

    for i, data in enumerate(test_dataloader, 0):
        net.zero_grad()
        img = data[0].to(device)
        lb = data[1].to(device)
        batch_size = img.size()[0]

        code,_ = net(img)

        b1 = (code<0.5).float()*-1
        b2 = (code>=0.5).float()*1
        b = b1 + b2
        qB.extend(b.cpu().data.numpy())
        qL.extend(lb.cpu().data.numpy())

rB = np.array(rB)
rL = np.array(rL)

qB = np.array(qB)
qL = np.array(qL)
    

map_v, p_5000,p_r_2 = calc_map(qB=qB, rB=rB, queryL=qL, retrievalL=rL, knn=5000)
print_str = '| map: %f | p_5000: %f| p_r_2: %f' % ( map_v, p_5000,p_r_2)
print(print_str)

