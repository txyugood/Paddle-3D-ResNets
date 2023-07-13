import os
import torch
from collections import OrderedDict
paddle_list = open('paddle.txt')
torch_list = open('pytorch.txt')
dict = torch.load('/home/aistudio/data/data51188/r3d50_K_200ep.pth')
state_dict = dict['state_dict']


paddle_state_dict = OrderedDict()
for p, t in zip(paddle_list, torch_list):
    p = p.strip()
    t = t.strip()
    if 'fc' not in t:
        paddle_state_dict[p] = state_dict[t].detach().cpu().numpy()
    else:
        paddle_state_dict[p] = state_dict[t].detach().cpu().numpy().T

f = open('/home/aistudio/Paddle-3D-ResNets/paddle_resnet50_mk.pdparams', 'wb')
import pickle
pickle.dump(paddle_state_dict, f)
f.close()
