import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import datetime
import random
print(datetime.datetime.now().month*100+datetime.datetime.now().day)
class discriminator(nn.Module):
    def __init__(self,itemCount):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(itemCount*2,125),

            nn.ReLU(True),

            nn.Linear(125,1),
            nn.Sigmoid()
        )
    def forward(self,data,condition):

        result=self.dis( torch.cat((data,condition),1) )
        return result
class generator(nn.Module):
    def __init__(self,itemCount):
        super(generator,self).__init__()
        self.itemCount=itemCount

        self.gen=nn.Sequential(
            nn.Linear(itemCount,400),

            nn.ReLU(True),
            nn.Linear(400, 400),

            nn.ReLU(True),
            nn.Linear(400,400),

            nn.ReLU(True),
            nn.Linear(400, itemCount),
            nn.Tanh()
        )
    def forward(self,x):

        result=self.gen(x)
        return result