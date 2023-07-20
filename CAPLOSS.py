import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb
import cv2


def CAPLoss(img):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    unloader = transforms.ToPILImage()
    image = img.cpu().clone()
    image = image.squeeze(0)
    haze = unloader(image)
    x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

    HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    image = np.asarray(HSV_img)
    H, S, V = cv2.split(image)
    # print(img)
    # print(S - V)
    totensor = transforms.ToTensor()
    S = totensor(S)
    V = totensor(V)
    l1loss = L1Loss()
    loss = l1loss(S, V)
    # print(loss)



    # loss = L1Loss(size_average=True)(S, V)
    return loss


    # loss.backward()