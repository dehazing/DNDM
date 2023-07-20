
from skimage import color
from torch.nn import L1Loss, MSELoss
from torchvision import transforms
import cv2


def LabLoss(dehaze,hazy):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    unloader = transforms.ToPILImage()
    image = dehaze.cpu().clone()
    image = image.squeeze(0)
    dehaze = unloader(image)

    unloader = transforms.ToPILImage()
    image = hazy.cpu().clone()
    image = image.squeeze(0)
    hazy = unloader(image)
    # x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)
    #
    # HSV_img = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # image = np.asarray(HSV_img)
    # H, S, V = cv2.split(image)
    # # print(img)
    # # print(S - V)
    totensor = transforms.ToTensor()
    # S = totensor(S)
    # V = totensor(V)
    # l1loss = L1Loss()
    # loss = l1loss(S, V)
    # print(loss)
    # skimage.color
    lab_dehaze=color.rgb2lab(dehaze)
    ldehaze, a, b = cv2.split(lab_dehaze)
    lab_hazy = color.rgb2lab(hazy)
    lhazy, a, b = cv2.split(lab_hazy)
    ldehaze = totensor(ldehaze)
    lhazy = totensor(lhazy )
    l1loss = L1Loss()
    loss = l1loss(ldehaze, lhazy )



    # loss = L1Loss(size_average=True)(S, V)
    return loss

