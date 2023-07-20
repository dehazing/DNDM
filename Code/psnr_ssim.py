import os
import numpy as np
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
import natsort
val_ite = 0
psnr = 0
ssim = 0



ref_dir = './output/A'
in_dir = './output/C'#/home/lyp/PycharmProjects/LIR-for-Unsupervised-IR/Val_Indoor'  #/home/lyp/3T/RESIDE/SOTS/nyuhaze500/hazy'

ref_filenames = [os.path.join(ref_dir, x) for x in natsort.natsorted(os.listdir(ref_dir)) ]#for q in range(10)
in_filenames = [os.path.join(in_dir, x) for x in natsort.natsorted(os.listdir(in_dir)) ]

# in_filenames = ref_filenames

imglist = zip(in_filenames, ref_filenames)
for i, (path_in, path_ref) in enumerate(imglist):
    print(path_in, path_ref)
    ref = Image.open(path_ref).convert('RGB')
    # print(ref.type)

    ref = np.array(ref) / 255.
    # print(ref.shape)
    #ref = ref[10:-10, 10:-10, :]
    # ref = ref[0:10, 0:-10, :]
    # print(ref)
    inputs = Image.open(path_in).convert('RGB')

    inputs = np.array(inputs) / 255.
    # print(inputs.shape)

    psnr += compare_psnr(ref, inputs )#,data_range=1
    ssim += compare_ssim(ref, inputs, multichannel=True)
    # plt.figure('ref')
    # plt.imshow(ref, interpolation='nearest')
    # plt.figure('out')
    # plt.imshow(outputs, interpolation='nearest')
    # plt.figure('in')-10:10
    # plt.imshow(noi, interpolation='nearest')
    # plt.show()
    val_ite += 1
psnr /= val_ite
ssim /= val_ite
print(val_ite)
print('psnr:{}, ssim:{}'.format(psnr, ssim))