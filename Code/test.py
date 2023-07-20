#!/usr/bin/python3

import argparse
from torch.utils.data import DataLoader
from datasets import TestDatasetFromFolder2
from skimage.metrics import structural_similarity as ski_ssim
from GFN20 import *
import torchvision.utils as vutils
# from model11242 import  *
from FFA02 import ffa
import os
from utils21 import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/omnisky/4t/JTY/testdataset', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', default='Ture',help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_content=Net_content()
netG_haze = Net_hazy()
net_dehaze  = ffa(3,5)
net_G= Net_G()


netG_content .cuda()
netG_haze .cuda()
net_dehaze.cuda()
net_G.cuda()

# Load state dicts
netG_content.load_state_dict(torch.load('./output/netG_content_5.pth'))#unsupervised
netG_haze.load_state_dict(torch.load('./output/netG_haze_5.pth'))
net_dehaze.load_state_dict(torch.load('./output/net_dehaze_5.pth'))
print("net_dehaze have {} parameters in total".format(sum(x.numel() for x in net_dehaze.parameters())))
print("netG_haze have {} parameters in total".format(sum(x.numel() for x in netG_haze.parameters())))
print("netG_content have {} parameters in total".format(sum(x.numel() for x in netG_content.parameters())))


net_dehaze.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

dataloader = DataLoader(TestDatasetFromFolder2('/home/omnisky/4t/JTY/testdataset/test_new'))#real test-rrrrrr  mytest_A  test-rrrrrr  /home/omnisky/4t/JTY/testdataset/test_new

###################################

###### Testing######
if not os.path.exists('./output/A'):
    os.makedirs('./output/A')
if not os.path.exists('./output/B'):
    os.makedirs('./output/B')
if not os.path.exists('output/C'):
    os.makedirs('./output/C')

test_ite = 0
test_psnr = 0
test_ssim = 0
eps = 1e-10
tt = 0
with torch.no_grad():
 for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(batch['A']).cuda()
    real_B = Variable(batch['B']).cuda()
    # content_B = netG_content(real_B)
    #
    # hazy_mask_B= netG_haze(real_B)
    #
    # # meta_B = cat([con_B, mask_B],1)
    #
    # dehaze_B = net_dehaze(real_B,con_B, mask_B)
    t0 = time.time()
    # visuals = model.get_current_visuals()
    content_B, con_B = netG_content(real_B)
    hazy_mask_B, mask_B = netG_haze(real_B)
    meta_B = cat([con_B, mask_B], 1)
    dehaze_B = net_dehaze(real_B, meta_B)

    t1 = time.time()
    tt = tt + t1 - t0
    print('time:',str((t1 - t0)))




    #dehaze_B = net_dehaze(real_B,content_B)
    output = dehaze_B
    #print(dehaze_B)
    #print(real_A)
    hr_patch = (real_A)

    #content_dR = netG_content(dehaze_R)

    #haze_mask_A = netG_haze(hazy_A)

    #dehaze_A = net_dehaze(hazy_A)

   # fake_haze_R = net_G(content_dR * haze_mask_A)

    #trans_real_A = netG_H_A2B(real_A)
    #recovered_haze = real_B * trans_real_A + a * (1 - trans_real_A)

    #dehaze = (recovered_haze - a * (1 - trans_real_A)) / trans_real_A

    # Generate output
    #fake_B = netG_A2B(real_A).data
    #fake_A = netG_B2A(real_B).data
    #fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    #fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    #save_image(real_A, 'output/A/%04d.png' % (i+1))
    #save_image(dehaze_R, 'output/B/%04d.png' % (i+1))
    #save_image(hazy_R , 'output/C/%04d.png' % (i + 1))
    vutils.save_image(real_A.data, './output/A/%04d.png' % (int(i)), padding=0, normalize=True)  # False
    vutils.save_image(real_B.data, './output/B/%04d.png' % (int(i)), padding=0, normalize=True)
    vutils.save_image(dehaze_B.data, './output/C/%04d.png' % (int(i)), padding=0, normalize=True)#True
    #print(dehaze_B)
    #print(real_A)


    output = output.data.cpu().numpy()[0]
    output[output >1] = 1
    output[output < 0] = 0
    output = output.transpose((1, 2, 0))
    #print(output)
    hr_patch = real_A.data.cpu().numpy()[0]
    hr_patch[hr_patch > 1] = 1
    hr_patch[hr_patch < 0] = 0
    hr_patch = hr_patch.transpose((1, 2, 0))
    #print(hr_patch)
    # SSIM
    ssim = ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    test_ssim += ssim  # ski_ssim(output, hr_patch, data_range=1, multichannel=True)
    # PSNR
    imdf = (output - hr_patch) ** 2
    mse = np.mean(imdf) + eps
    psnr = 10 * math.log10(1.0 / mse)
    #psnr = compare_psnr(output,hr_patch)

    test_psnr += psnr  # 10 * math.log10(1.0/mse)
    test_ite += 1
    print('PSNR: {:.4f}'.format(psnr))
    print('SSIM: {:.4f}'.format(ssim))
    print('m_PSNR: {:.4f}'.format(test_psnr / test_ite))
    print('m_SSIM: {:.4f}'.format(test_ssim / test_ite))



    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
 print('mtt',str(tt/500))
 print('m_PSNR: {:.4f}'.format(test_psnr/test_ite))
 print('m_SSIM: {:.4f}'.format(test_ssim/test_ite))
 sys.stdout.write('\n')
###################################

