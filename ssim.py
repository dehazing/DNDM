from pytorch_msssim import MS_SSIM, SSIM

class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return - super(MS_SSIM_Loss, self).forward(img1, img2)

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return - super(SSIM_Loss, self).forward(img1, img2)