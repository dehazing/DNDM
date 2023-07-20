import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose,ToPILImage, CenterCrop, Resize  ,ToTensor, Normalize
import torchvision.transforms as transforms
import  natsort
#import  h5py
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp','.BMP'])


def calculate_valid_crop_size(crop_size):
    return crop_size




def train_h_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),

        # random_crop_and_resize(size=crop_size),


        ToTensor()

    ])

def train_s_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        #random_crop_and_resize(size=crop_size),
        ToTensor(),

    ])

def train_trans_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),

    ])

def test_transform():
    return Compose([
        #CenterCrop(crop_size),

        # random_crop_and_resize(size=crop_size),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),


        ToTensor()

    ])
'''
def mattoimage(dataFile):
    data = h5py.File(dataFile)
    print(data.keys())
    # label = h5py.File('fuzzy.h5')['lable']
    # lenght=len(hr_dataset)
    print(data)  # 字典
    print(type(data))  # 字典
    depth = data['depth']
    print(depth.shape)  # W H
    depth = np.transpose(depth, (1, 0))
    print(depth.shape)
    maxhazy = depth.max()
    #minhazy = depth.min()
    depth = (depth) / (maxhazy) * 255

    return depth
'''
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='OTS_B'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/clear' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset1(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='real'):
        self.transform = transforms.Compose(transforms_)
        #self.transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/gt' % mode) + '/*.*'))#testA  #gt_new
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/hazy' % mode) + '/*.*'))#testB


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class ImageDataset2(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='test-rrrrrr'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/h' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/t' % mode) + '/*.*'))


        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/trainA' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/trainB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/testA'#'/hazy'
        self.s_path = dataset_dir + '/testB'#'/clear'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]
        #self.h_transform = test_h_transform()
        #self.s_transform = test_s_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = Image.open(self.h_filenames[index])
        s_image = Image.open(self.s_filenames[index])
        #return ToTensor()(h_image), ToTensor()(s_image)#image_name,
        return {'A': h_image, 'B': s_image}

    def __len__(self):
        return len(self.h_filenames)


class TestDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder1, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/outdoor'+'/hazy_new'#'/hazy' indoor
        self.s_path = dataset_dir + '/outdoor'+'/gt_new'#'/clear'
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]#for p in range(10)
        #self.h_transform = test_transform()
        #self.s_transform = test_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image =  ToTensor()(Image.open(self.h_filenames[index]))
        s_image =  ToTensor()(Image.open(self.s_filenames[index]))
        return {'A': s_image, 'B': h_image}
        # return {'A': h_image, 'B': s_image}

    def __len__(self):
        return len(self.h_filenames)
# class TestDatasetFromFolder1(Dataset):
#     def __init__(self, dataset_dir):
#         super(TestDatasetFromFolder1, self).__init__()
#         #self.h_path = dataset_dir + '/h'#'/hazy'
#         self.h_path = dataset_dir + '/realworld'+'/hazy_new'#'/hazy' indoor
#         self.s_path = dataset_dir + '/realworld'+'/hazy_new'#'/clear'
#         #self.s_path = dataset_dir + '/t'#'/gt'
#         self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
#         self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]#for p in range(10)
#         #self.h_transform = test_h_transform()
#         #self.s_transform = test_s_transform()
#
#     def __getitem__(self, index):
#         image_name = self.h_filenames[index].split('/')[-1]
#         h_image = ToTensor()(Image.open(self.h_filenames[index]))
#         s_image = ToTensor()(Image.open(self.s_filenames[index]))
#         return {'A': s_image, 'B': h_image}
#         # return {'A': h_image, 'B': s_image}
#
#     def __len__(self):
#         return len(self.h_filenames)
class TestDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir):
        super(TestDatasetFromFolder2, self).__init__()
        #self.h_path = dataset_dir + '/h'#'/hazy'
        self.h_path = dataset_dir + '/hazy_new'#'/hazy'
        self.s_path = dataset_dir + '/hazy_new'#'/clear' gt_new  hazy_new
        #self.s_path = dataset_dir + '/t'#'/gt'
        self.h_filenames = [join(self.h_path, x) for x in natsort.natsorted(listdir(self.h_path)) if is_image_file(x)]
        self.s_filenames = [join(self.s_path, x) for x in natsort.natsorted(listdir(self.s_path)) if is_image_file(x)]  # for p in range(10)
        self.h_transform = test_transform()
        self.s_transform = test_transform()

    def __getitem__(self, index):
        image_name = self.h_filenames[index].split('/')[-1]
        h_image = ToTensor()(Image.open(self.h_filenames[index]))
        s_image = ToTensor()(Image.open(self.s_filenames[index]))
        return {'A':s_image, 'B':  h_image}

    def __len__(self):
        return len(self.h_filenames)



class TrainDatasetFromFolder1(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s,dataset_dir_trans, crop_size):
        super(TrainDatasetFromFolder1, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:2061] if is_image_file(x)]
        self.image_filenames_trans = [join(dataset_dir_trans, x) for x in natsort.natsorted(listdir(dataset_dir_trans))[0:2061] if is_image_file(x)]

        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:2061] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)
        self.trans_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        trans_image = self.trans_transform(Image.open(self.image_filenames_trans[index]))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        #return h_image, s_image

        return {'A': h_image, 'B': s_image,'T':trans_image}

    def __len__(self):
        return  len(self.image_filenames_h) #max(len(self.image_filenames_h), len(self.image_filenames_s))



class TrainDatasetFromFolder2(Dataset):
    def __init__(self, dataset_dir_c,dataset_dir_h,  dataset_real,crop_size):
        super(TrainDatasetFromFolder2, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_A = [join(dataset_dir_c, x) for x in natsort.natsorted(listdir(dataset_dir_c))[0:1920] for p in range(14) if is_image_file(x)]#5687    27327   2000  #is_image_file(10*x)] [0:5687]
        self.image_filenames_B = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:26880] if is_image_file(x)]  #[0:5687]
        self.image_filenames_C = [join(dataset_real, x) for x in  natsort.natsorted(listdir(dataset_real))[0:26880] if is_image_file(x)]

       # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:5000] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.c_transform = train_s_transform(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.r_transform = train_s_transform(crop_size)

    def __getitem__(self, index):
        A_image = self.c_transform(Image.open(self.image_filenames_A[index]))
        B_image = self.h_transform(Image.open(self.image_filenames_B[index]))
        R_image = self.r_transform(Image.open(self.image_filenames_C[index]))

        #return h_image, s_image

        return {'A': A_image, 'B':B_image, 'R':R_image}#'B': s_image,

    def __len__(self):
        return  len(self.image_filenames_A)#max(len(self.image_filenames_h), len(self.image_filenames_s))



class TrainDatasetFromFolder3(Dataset):
    def __init__(self, dataset_dir_h, dataset_dir_s,dataset_dir_trans, crop_size):
        super(TrainDatasetFromFolder3, self).__init__()
        # self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:350] if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:520] for p in range(35) if is_image_file(x)]
        # self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:10] for p in range(35) if is_image_file(x)]

        self.image_filenames_h = [join(dataset_dir_h, x) for x in natsort.natsorted(listdir(dataset_dir_h))[0:2061] if is_image_file(x)]
        self.image_filenames_trans = [join(dataset_dir_trans, x) for x in natsort.natsorted(listdir(dataset_dir_trans))[0:2061] if is_image_file(x)]

        self.image_filenames_s = [join(dataset_dir_s, x) for x in natsort.natsorted(listdir(dataset_dir_s))[0:2061] if is_image_file(x)] #for p  in range(10)

        crop_size = calculate_valid_crop_size(crop_size)
        self.h_transform = train_s_transform(crop_size)
        self.s_transform = train_s_transform(crop_size)
        self.trans_transform = train_s_transform(crop_size)
        #self.mattoimage= mattoimage(dataset_dir_trans)

    def __getitem__(self, index):
        h_image = self.h_transform(Image.open(self.image_filenames_h[index]))
        ##depth =h5py.File(self.image_filenames_trans[index], 'r')['depth']
        #depth = mattoimage(self.image_filenames_trans[index])
        trans_image = self.trans_transform(Image.open((self.image_filenames_trans[index])))
        s_image = self.s_transform(Image.open(self.image_filenames_s[index]))

        #return h_image, s_image

        return {'A': h_image, 'B': s_image,'T':trans_image}

    def __len__(self):
        return  len(self.image_filenames_h) #max(len(self.image_filenames_h), len(self.image_filenames_s))