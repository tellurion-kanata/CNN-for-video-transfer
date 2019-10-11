import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import cv2
import PIL.Image as Image
from glob import glob
from flow_utils import readFlow


def transform(opt):
    transform_list = []
    if opt.transform == 'crop':
        transform_list.append(transforms.RandomCrop(opt.content_size))
    elif opt.transform == 'none':
        pass
    else:
        raise ValueError('{} transform haven\'t been finished yet.'.format(opt.transform))
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

class VideoDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.file_path = self.opt.dataroot
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('data file is not found.')
        foldername = os.listdir(opt.dataroot)
        self.folders = [os.path.join(opt.dataroot, folder) for folder in foldername]

    def __getitem__(self, index):
        folder = self.folders[index]
        frame_files = glob(os.path.join(folder, 'frame/'+self.opt.image_pattern))
        frame_count = len(frame_files)

        frames = torch.zeros([frame_count, 3, self.opt.content_height, self.opt.content_width])
        optflow = torch.zeros([frame_count-1, 2, self.opt.content_height, self.opt.content_width])
        downflow = torch.zeros([frame_count-1, 2, self.opt.content_height//4, self.opt.content_width//4])
        mask = torch.zeros([frame_count-1, 1, self.opt.content_height, self.opt.content_width])
        downmask = torch.zeros([frame_count-1, 1, self.opt.content_height//4, self.opt.content_width//4])

        for i in range(frame_count):
            imageName = os.path.join(folder, 'frame/'+str(i)+'.jpg')
            # img = Image.open(imageName).convert('L')
            img = Image.open(imageName)
            img = transforms.ToTensor()(img)
            img = transforms.Lambda(lambda x: x.mul(255))(img)
            frames[i] = img

        for i in range(frame_count - 1):
            # flow: (h, w, c) -> (c, h, w)
            flowName = os.path.join(folder, 'flow/'+str(i)+'.flo')
            flow = torch.Tensor(readFlow(flowName)).permute(2, 0, 1)
            optflow[i] = flow
            downflow[i] = nn.functional.interpolate(flow.unsqueeze(0), scale_factor=(0.25, 0.25),
                                                    mode='bilinear', align_corners=True).squeeze(0)

        for i in range(frame_count - 1):
            maskName = os.path.join(folder, 'mask/'+str(i))
            cmask = torch.Tensor(cv2.imread(maskName, cv2.IMREAD_GRAYSCALE) / 255.).unsqueeze(0)
            mask[i] = cmask
            downmask[i] = nn.functional.interpolate(cmask.unsqueeze(0), scale_factor=(0.25, 0.25),
                                                    mode='bilinear', align_corners=True).squeeze(0)


        return {'frames': frames,
                'optflow': optflow,
                'downflow': downflow,
                'mask': mask,
                'downmask': downmask,
                }

    def __len__(self):
        return len(self.folders)


class ImageDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.file_path = opt.dataroot
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('data file is not found.')
        self.files = glob(os.path.join(opt.dataroot, opt.image_pattern))

    def __getitem__(self, index):
        file = self.files[index]
        # input = Image.open(file).convert('L')
        input = Image.open(file)
        input = transforms.ToTensor()(input)
        input = transforms.Lambda(lambda x: x.mul(255))(input)

        return {
            'A': input,
            'A_idx': file.replace(self.file_path+'\\', '')
        }


    def __len__(self):
        return len(self.files)


class CustomDataLoader():
    def initialize(self, opt):
        if opt.dataset_mode == 'video' and not opt.eval:
            self.dataset = VideoDataset(opt)
        elif opt.dataset_mode == 'image' or opt.eval:
            self.dataset = ImageDataset(opt)
        else:
            raise NotImplementedError('Not a valid data mode.')

        self.dataloader = data.DataLoader(
            dataset = self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.no_shuffle and not opt.eval,
            num_workers = opt.num_threads)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataloader:
            yield data