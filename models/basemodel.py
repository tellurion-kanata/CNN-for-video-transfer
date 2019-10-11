import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import CustomDataLoader

import os
import numpy as np
import time
from glob import glob
from utils import *

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_file = os.path.join(opt.save_path,opt.name)
        self.ckpt_path = self.save_file + '/'
        self.sample_path = self.ckpt_path + '/sample/'
        self.test_path = self.ckpt_path + '/test/'
        self.batch_size = opt.batch_size
        self.lr = opt.learning_rate
        self.betas = (opt.beta1, opt.beta2)
        self.image_pattern = opt.image_pattern
        self.print_train_freq = opt.print_train_freq
        self.ed_epoch = opt.niter + opt.niter_decay + 1
        self.st_epoch = opt.epoch_count

    def initialize(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def forward(self):
        pass

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.opt.deterministic is True:
            torch.backends.cudnn.deterministic = True

    def setup(self):
        def mkdir(path):
            if not os.path.exists(path):
                os.mkdir(path)

        mkdir(self.save_file)
        mkdir(self.sample_path)
        mkdir(self.ckpt_path)
        mkdir(self.test_path)

        if self.opt.is_train:
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        torch.backends.cudnn.benchmark = True
        self.setup_seed(self.opt.random_seed)
        self.data_loader = CustomDataLoader()
        self.data_loader.initialize(self.opt)
        self.datasets = self.data_loader.load_data()
        self.data_size = len(self.datasets)

        self.start_time = time.time()
        self.pre_epoch_time = self.start_time
        self.pre_iter_time = self.start_time

        self.print_states()

    def eval(self):
        for net in self.models.keys():
            self.models[net].eval()

    def write_states(self, filename):
        file = open(self.ckpt_path + filename, 'a')
        file.write('**************** model states *******************\n')
        file.write('           model_name: %s\n' % self.opt.name)
        file.write('           model_type: %s\n' % self.opt.model)
        file.write('          transformer: %s\n' % self.opt.transformer)
        file.write('            loss_type: %s\n' % self.opt.loss)
        file.write('        loss_networks: {}\n'.format('vgg19' if not self.opt.vgg16 else 'vgg16'))
        if self.opt.is_train:
            file.write('       training_epoch: %d\n' % (self.ed_epoch - 1))
            file.write('  start_learning_rate: %f\n' % self.lr)
        else:
            file.write('                 eval: %s\n' % self.opt.eval)
        file.write('              dataset: %s\n' % self.opt.dataroot)
        file.write('          style_image: %s\n' % self.opt.style_image)
        file.write('         dataset_mode: %s\n' % self.opt.dataset_mode)
        file.write('              no_flip: %s\n' % self.opt.no_flip)
        file.write('            data_size: %d\n' % self.data_size)
        file.write('           batch_size: %d\n' % self.batch_size)
        file.write('           style_size: (%d, %d)\n' % (self.opt.style_size, self.opt.style_size))
        file.write('         content_size: (%d, %d)\n' % (self.opt.content_height, self.opt.content_width))
        file.write('         lambda_style: %f\n' % self.opt.lambda_style)
        file.write('       lambda_content: %f\n' % self.opt.lambda_content)
        file.write('            lambda_tv: %f\n' % self.opt.lambda_tv)
        if self.opt.model != 'style':
            file.write('        lambda_temp_f: %f\n' % self.opt.lambda_temp_f)
            file.write('        lambda_temp_o: %f\n' % self.opt.lambda_temp_o)
        file.write('           no_dropout: %s\n' % self.opt.no_dropout)
        file.write('          random_seed: %d\n' % self.opt.random_seed)
        file.write('        deterministic: %s\n' % self.opt.deterministic)
        file.write('*************************************************\n\n')
        file.close()

    def print_states(self):
        print('**************** model states *******************')
        print('           model_name: %s' % self.opt.name)
        print('           model_type: %s' % self.opt.model)
        print('          transformer: %s' % self.opt.transformer)
        print('            loss_type: %s' % self.opt.loss)
        print('        loss_networks: {}'.format('vgg19' if not self.opt.vgg16 else 'vgg16'))
        if self.opt.is_train:
            print('       training_epoch: %d' % (self.ed_epoch - 1))
            print('  start_learning_rate: %f' % self.lr)
        else:
            print('                 eval: %s' % self.opt.eval)
        print('              dataset: %s' % self.opt.dataroot)
        print('          style_image: %s' % self.opt.style_image)
        print('         dataset_mode: %s' % self.opt.dataset_mode)
        print('              no_flip: %s' % self.opt.no_flip)
        print('            data_size: %d' % self.data_size)
        print('           batch_size: %d' % self.batch_size)
        print('           style_size: (%d, %d)' % (self.opt.style_size, self.opt.style_size))
        print('         content_size: (%d, %d)' % (self.opt.content_height, self.opt.content_width))
        print('         lambda_style: %f' % self.opt.lambda_style)
        print('       lambda_content: %f' % self.opt.lambda_content)
        print('            lambda_tv: %f' % self.opt.lambda_tv)
        if self.opt.model != 'style':
            print('        lambda_temp_f: %f' % self.opt.lambda_temp_f)
            print('        lambda_temp_o: %f' % self.opt.lambda_temp_o)
        print('          random_seed: %d' % self.opt.random_seed)
        print('        deterministic: %s' % self.opt.deterministic)
        print('*************************************************')

        self.opt_log = open(self.ckpt_path+'model_opt.txt', 'w')
        self.write_states('model_opt.txt')

        if self.opt.is_train:
            self.train_log = open(self.ckpt_path + 'train_log.txt', 'w')
            self.write_states('train_log.txt')

    def save(self, epoch='latest'):
        if epoch != 'latest':
            training_state = {'epoch': epoch, 'lr': self.lr}
            torch.save(training_state, self.ckpt_path + 'model_states.pth')

        for net in self.models.keys():
            torch.save(self.models[net].state_dict(), self.ckpt_path + '{}_'.format(epoch) + net + '_params.pth')

    def load(self, epoch='latest'):
        print('\n**************** loading model ******************')

        for net in self.models.keys():
            file_path = self.ckpt_path + epoch + '_' + net + '_params.pth'
            if not os.path.exists(file_path):
                raise FileNotFoundError('%s is not found.' % file_path)
            self.models[net].load_state_dict(torch.load(file_path))

        print('\n********** load model successfully **************')

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']

    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def read_input(self, input):
        self.real_A = input['A'].cuda()
        self.real_B = input['B'].cuda()

    def set_loss_dict(self):
        pass

    # output training message iter ver.
    def print_training_iter(self, epoch, idx):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        self.pre_iter_time = current_time

        print(
            'iter_time: %4.4f s, epoch: [%d/%d], step: [%d/%d], learning_rate: %.7f'
            %
            (iter_time, epoch, self.ed_epoch-1, idx+1, self.data_size, self.lr), end=''
        )

        self.train_log = open(self.ckpt_path+'train_log.txt', 'a')
        self.train_log.write(
            'iter_time: %4.4f s, epoch: [%d/%d], step: [%d/%d], learning_rate: %.7f'
            %
            (iter_time, epoch, self.ed_epoch - 1, idx + 1, self.data_size, self.lr)
        )

        for label in self.loss_dict.keys():
            print(', %s: %.7f' % (label, self.loss_dict[label]), end='')
            self.train_log.write(', %s: %.7f' % (label, self.loss_dict[label]))
        print('')
        self.train_log.write('\n')
        self.train_log.close()

    # output training message epoch ver.
    def print_training_epoch(self, epoch):
        current_time = time.time()
        epoch_time = current_time - self.pre_epoch_time
        total_time = current_time - self.start_time
        self.pre_epoch_time = current_time

        print(
            'total time: %4.4f s, epoch_time: %4.4f s, epoch: [%d/%d], learning_rate: %.7f'
              %
            (total_time, epoch_time, epoch, self.ed_epoch-1, self.lr), end=''
        )

        self.train_log = open(self.ckpt_path+'train_log.txt', 'a')
        self.train_log.write(
            'total time: %4.4f s, epoch_time: %4.4f s, epoch: [%d/%d], learning_rate: %.7f'
              %
            (total_time, epoch_time, epoch, self.ed_epoch-1, self.lr)
        )

        for label in self.loss_dict.keys():
            print(', %s: %.7f' % (label, self.loss_dict[label]), end='')
            self.train_log.write(', %s: %.7f' % (label, self.loss_dict[label]))
        print('')
        self.train_log.write('\n')
        self.train_log.close()

# code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

