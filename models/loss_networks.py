from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models as models
from flow_utils import warp


def vggNet(vgg19=False):
    if not vgg19:
        model = models.vgg16()
        states_dict = torch.load('../vgg_model/vgg16-397923af.pth')
    else:
        model = models.vgg19()
        states_dict = torch.load('../vgg_model/vgg19-dcbb9e9d.pth')
    model.load_state_dict(states_dict)
    return model


# Code from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vggNet().features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vggNet(vgg19=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4)
        return out



class TVLoss(nn.Module):
    def __init__(self, gpu=True):
        super(TVLoss, self).__init__()
        if gpu:
            self.loss = nn.MSELoss().cuda()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x):
        b, c, h, w = x.shape
        dy = self.loss(x[:,:,1:,:], x[:,:,:h-1,:])
        dx = self.loss(x[:,:,:,1:], x[:,:,:,:w-1])
        return 2 * (dy + dx)



# Based on paper ReCoNet: Real-time Coherent Video Style Transfer Network

class TemporalLoss(nn.Module):
    def __init__(self, gpu=True):
        super(TemporalLoss, self).__init__()
        if gpu:
            self.loss = nn.MSELoss(reduction='none').cuda()
        else:
            self.loss = nn.MSELoss(reduction='none')

    def forward(self, flow, mask, prev_output, next_output, prev_input=None, next_input=None):
        mask[mask > 0] = 1.
        warped_output = warp(prev_output, flow)

        if prev_input is None:
            return torch.mean(mask * self.loss(next_output, warped_output))

        else:
            warped_input = warp(prev_input, flow)
            diff_output = next_output - warped_output
            diff_input = 0.2126 * (next_input[:, 0, :, :] - warped_input[:, 0, :, :]) + \
                         0.7152 * (next_input[:, 1, :, :] - warped_input[:, 1, :, :]) + \
                         0.0722 * (next_input[:, 2, :, :] - warped_input[:, 2, :, :])

            loss = torch.sum(mask * self.loss(diff_output, diff_input.expand(diff_output.shape))) / (prev_output.shape[2] * prev_output.shape[3])

            return loss
