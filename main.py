from models.videonet import *
from models.fast_style_neural import *
from options import get_options
import os


def get_model(opt):
    if opt.model == 'videonet':
        return videoNet(opt)
    elif opt.model == 'style':
        return styleNet(opt)
    else:
        raise NotImplementedError('Such model doesn\'t exist.')


def train(opt):
    opt.is_train = True

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    model = get_model(opt)
    model.train()

def test(opt):
    opt.is_train = False

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    model = get_model(opt)
    model.test(opt.load_epoch)

if __name__ == '__main__':
    opt = get_options()

    if not opt.eval:
        train(opt)
    else:
        test(opt)
