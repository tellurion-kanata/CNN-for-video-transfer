from .loss_networks import *
from .transformer import *
from .basemodel import BaseModel
import torch
import torch.optim as optim
import utils
import torchvision.transforms as transforms
import time

class videoNet(BaseModel):
    def __init__(self, opt):
        super(videoNet, self).__init__(opt)
        self.initialize()


    def initialize(self):
        self.opt.model = 'videoNet'
        self.lambda_content = self.opt.lambda_content
        self.lambda_style = self.opt.lambda_style
        self.lambda_tv = self.opt.lambda_tv

        if self.opt.loss != 'feature' and self.opt.loss != 'simple':
            raise NotImplementedError('Such loss function doesn\'t exist')

        if self.opt.transformer == 'skip':
            self.transformer = SkipTransformer(video=True).cuda()
        elif self.opt.transformer == 'simple':
            self.transformer = SimpleNet(video=True).cuda()
        elif self.opt.transformer == 'skip-simple':
            self.transformer = SimpleSkipTransformer(video=True).cuda()
        else:
            raise NotImplementedError('Such transformer model doesn\'t exist')

        self.optimizer = optim.Adam(self.transformer.parameters(),
                                    lr=self.lr, betas=self.betas)

        self.models = {'transformer': self.transformer}
        self.optimizers = [self.optimizer]

        if not self.opt.eval:
            if not self.opt.vgg16:
                self.vgg = Vgg19().cuda()
            else:
                self.vgg = Vgg16().cuda()

            self.l2_criterion = nn.MSELoss().cuda()
            self.tv_criterion = TVLoss()

            self.f_temp_criterion = TemporalLoss()
            self.o_temp_criterion = TemporalLoss()

            self.calc_style()
        self.setup()


    def calc_style(self):
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        style = utils.load_image(self.opt.style_image, size=self.opt.style_size)
        style = style_transform(style).cuda()
        style = style.repeat(self.opt.batch_size, 1, 1, 1)
        feature_style = self.vgg(utils.normalize_batch(style))
        self.gram_style = [utils.GramMatrix(y) for y in feature_style]


    def read_input(self, input):
        self.frames = input['frames'].squeeze(0)
        self.optflows = input['optflow'].squeeze(0)
        self.downflows = input['downflow'].squeeze(0)
        self.masks = input['mask'].squeeze(0)
        self.downmasks = input['downmask'].squeeze(0)


    def calc_spatial_loss(self, x, y):
        feature_x = self.vgg(utils.normalize_batch(x))
        feature_y = self.vgg(utils.normalize_batch(y))

        if self.opt.loss == 'simple':
            content_loss = self.l2_criterion(feature_y.relu4_2, feature_x.relu4_2)
        else:
            content_loss = self.l2_criterion(feature_y.relu2_2, feature_x.relu2_2)
        self.content_loss += content_loss

        style_loss = 0
        for fy, gram_s in zip(feature_y, self.gram_style):
            gram_y = utils.GramMatrix(fy)
            self.style_loss += self.l2_criterion(gram_y, gram_s[:self.batch_size, :, :])
        self.style_loss += style_loss

        self.tv_loss += self.tv_criterion(y)


    def calc_temporal_loss(self):
        if self.opt.loss == 'feature':
            self.o_temp_loss = self.opt.lambda_temp_o * self.o_temp_criterion(self.flow, self.mask, self.prev_y, self.y,
                                                                          self.prev_x, self.x)
            self.f_temp_loss = self.opt.lambda_temp_f * self.f_temp_criterion(self.downflow, self.downmask, self.prev_fy, self.fy)
            self.temporal_loss = self.f_temp_loss + self.o_temp_loss
        else:
            self.temporal_loss = self.opt.lambda_temp_o * self.o_temp_criterion(self.flow, self.mask, self.prev_y, self.y)


    def forward(self):
        self.prev_y, self.prev_fy = self.transformer(self.prev_x)
        self.y, self.fy = self.transformer(self.x)


    def backward(self):
        self.content_loss = 0
        self.style_loss = 0
        self.tv_loss = 0

        self.calc_spatial_loss(self.prev_x, self.prev_y)
        self.calc_spatial_loss(self.x, self.y)
        self.calc_temporal_loss()

        self.content_loss *= self.lambda_content
        self.style_loss   *= self.lambda_style
        self.tv_loss      *= self.lambda_tv

        self.spatial_loss = self.content_loss + self.style_loss
        self.loss = self.spatial_loss + self.temporal_loss
        self.loss.backward()


    def generate_first_frame(self):
        _, c, h, w = self.frames.shape
        self.x = self.frames[0].cuda().expand(self.batch_size, 3, self.opt.content_height, self.opt.content_width)
        self.y, self.fy = self.transformer(self.x)

        self.optimizer.zero_grad()

        self.style_loss = 0
        self.content_loss = 0
        self.tv_loss = 0

        self.calc_spatial_loss(self.x, self.y)
        self.style_loss   *= self.lambda_style
        self.content_loss *= self.lambda_content
        self.tv_loss      *= self.lambda_tv

        self.spatial_loss = self.style_loss + self.content_loss
        self.spatial_loss.backward()

        self.optimizer.step()

        self.prev_x = self.x


    def train(self):
        step = 0
        if self.opt.load_model:
            self.load(self.opt.load_epoch)

        for epoch in range(self.st_epoch, self.ed_epoch):
            for idx, data in enumerate(self.datasets):
                self.read_input(data)
                _, c, h, w = self.frames.shape
                self.generate_first_frame()
                step += 1

                for i in range(1, len(self.frames)):
                    self.x = self.frames[i].cuda().upsqueeze(0)
                    self.flow = self.optflows[i-1].cuda().unsqueeze(0)
                    self.downflow = self.downflows[i-1].cuda().unsqueeze(0)
                    self.mask = self.masks[i-1].cuda().unsqueeze(0)
                    self.downmask = self.downmasks[i-1].cuda().unsqueeze(0)

                    self.optimizer.zero_grad()
                    self.forward()
                    self.backward()
                    self.optimizer.step()
                    step += 1

                    self.prev_x = self.x

                    if step % self.opt.print_state_freq == 0:
                        self.set_loss_dict()
                        self.print_training_iter(epoch, idx)

                self.save_video(epoch)
                self.save()
                self.set_loss_dict()
                self.print_training_iter(epoch, idx)

            self.print_training_epoch(epoch)
            self.update_learning_rate()
            self.save(epoch)
        self.save()


    def set_loss_dict(self):
        if self.opt.loss == 'feature':
            self.loss_dict = {
                'style_loss': self.style_loss,
                'content_loss': self.content_loss,
                'tv_loss': self.tv_loss,
                'temporal_loss_f': self.f_temp_loss,
                'temporal_loss_o': self.o_temp_loss
            }
        else:
            self.loss_dict = {
                'style_loss': self.style_loss,
                'content_loss': self.content_loss,
                'tv_loss': self.tv_loss,
                'temporal_loss_o': self.o_temp_loss
            }


    def save_video(self, epoch):
        with torch.no_grad():
            self.transformer.eval()
            b, c, h, w = self.frames.shape
            for i in range(100):
                self.x = self.frames[i].cuda().unsqueeze(0)
                self.y, _ = self.transformer(self.x)
                utils.save_image(self.sample_path+str(epoch)+'_'+str(i)+'.jpg', self.y.cpu().squeeze(0))

            self.transformer.train()


    def test(self, epoch='latest'):
        with torch.no_grad():
            self.transformer.eval()
            state_dict = torch.load(self.ckpt_path + epoch + '_transformer_params.pth')
            self.transformer.load_state_dict(state_dict)
            data_size = len(self.datasets)

            for idx, data in enumerate(self.datasets):
                self.x = data['A'].cuda().expand(1, 3, self.opt.content_height, self.opt.content_width)
                name = data['A_idx'][0]
                self.y, _ = self.transformer(self.x)
                utils.save_image(self.test_path + name, self.y.cpu().squeeze(0))
                print('test process: [%d / %d] ...' % (idx+1, data_size))
