from .networks import *

class SimpleNet(torch.nn.Module):
    def __init__(self, video=False):
        super(SimpleNet, self).__init__()
        self.ret_feature = video
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 48, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(48, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(48)
        self.res2 = ResidualBlock(48)
        self.res3 = ResidualBlock(48)
        self.res4 = ResidualBlock(48)
        self.res5 = ResidualBlock(48)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(48, 32, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv3 = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.tanh = torch.nn.Tanh()
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        if self.ret_feature:
            fy = y

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        y = self.tanh(y/255.) * 150. + 127.5

        if self.ret_feature:
            return y, fy
        return y


class SkipTransformer(torch.nn.Module):
    def __init__(self, video=False):
        super(SkipTransformer, self).__init__()
        self.ret_feature = video
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 48, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(48, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(48)
        self.res2 = ResidualBlock(48)
        self.res3 = ResidualBlock(48)
        self.res4 = ResidualBlock(48)
        self.res5 = ResidualBlock(48)
        # Upsampling Layers
        self.conv4 = ConvLayer(96, 48, kernel_size=3, stride=1)
        self.in4 = torch.nn.InstanceNorm2d(48, affine=True)
        self.deconv1 = UpsampleConvLayer(48, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv5 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.in6 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = UpsampleConvLayer(32, 16, kernel_size=3, stride=1, upsample=2)
        self.in7 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv3 = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.tanh = torch.nn.Tanh()
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y2 = y
        y = self.relu(self.in3(self.conv3(y)))
        y3 = y
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        if self.ret_feature:
            ry = y

        # skip connection 1
        y = torch.cat((y, y3), 1)
        y = self.relu(self.in4(self.conv4(y)))
        y = self.relu(self.in5(self.deconv1(y)))

        # skip connection 2
        y = torch.cat((y, y2), 1)
        y = self.relu(self.in6(self.conv5(y)))
        y = self.relu(self.in7(self.deconv2(y)))
        y = self.deconv3(y)
        y = self.tanh(y / 255.) * 150. + 127.5

        if self.ret_feature:
            return y, ry
        return y


class SimpleSkipTransformer(torch.nn.Module):
    def __init__(self, video=False):
        super(SimpleSkipTransformer, self).__init__()
        self.ret_feature = video
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 16, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(16, affine=True)
        self.conv2 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 48, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(48, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(48)
        self.res2 = ResidualBlock(48)
        self.res3 = ResidualBlock(48)
        self.res4 = ResidualBlock(48)
        self.res5 = ResidualBlock(48)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(96, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 16, kernel_size=3, stride=1, upsample=2)
        self.in7 = torch.nn.InstanceNorm2d(16, affine=True)
        self.deconv3 = ConvLayer(16, 3, kernel_size=3, stride=1)
        self.tanh = torch.nn.Tanh()
        # Non-linearities
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.LeakyReLU(0.2, True)

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y2 = y
        y = self.relu(self.in3(self.conv3(y)))
        y3 = y
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        if self.ret_feature:
            ry = y

        # skip connection 1
        y = torch.cat((y, y3), 1)
        y = self.relu(self.in5(self.deconv1(y)))

        # skip connection 2
        y = torch.cat((y, y2), 1)
        y = self.relu(self.in7(self.deconv2(y)))
        y = self.deconv3(y)
        y = self.tanh(y/255.) * 150. + 127.5

        if self.ret_feature:
            return y, ry
        return y
     