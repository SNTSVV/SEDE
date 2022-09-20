"""
Copyright (C) 2018-2022 IEE S.A. (https://iee-sensing.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""
created by: jun wang @ iee
"""
"""
Modified by: Hazem Fahmy - hazem.fahmy@uni.lu - Added function relprob() - Modified class KPNet()
the model size is around 2M
TODO: quantize and prune toe model

"""
import torch
from torch import nn, optim
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, padding_mode='zeros', output_padding=0):
    #    super(nn.Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
    # False, groups, bias, padding_mode, output_padding)
    #       self.padding_mode='zeros'
    #      self.output_padding = 0
    #     self.weight = conv2d.weight
    #    self.bias = conv2d.bias
        def gradprop(self, DY):
                output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
                return F.conv_transpose2d(DY, self.weight, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)
    #def forward(self, x):
    #    x = self.Y
    #    return x
        def test(self):
                print("test")
                return
        def relprop(self, R):
                Z= self.Y+1e-9
                S=R/Z
                C = self.gradprop(S)
                R = self.X*C
                self.HM=R
                return R

class ConvTranspose2d(nn.ConvTranspose2d):
    # def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, padding_mode='zeros', output_padding=0):
    #    super(nn.Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
    # False, groups, bias, padding_mode, output_padding)
    #       self.padding_mode='zeros'
    #      self.output_padding = 0
    #     self.weight = conv2d.weight
    #    self.bias = conv2d.bias
    def gradprop(self, DY):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv2d(DY, self.weight, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)
    #def forward(self, x):
    #    x = self.Y
    #    return x

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        self.HM = R
        return R


class KPNet(nn.Module):
        def __init__(self):
                super(KPNet, self).__init__()
                in_channel = 1
                out_conv2d_1 = 64
                out_conv2d_2 = 64
                out_conv2d_3 = 128
                out_conv2d_4 = 128

                out_conv2d_5 = 32*5
                out_conv2d_6 = 32*5
                out_conv2d_trans_1 = 128
                n_class = 36 # tthe number of coordinates
                self.conv2d_1 = Conv2d(in_channels=in_channel, out_channels=out_conv2d_1, kernel_size=3, padding=1)
                self.conv2d_2 = Conv2d(in_channels=out_conv2d_1, out_channels=out_conv2d_2, kernel_size=3, padding=1)
                self.conv2d_3 = Conv2d(in_channels=out_conv2d_2, out_channels=out_conv2d_3, kernel_size=3, padding=1)
                self.conv2d_4 = Conv2d(in_channels=out_conv2d_3, out_channels=out_conv2d_4, kernel_size=3, padding=1)

                self.conv2d_5 = Conv2d(in_channels=out_conv2d_4, out_channels=out_conv2d_5, kernel_size=3, padding=1)
                self.conv2d_6 = Conv2d(in_channels=out_conv2d_5, out_channels=out_conv2d_6, kernel_size=1, padding=0)

                self.conv2d_trans_1 = ConvTranspose2d(in_channels=out_conv2d_6, out_channels=out_conv2d_trans_1, kernel_size=2, stride=2,bias=False)
                self.conv2d_trans_2 = ConvTranspose2d(in_channels=out_conv2d_trans_1, out_channels=n_class, kernel_size=2, stride=2,bias=False)
                return

        def forward(self,x):
                x = F.relu(self.layers[0](x))
                x = F.relu(self.conv2d_2(x))
                x = F.max_pool2d(x,kernel_size=2, stride=2)
                x = F.relu(self.conv2d_3(x))
                x = F.relu(self.conv2d_4(x))
                x = F.max_pool2d(x,kernel_size=2, stride=2)
                x = F.relu(self.conv2d_5(x))
                x = F.relu(self.conv2d_6(x))
                x = F.relu(self.conv2d_trans_1(x))
                x = self.conv2d_trans_2(x)
                #print(x.size())
                return x
        def relprop(self, R):
                R = self.conv2d_trans_2.relprop(R)
                R = self.conv2d_trans_1.relprop(R)
                R = self.conv2d_6.relprop(R)
                R = self.conv2d_5.relprop(R)
                R = self.conv2d_4.relprop(R)
                R = self.conv2d_3.relprop(R)
                R = self.conv2d_2.relprop(R)
                R = self.conv2d_1.relprop(R)
                return R
if __name__ == '__main__':
	model = KPNet().cuda()
	print(model)
	summary(model, (1,96,96))


