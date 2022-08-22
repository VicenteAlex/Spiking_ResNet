import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class SurrogateBPFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def poisson_gen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class SResnet(nn.Module):
    def __init__(self, n, nFilters, num_steps, leak_mem=0.95, img_size=32,  num_cls=10, poisson_gen=False):
        super(SResnet, self).__init__()

        self.n = n
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.poisson_gen = poisson_gen

        print(">>>>>>>>>>>>>>>>>>> S-ResNet >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(3, self.nFilters, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2*n):
                if block is not 0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(nn.Conv2d(self.nFilters*(2**(block + prev_nFilters)), self.nFilters*(2**block), kernel_size=3, stride=stride, padding=1, bias=bias_flag))
                self.bntt_list.append(nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*(2**block), eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)]))

        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*2, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*4, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])

        self.pool2 = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(self.nFilters*4, self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])

        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1,self.resize_bn_2])

        # Turn off bias of BNTT
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        batch_size = inp.size(0)

        mem_conv_list = [torch.zeros(batch_size, self.nFilters, self.img_size, self.img_size).cuda()]

        for block in range(3):
            for layer in range(2*self.n):
                mem_conv_list.append(torch.zeros(batch_size, self.nFilters*(2**block), self.img_size // 2**block,
                                                 self.img_size // 2**block).cuda())

        mem_fc = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):
            if self.poisson_gen:
                spike_inp = poisson_gen(inp)
                out_prev = spike_inp
            else:
                out_prev = inp

            index_1x1 = 0
            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[
                    i].threshold) - 1.0  # Positive values have surpassed the threshold
                out = self.spike_fn(mem_thr)

                if i>0 and i%2 == 0:  # Add skip conn spikes to the current output spikes
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:  # Beggining of block 2 and 3 downsize
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))  # Connections guided by 1x1 conv instead of 1 to 1 correspondance
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold  # Matrix of 0s with Th in activated cells
                mem_conv_list[i] = mem_conv_list[i] - rst  # Reset by subtraction
                out_prev = out.clone()

                if i == len(self.conv_list)-1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            #  Accumulate voltage in the last layer
            mem_fc = mem_fc + self.fc(out_prev)

        out_voltage = mem_fc / self.num_steps

        return out_voltage


class SResnetNM(nn.Module):
    def __init__(self, n, nFilters, num_steps, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SResnetNM, self).__init__()

        self.n = n
        self.img_size = int(img_size/2)
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = SurrogateBPFunction.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print(">>>>>>>>>>>>>>>>>>> S-ResNet NM >>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False
        self.nFilters = nFilters

        self.conv1 = nn.Conv2d(2, self.nFilters, kernel_size=3, stride=2, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList(
            [nn.BatchNorm2d(self.nFilters, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])

        self.conv_list = nn.ModuleList([self.conv1])
        self.bntt_list = nn.ModuleList([self.bntt1])

        for block in range(3):
            for layer in range(2*n):
                if block is not 0 and layer == 0:
                    stride = 2
                    prev_nFilters = -1
                else:
                    stride = 1
                    prev_nFilters = 0
                self.conv_list.append(nn.Conv2d(self.nFilters*(2**(block + prev_nFilters)), self.nFilters*(2**block), kernel_size=3, stride=stride, padding=1, bias=bias_flag))
                self.bntt_list.append(nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*(2**block), eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)]))

        self.conv_resize_1 = nn.Conv2d(self.nFilters, self.nFilters * 2, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_1 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*2, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])
        self.conv_resize_2 = nn.Conv2d(self.nFilters * 2, self.nFilters * 4, kernel_size=1, stride=2, padding=0,
                                       bias=bias_flag)
        self.resize_bn_2 = nn.ModuleList(
                    [nn.BatchNorm2d(self.nFilters*4, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
                     range(self.batch_num)])

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.nFilters * 4, self.num_cls, bias=bias_flag)

        self.conv1x1_list = nn.ModuleList([self.conv_resize_1, self.conv_resize_2])

        self.bn_conv1x1_list = nn.ModuleList([self.resize_bn_1, self.resize_bn_2])

        # Turn off bias of BNTT
        for bn_temp in self.bntt_list:
            bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp):

        inp = inp.permute(1, 0, 2, 3, 4)  # changes to: [T, N, 2, *, *] T=timesteps, N=batch_size

        batch_size = inp.size(1)

        mem_conv_list = [torch.zeros(batch_size, self.nFilters, self.img_size, self.img_size).cuda()]

        for block in range(3):
            for layer in range(2*self.n):
                mem_conv_list.append(torch.zeros(batch_size, self.nFilters*(2**block), self.img_size // 2**block,
                                                 self.img_size // 2**block).cuda())

        mem_fc = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(inp.size(0)):

            out_prev = inp[t,:]
            out_prev = transforms.Resize([64,64])(out_prev)

            index_1x1 = 0
            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[
                    i].threshold) - 1.0  # Positive values have surpassed the threshold
                out = self.spike_fn(mem_thr)

                if i>0 and i%2 == 0:  # Add skip conn spikes to the current output spikes
                    if i == 2 + 2 * self.n or i == 2 + 4 * self.n:  # Beggining of block 2 and 3 downsize
                        skip = self.bn_conv1x1_list[index_1x1][t](self.conv1x1_list[index_1x1](skip))  # Connections guided by 1x1 conv instead of 1 to 1 correspondance
                        index_1x1 += 1
                    out = out + skip
                    skip = out.clone()
                elif i == 0:
                    skip = out.clone()

                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold  # Matrix of 0s with Th in activated cells
                mem_conv_list[i] = mem_conv_list[i] - rst  #  Reset by subtraction
                out_prev = out.clone()

                if i == len(self.conv_list) - 1:
                    out = self.pool2(out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            #  Accumulate voltage in the last layer
            mem_fc = mem_fc + self.fc(out_prev)

        out_voltage = mem_fc / self.num_steps

        return out_voltage
