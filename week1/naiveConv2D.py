import numpy as np
from scipy import signal
import torch
from torch.nn import functional as F

def naiveConv2d(input, kernel, bias=0, stride=1, padding=[0, 0, 0, 0], flip=True):
    '''
    param input: 输入图像，二维
    param kernel: 卷积核，二维
    param bias: 卷积后的偏执
    param stride: 卷积步长
    param padding: 上\下\左\右膨胀宽度
    param flip: 布尔类型, True为卷积, False为互相关
    '''
    input_h, input_w  = input.shape
    kernel_h, kernel_w = kernel.shape

    out_w = int((input_w + padding[0] + padding[1] - kernel_w) // stride) + 1
    out_h = int((input_h + padding[2] + padding[3] - kernel_h) // stride) + 1
    output = np.zeros((out_w, out_h))

    if padding:
        input = np.pad(input, ((padding[0], padding[1]), (padding[2], padding[3])))
        input_h, input_w  = input.shape
    if flip:
        kernel = np.fliplr(np.flipud(kernel))
    
    for i in range(0, input_h-kernel_h+1, stride):
        for j in range(0, input_w-kernel_w+1, stride):
            output[i // stride, j // stride] = np.sum(input[i:i+kernel_h, j:j+kernel_w] * kernel) + bias

    return output

def Conv2d(input, kernel, bias=None, stride=1, padding=None):
    '''
    param input: 输入图像, shape: B * C * H * w
    param kernel: 卷积核, shape: output_channel * input_channel * H * w
    param bias: 卷积后的偏执, shape: output_channel
    param stride: 卷积步长
    param padding: 上\下\左\右膨胀宽度
    '''

    input_b, input_c, input_h, input_w = input.shape
    out_c, _, kernel_h, kernel_w = kernel.shape

    if padding is None:
        padding = np.zeros(4, dtype=np.int)
    if bias is None:
        bias = np.zeros(out_c)

    # 计算输出尺寸
    out_w = int((input_w + padding[0] + padding[1] - kernel_w) // stride) + 1
    out_h = int((input_h + padding[2] + padding[3] - kernel_h) // stride) + 1
    output = np.zeros((input_b, out_c, out_w, out_h))

    # padding，只在图片的宽和高两个维度上padding
    input = np.pad(input, ((0, 0), (0, 0), (padding[0], padding[1]), (padding[2], padding[3])))
    _, _, input_h, input_w  = input.shape

    for b in range(input_b):
        for oc in range(out_c):
            for i in range(0, input_h-kernel_h+1, stride):
                for j in range(0, input_w-kernel_w+1, stride):
                    output[b, oc, i // stride, j // stride] += \
                        np.sum(input[b, :, i:i+kernel_h, j:j+kernel_w] * kernel[oc, :, :, :]) + bias[oc]
    return output

if __name__ == '__main__':
    # 单通道二维卷积测试
    # input = np.random.rand(5,5)
    # kernel = np.array([[0, 1, 0, 1], [0, 2, 0, 1], [1, 0, -1, 1], [1, 1, 1, 1]])
    # out = naiveConv2d(input, kernel, stride=1)
    # print(out)
    # grad = signal.convolve2d(input, kernel, boundary='fill', mode='valid',)
    # print(grad)

    # 多通道二维卷积测试
    input = np.random.rand(2, 3, 5, 5)
    kernel = np.random.rand(4, 3, 3, 3)
    sel = Conv2d(input, kernel)
    lab = F.conv2d(torch.tensor(input), torch.tensor(kernel))
    print(np.around(sel, 4) == np.around(lab.numpy(), 4))
    