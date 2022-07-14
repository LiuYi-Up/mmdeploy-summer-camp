### 本周目标
>1.环境配置  
    2.编译ncnn  
    3.使用 tool/quantize 工具量化 squeezenet_v1.1 模型  
    4.实现 naive Conv2d 代码  
    
### 1.环境配置 + 2.编译ncnn
环境说明：  
>Ubuntu 18.04  
NVIDIA GeForce RTX 2080 Ti  

参考 ncnn    [官方文档](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux) 即可（官方教程很详细）。
### 3.使用 tool/quantize 工具量化 squeezenet_v1.1 模型
主要步骤为 [参考官方文档](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md) ：  
>a.编译ncnn  
>b.优化模型  
>c.生成校准表文件  
>d.量化模型  

#### a.编译ncnn  
在实践过程中发现，使用 vulkan 加速和不使用 vulkan 加速对模型推理的结果有影响（ps:原因不详，对vulkan没有好好学习了解过&#x1F633;）  
先来看看两种情况下 squeezenet_v1.1 对 [测试图片](https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png)  的推理结果：  
<img alt="test.png" src="https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png" width="360" height="640">   
将 [测试图片](https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png)  下载到  `${NCNN_DIR}/images/sceenshot.png`   
- 使用 vulkan  
```
cd ${NCNN_DIR}/build20220713  
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..  
make -j$(nproc)  
ln -sf ../examples/squeezenet_v1.1.param squeezenet_v1.1.param  
ln -sf ../examples/squeezenet_v1.1.bin squeezenet_v1.1.bin
./examples/squeezenet ../images/screenshot.png
```
结果为：  
![w](https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/wovulkan.png)  
查看标签使用上图等号前的数字 `(128/143/98)+1` 找出 [GT表格](https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt) 对应行号的结果，此时 `128+1` 对应的标签为 `black stork, Ciconia nigra` 显然出了问题。
- 不使用 vulkan  
```
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..  
make -j$(nproc)  
./examples/squeezenet ../images/screenshot.png  
```
结果为：  
![wo](https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/wvulkan.png)  
`281+1` 对应的标签为 `tabby, tabby cat` ，结果正确。因此之后的操作咱都禁止 vulkan。  
#### b.优化模型  
好，咱接着来，优化模型这一步在我的实验过程中似乎对网络的影响是负面的，咱先不管，跟着官方文档走一遍 &#x1F910; ：
```
./tools/ncnnoptimize ./squeezenet_v1.1.param ./squeezenet_v1.1.bin sqznet-opt.param sqznet-opt.bin 0  
```
#### c.生成校准表文件  
首先，下载 [校准数据集](https://github.com/EliSchwartz/imagenet-sample-images) 到 `${NCNN_DIR}/images/` 并且生成路径文件，然后生成校准表：
```
find ../images/imagenet-sample-images-master/ -type f > ../images/imagelist.txt
./tools/quantize/ncnn2table sqznet-opt.param sqznet-opt.bin ../images/imagelist.txt sqznet.table mean=[104,117,123] norm=[1,1,1] shape=[227,227,3] pixel=BGR thread=1 method=kl  
```
这里需要注意 `mean=, norm=, shape=, thread=, method=` 这些参数，对应不同的模型有不同的参数（note:我在敲这行代码的时候，习惯性的在参数[104,117,123]这些逗号后跟上一个空格，这样是会报错滴！ &#x1F605; ）。
#### d.量化模型  
来到关键步骤，此时可以先删除 `a.编译ncnn` 步骤中生成的两个软连接，然后量化模型生成新的模型参数文件，最后测试结果：  
```
rm squeezenet_v1.1.param  
rm squeezenet_v1.1.bin  
./tools/quantize/ncnn2int8 sqznet-opt.param sqznet-opt.bin squeezenet_v1.1.param  squeezenet_v1.1.bin sqznet.table  
./examples/squeezenet ../images/screenshot.png    
```  
结果如下：  
<img alt='opt' src='https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/test1.png'>  
发现结果为 `921+1` 对应 `book jacket, dust cover, dust jacket, dust wrapper` 这结果都飞到宇宙了，不知道是什么原因 &#x1F914; 。那换个思路，在生成校准表的时候使用的 `method=kl` ，索性换一个方法 `method=aciq` 试试：  
```
./tools/quantize/ncnn2table sqznet-opt.param sqznet-opt.bin ../images/imagelist.txt sqznet.table mean=[104,117,123] norm=[1,1,1] shape=[227,227,3] pixel=BGR thread=1 method=aciq  
./tools/quantize/ncnn2int8 sqznet-opt.param sqznet-opt.bin squeezenet_v1.1.param  squeezenet_v1.1.bin sqznet.table  
./examples/squeezenet ../images/screenshot.png    
```
看看结果，欸，至少分类结果正确啦，但是精度下降的有点厉害：  
<img alt='2' src='https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/test2.png'>  
再试试不经过模型优化，直接生成校准表呢：  
```
./tools/quantize/ncnn2table ../examples/squeezenet_v1.1.param ../examples/squeezenet_v1.1.bin ../images/imagelist.txt sqznet.table mean=[104,117,123] norm=[1,1,1] shape=[227,227,3] pixel=BGR thread=1 method=kl  
./tools/quantize/ncnn2int8 ../examples/squeezenet_v1.1.param ../examples/squeezenet_v1.1.bin squeezenet_v1.1.param  squeezenet_v1.1.bin sqznet.table  
./examples/squeezenet ../images/screenshot.png    
```  
看看结果，好像好多了：  
<img alt='3' src='https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/te3.png'>  
实际上，每次同样操作的结果都不太相同。  
### 4.实现 naive Conv2d 代码
对于二维卷积，也是通过这次学习才知道，通常在神经网络中说的 Conv2d 是 `互相关` 操作，它与数学意义上的 `卷积` 区别主要在于，卷积核与原始图片相乘求和之前， `卷积` 操作是需要对卷积核进行顺时针180°的旋转（等价于上下翻转一次，再左右翻转一次）。而在神经网络中通常省略 `翻转` 操作是因为，神经网络的卷积核参数本身就是 `trainable` 的，网络是通过训练学习卷积核参数，因此翻不翻转是非必要的，通常就省略了。  
所以通过python实现了单通道二维卷积和多通道二维卷积，[代码](https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/naiveConv2D.py)如下：  
- 单通道 Conv2d  
```
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
```  
- 多通道 Conv2d  
```
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
```  
分别与python的 `signal.convolve2d()` `torch.nn.functional.con2d()` 两个库函数对比结果：  
```
# 单通道二维卷积测试
input = np.random.rand(5,5)
kernel = np.array([[0, 1, 0, 1], [0, 2, 0, 1], [1, 0, -1, 1], [1, 1, 1, 1]])
out = naiveConv2d(input, kernel, stride=1)
grad = signal.convolve2d(input, kernel, boundary='fill', mode='valid',)
print(np.around(out, 4) == np.around(grad, 4))

# 多通道二维卷积测试
input = np.random.rand(2, 3, 5, 5)
kernel = np.random.rand(4, 3, 3, 3)
sel = Conv2d(input, kernel)
lab = F.conv2d(torch.tensor(input), torch.tensor(kernel))
print(np.around(sel, 4) == np.around(lab.numpy(), 4))
```  
结果正确。
