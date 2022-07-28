### 本周目标  
>1.理解非对称量化  
2.理解对称量化  
3.对比对称量化 & 非对称量化  
4.理解对称量化的 conv 计算过程  

### 1.理解非对称量化  
该部分通过阅读论文 [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/pdf/1712.05877.pdf) 来理解非对称量化。  
论文的动机就不过多介绍了，量化的目的主要就是为了尽量降低精度损失的同时，将模型的权重或激活（即特征图）从浮点型转换为整型，使模型更有效地部署在边缘设备上以更小的计算资源、更快的速度进行推理。
#### 0x01 介绍
文章的主要贡献包括： 
 
    - 提出一种量化方法，将模型的权重和激活量化到8位整型且只有偏置等少量参数量化到32位整型。  
    - 提出一种能够在只支持整型运算的硬件设备上有效部署的量化推理框架。  
    - 提出一种与量化训练框架，以最大限度减少精度损失。
#### 0x02 量化推理
- 量化方法  

本文提出的非对称量化方案即为量化后的整数 `q` 到原始的实数 `r` 的仿射映射，数学形式表达如下：  
$$r=S(q-Z) \ \ \ \ \ (1)$$
其中， `S` 与 `Z` 为量化参数，对激活和权重的每一个数组使用独立的参数。对于 `B-bit` 量化 `q` 就是 `B-bit` 的整数，当 `B==8` 时，对于偏置这一类参数通常是 `32-bit` 整数。 `S` 可以是任意正实数， `Z` 的数据类型与 `q` 相同。
- 整数矩阵乘法  

这一小节的目的主要是阐述两个问题，也是这篇文章的重难点：  

a. 如何使用上述公式（1）将实数（即 `r` ，浮点型）的运算转换为量化值（即 `q` ，整型）的计算？  
b. 由于 `S` 为浮点型，如何在后续计算中规避 `S` 浮点型的运算而转为定点运算？ 

假设两个 `N×N` 的矩阵相乘，公式（1）转变为（2），两矩阵乘法表示为（3）：  
$$r_α^{(i,j)}=S_α(q_α^{(i,j)}-Z_α) \ \ \ \ \ \ \ α=1,2 \ or \ 3,1≤i,j≤N \ \ \ \ \ (2)$$
$$S_3(q_3^{(i,j)}-Z_3)=∑_{j=1}^{N}S_1(q_1^{(i,j)}-Z_1)S_2(q_2^{(i,j)}-Z_2) \ \ \ \ \ (3)$$
为了解决问题 `a` ，将公式（3）化简为（4）-（5）：  
$$q_3^{(i,k)}=Z_3+M∑_{j=1}^{N}(q_1^{(i,j)} -Z_1)(q_2^{(j,k)}-Z_2) \ \ \ \ \ (4)$$
$$M≔\frac{S_1 S_2}{S_3} \ \ \ \ \ (5)$$
要将原始浮点型算术转为量化后整型算数，我们首先得到公式（4），其中除了 `M` 为浮点型，其他均为整型，看来我们就差一步之遥了。接着就遇到了问题 `b` ，要如何规避浮点型 $S_α$ 的计算呢？有了公式（5）能够看出M只和$S_1$ 、$S_2$和$S_3$有关，而且在量化模型后这三个参数是已知的常量，也就是说M能够被离线计算。作者在论文中说到由经验所得 `M` 的分布总是在(0,1)之间，那么将 `M` 表示为公式（6）：  
$$M=2^{-n}M_0 \ \ \ \ \ (6)$$
&#x1F449;难点来了，根据公式（6）要让浮点型的M通过整数或整数的运算来表示，首先 $2^{-n}$ 在计算机运算中可以直接通过位移来实现，这是非常有效且方便的，那么 $M_0$ 要取什么样的整数呢？个人理解&#x1F914;为下面几步&#x1F199;：  

i. $M=2^{-n}M_0→M_0=2^{n}M$。  
ii. 对 `n` 进行枚举使得 $M_0∈[0.5,1)$。  
iii. 将 $M_0$ 定点化即 $M_0^{'}=⌊2^{b}×M_0⌉$，这里的 `b` 根据硬件设备支持的最高位数设置，如32位的设备上， `b=31` ，这里就解释了在第ii步中为什么要使得 $M_0∈[0.5,1)$。  
iv. 前三个步骤都是离线进行的，这样一来，当计算X×M时就可以近似为$X×M≈(X×M_0^{'})>>2^{-n-b}$，此时公式里出现的参数都是整型表示，浮点乘法转换为了定点乘法和移位操作。  

经过上述公式（2）-（6）以及i-iv一系列操作，我们已经解决了问题 `a` 和 `b` 。  
- 高效处理零点  

在非对称量化中，除了缩放因子 `S` 这个参数以外，还有一个零点参数 `Z` 。回顾公式（4），对于矩阵的每一个元素 $q_3^{(i,k)}$ 需要 $2N^2$ 次减法，矩阵一共有 $N^2$ 个元素，因此共有 $2N^3$ 次减法。若化简如下：  
$$q_3^{(i,k)}=Z_3+M(NZ_1 Z_2-Z_1 ∑_{j=1}^{N}q_2^{(i,k)}-Z_2∑_{(j=1)}^{N}q_1^{(i,j)}+∑_{(j=1)}^{N}q_1^{(i,j)}q_2^{(i,k)} \ \ \ \ \ (7)$$
此时间复只需 $2N^2$ 次，其他没咋变化，也是减少了不少计算量。  
- 典型的融合层部署  

通常量化后的推理是uint8输入和uint8权重，根据之前的公式（7）能够看出卷积操作中有很多乘法和加法。两个uint8的数相乘为了防止溢出，至少需要16位寄存器存储结果；同理两个16位数相加（就相当于两个16位数乘2），多个16位数相加使用32位寄存器比较保险。因此乘法后的数据为int32，偏执在之前说过也是int32，两者再相加。卷积操作后获得了int32位结果，但为了后续的推理，对后续的网络层来说需要保持uint8的输入，因此还需要以下三步： 

    i.scale down（将int32的结果缩放为8bit）  
    ii.cast down（接着转换到uint8）  
    iii.activation function（使用激活函数产生非线性的8-bit输出）
#### 0x03 模拟量化训练  
    - 权重通常在与输入卷积之前量化，若有BN层，将会使用“BN折叠”操作将BN的计算融合到量化前的权重中。  
    - 激活通常在经过激活函数或全连接层之后量化，或者在与旁路’add’或’concatinate’之后量化。 
但是有了前文所述的量化推理，但是`S`、`Z`、`q`等参数怎么得到的呢？我们有公式组（8）：  
$$clamp(r;a,b)≔min⁡(max⁡(x,a),b) $$
$$s(a,b,B)≔\frac{b-a}{2^B-1}$$
$$q(r;a,b,B)≔⌊\frac{clamp(r;a,b)-a}{s(a,b,B)}⌉s(a,b,B)+a \ \ \ \ \ (8)$$
`B` 为量化等级，如int8量化，`B=8` 以上就是量化方程的定义。  
- 学习量化范围  

故事越来越清晰&#x1F64B;，我们有了公式（8）的定义，问题就从 `S`、`Z`、`q` 等参数怎么得到转换为怎么求 `a`、`b`、`B`。首先对于 `B` 是量化等级所决定的，如int8的量化， `B=8` 。  
&#x1F6A9;对于权重量化，通常设 $a≔min⁡ w,b≔max⁡w$，经微调是量化为int8的权重分布在[-127,127]范围内。  
&#x1F6A9;对激活量化，`a`、`b` 的取值从训练数据中统计，使用的方法多样。例如EMA（Exponential Moving Average）的方式，具体公式如下： 
$$movMax=movMax×momenta+max⁡(abs(currAct))×(1-momenta) \ \ \ \ (9)$$
其中 `curAct` 表示当前batch的激活值，`momenta` 一般取 `0.95`，训练完成后量化银子可以通过公式组（8）的第二子公式令 `b-a=movMax` 获得。

- BN折叠
BN折叠主要是以下公式表示：  
$$BN(x)=\frac{γ(x-μ)}{\sqrt{σ^2+ϵ}}+β$$
$$y=BN(W:x)=\frac{γ(W:x-μ)}{\sqrt{σ^2+ϵ}}+β=\frac{γW}{\sqrt{σ^2+ϵ}}x+(β-\frac{γμ}{\sqrt{σ^2+ϵ}}) \ \ \ \ \ (10)$$

其中 $W:x$ 表示卷积，`γ`、`β`、`ϵ`为超参，`μ`、`σ`分别为均值和方差。此时，第一项对 `x` 的系数就为折叠后的新权重，第二项位新偏执。
### 2.理解对称量化  
同样的，该部分通过阅读论文 [EasyQuant: Post-training Quantization via Scale Optimization](https://arxiv.org/pdf/2006.16669.pdf) 和[NVIDIA 8-bit Inference with TensorRT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)来理解对称量化。

我们先来看看论文，量化分为训练后量化（Post-training Quantization-PTQ）和量化感知训练（Quantization-aware training）。两者区别主要为前者是在已经训练好的网络上使用极少量的不带标签的数据进行量化，后者是在网络训练过程中使用量化来干预网络的分布。在非对称量化中，我们已经知道缩放因子S通常直接取绝对值最大值的数，或者使用EMA方法统计等。EasyQuant提出了一种使用cos相似性作为目标函数来交替搜索权重和激活的最优量化因子的训练后量化方法。

#### 0x01 介绍
这篇论文的主要贡献为： 

    - 作者提出一种训练后量化的缩放因子的优化方法，它交替搜索权重和激活的目标缩放因子，并获得与量化感知训练相当的精度。
    - 作者将提出的方法实施到int7量化中，提高了int16的存储效率。
    - 大量实验表明作者提出的量化方法在int7量化中去了与int8相当的精度。
#### 0x02 方法
- 线性量化公式  

线性量化公式可表示为：
$$Q(X,S)=Clip(Round(X \cdot S)) \ \ \ \ (1)$$
其中X是一个张量， `S` 为正实数表示的缩放因子， `Round()` 表示向上取整（在不同方法中，还能够使用不同的取整方式，如四舍五入、向下取整等），`∙` 表示逐元素乘积， `Clip()` 表示超出量化范围的值被直接裁剪。

我们设神经网络中的第 `l` 层量化为${\lbrace A_l,W_l,S_l\rbrace}_{l=1}^L$，$A_l$，$W_l$，$S_l$分别为浮点型表示的输入、权重、量化因子。其中$S_l$包括权重和激活的量化因子$S_l^w$、$S_l^a$。第l层的输出的特征图为$O_l$（浮点型），其对应的量化后的特征图为$\hat{O}_l$。由此，网络的线性量化和去量化操作有如下公式表示：  
$$\hat{O}_l=\frac{Q(A_l,S_l^a) \ast Q(W_L,S_l^w)}{S_l^a \cdot S_l^w} \ \ \ \ (2)$$  
其中 `*` 表示卷积操作，若不量化，网络的输出将如下表示：
$$O_l=A_l*W_l \ \ \ \ (3)$$
这里就很有意思啦&#x1F609;，通过公式（2）-（3），我们希望量化后的$\hat{O}_l$要尽可能的接近原来的$O_l$，而它们之间的差异主要受量化因子$S_l$的影响（它会影响取整和裁剪的误差），所以，作者就提出了利用余弦相似度作为衡量$O_l$和$\hat{O}_l$之间的误差的目标函数，从而以某种方法枚举$S_l$找到最合适的量化因子&#x1F44D;。

- 缩放优化  

前面说到我们需要找到合适的缩放因子，而目前常用的方法是利用KL散度（即相对熵，它表示两个分布之间的差异）来衡量两者之间的差异。通常做法是利用至少1000个样本的校准数据（与训练数据相似，但不需要标签）来近似每一层激活（特征图）的分布，对权重则直接取绝对值最大值。这会忽略优化真实与量化之间分布的可能性，同时内存资源也还存在优化的空间。

因此作者提出使用余弦相似度来优化，公式如下：
$$max_{s_l}\frac{1}{N}∑_{i=1}^Ncos⁡(O_l^i \, \hat{O}_l^i)$$
$$s.t.  S_l∈R^+  \ \ \ \ (4)$$
主要步骤如下：  
i.首先固定$S_l^a$，利用公式（4）对$S_l^w$进行优化；  
ii.然后固定$S_l^w$，利用公式（4）对$S_l^a$进行优化；  
iii.重复i-ii直到公式（4）收敛或超出时间限制。  

其中为了快速收敛，$S_l^a$和$S_l^w$使用最大值进行初始化。对于它们的搜索空间，使用$[αS_l,βS_l]$区间n等分进行枚举，其中 `α=0.5`，`β=2`, `n=100`。对于在上述区间中对$S_l$的搜索，作者实验发现使用最简单的枚举对不规则的波动更鲁棒。注意，对权重的量化细腻度为 `per-channel`，对激活的量化细腻度为 `per-tensor`（或者叫做 `per-layer` ）。这里作者还贴出了整个网络的量化流程伪代码，非常清晰明了，建议大家阅读原文，对于对称量化的理解到这也差不多啦，论文剩下的部分就是说明int7量化的一些细节和实验部分（为了偷懒这里就不再描述了&#x1F601;）。
### 3.对比对称量化 & 非对称量化
通过前面的学习我们已经对对称和非对称两种量化方式有了基本的了解，他们的主要区别是对称量化没有零点偏移，而非对称量化存在一个零点偏移。那两种方式有什么优缺点呢。我个人理解如下：  

对称量化：  

    - 优点有：计算过程简单，部署高效，能够节省更多的计算资源，推理速度更快，量化过程也更便捷。  
    - 缺点有：精度损失可能会更大，尤其对于数据分布不均匀的网络，量化造成的噪声更大。  

非对称量化：  

    - 优点有：精度损失小。  
    - 缺点有：计算和量化过程复杂（量化参数的优化更复杂），计算资源占用相对于对称量化会更高一些。  
### 4.理解对称量化的 conv 计算过程  
对于这个部分，主要内容为：  

    - 了解堆成量化中校准数据的使用；
    - 了解ncnn int8的量化流程；
    - 了解ncnn int8的卷积过程。

- 校准数据的使用  

ncnn int8中的量化思想可以从[NVIDIA 8-bit Inference with TensorRT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)讲起。

首先我们结合nvidia的PPT以及大佬们的知乎软文来进一步了解对称量化的实施细节。理解过程中参考了以下软文以及源码，讲的都非常详细&#x1F44D;：  
[知乎- Int8量化-介绍（一）](https://zhuanlan.zhihu.com/p/58182172)  
[ncnn int8源码-python版本]( https://github.com/BUG1989/caffe-int8-convert-tools/blob/93ec69e465252e2fb15b1fc8edde4a51c9e79dbf/caffe-int8-convert-tool-dev-weight.py#L157)
（或者可以参考 [c++版本](https://github.com/Tencent/ncnn/blob/30ab31cc4194f57866ba48753aeceae40e823d81/tools/quantize/ncnn2table.cpp#L254)）  

首先看看这一页PPT：  
<img alt="1.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/1.png" width="319" height="123">   
有没有似曾相识，和咱们在第一节-非对称量化中提到的公式（12）很相似。PPT里的公式确实就是线性非对称量化的一种表示，接着假设以这种表示的两个张量相乘得到：  
<img alt="2.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/2.png" width="270" height="204">  
我们可以看出因为两个张量的偏置的存在，使得他们的乘法多出了三个子项。有啥解决办法呢？把偏执删掉&#x1F447;！  
<img alt="3.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/3.png" width="245" height="177">  
得到：  
<img alt="4.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/4.png" width="461" height="160">  
神清气爽有没有！等等，此时量化公式就变成了右边所表示的样子，这不就变成了对称量化嘛。但是直接把偏置删掉真的可以吗？作者们通过实验证明把偏置删掉对精度的损失是可以接受的（当然，通过后来陆续出现的对称量化的工作也能证明没有了偏置还是能够达到很高的精度的）。现在我们只需要考虑怎么得到合适的缩放因子。看看下面两种方式：  
<img alt="5.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/5.png" width="416" height="228">  
假设我们缩放范围是[-127,127]，左边的方法直接找到原数据的绝对值除以127作为缩放因子。这样虽然简单直白，但是如果原数据的正负分布不均匀，那么将会造成很严重的精度损失。因此可以选择右边的方式，找到一个阈值，把这个阈值以外的数据都缩到边界上，在用这个阈值计算缩放因子。

那么问题就到了怎么找到合适的阈值。还记得在第二篇论文EasyQuant中咱们提到过的KL散度（也称KLD）吗，NVIDIA就是这么干的。在PTQ中，对于权重量化，当网络训练好以后，权重的分布是固定的而激活的分布是根据输入来决定，因此权重的量化不需要校准数据而激活则需要。简要流程如下：  
<img alt="6.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/6.png" width="251" height="182">  
大概就是：  

    i.利用校准数据统计每一层激活的直方图；  
    ii.根据直方图，对不同的阈值生成不同的数据分布；
    iii.选择这些生成的分布与原数据相对熵（KL散度）最小的对应的阈值作为结果。  
再往细节一点走，上述 `i` 和 `iii` 中的步骤能够理解，但是 `ii` 中怎么根据直方图列出不同阈值呢？伪代码如下：  
<img alt="7.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/NVIDIA-8-bit-inference-with-TensorRT/7.png" width="393" height="212">    
结合[ncnn int8源码-python版本]( https://github.com/BUG1989/caffe-int8-convert-tools/blob/93ec69e465252e2fb15b1fc8edde4a51c9e79dbf/caffe-int8-convert-tool-dev-weight.py#L157)一起理解。首先输入为通过校准数据得到的激活的2048个bin直方图，每个bin长为 $len_{1}=\frac{max(abs)}{2048}$。因为int8量化范围在[-127,127]之间，所以阈值从128开始枚举到2048。对每一个bin循环：  
i. 将第 `i` 个bin作为截断区，第 `i` 及 `i` 之前的bin在内的数据作为 `P`；  

ii. 将截断区外的值求和，将求和加到边界上（我很赞同这个[知乎]( https://zhuanlan.zhihu.com/p/58182172)中大佬说的观点，将界外的值求和加到边界上有两个原因：一是在计算KL散度时要计算数据概率，需要所有数的总值；二是将界外信息添加进来）；  

iii. 计算 `P` 的概率分布；  

iv. 将P中的数据重新划分为128个bin，每个新的bin长为 $len_{2}=\frac{i×len_{1}}{128}$，在[ncnn int8源码-python版本]( https://github.com/BUG1989/caffe-int8-convert-tools/blob/93ec69e465252e2fb15b1fc8edde4a51c9e79dbf/caffe-int8-convert-tool-dev-weight.py#L157)中，这里的计算是通过 $num_{bin}=round(\frac{i}{128})$ 来确定，新的每一个bin替换为 $num_{bin}$ 个原来bin的和，即量化到int8记为 `Q`；  

v. 现在 `Q` 是128个bin，为了和 `P` 计算KL散度，需要将 `Q` 的长度拓展到 `i` 个bin，使得与 `P` 的长度相同。列举PPT中的例子，假设现在 `P=[1,0,2,3,5,3,1,7]` 有8个bin，根据第 `iv` 步需要量化到2个bin长，$\frac{8}{2}=4$ 则 `Q=[1+0+2+3,5+3+1+7]=[6,16]` 。来到第 `v` 步，需要将 `Q` 拓展到与 `P` 一样的长度，`Q=[6/3,0,6/3,6/3,16/4,16/4,16/4,16/4  ]=[2,0,2,2,4,4,4,4]`。根据PPT给出的例子以及[ncnn int8源码-python版本](https://github.com/BUG1989/caffe-int8-convert-tools/blob/93ec69e465252e2fb15b1fc8edde4a51c9e79dbf/caffe-int8-convert-tool-dev-weight.py#L157)，这里的扩展操作应该是使用量化后的数除以为该bin贡献的数对数量，`0` 除外；  

vi. 求 `Q` 的概率分布；  

vii. 计算KL散度；  

遍历所有128-2048 bin后，选择KL散度最小的bin的索引记为 `m`，计算 $(m+0.5)*len_{1}$作为最终阈值结果，再根据 $S=\frac{128}{threshold}$ 得到缩放因子。  
- ncnn int8的量化、卷积过程  

ncnn int8的量化流程和前面NVIDIA int8的流程很相似，使用的都是 `PTQ+对称量化+线性量化`，结合以下资料学习：

[知乎- MegFlow 和 ncnn int8 简介](https://zhuanlan.zhihu.com/p/476605320)  
[MegFloe and ncnn int8-PPT]( https://docs.qq.com/slide/DWWdkWFd1R0pqaVZp)  
[知乎- Int8量化-介绍（一）](https://zhuanlan.zhihu.com/p/58182172)  
[知乎-从TensorRT与ncnn看CNN卷积神经网络int8量化算法](https://zhuanlan.zhihu.com/p/387072703)  

首先看看ncnn int8如何对权重进行量化的：  
 <img alt="1.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/ncnn%20int8-img/1.png" width="538" height="325">  
对权重量化使用的是 `per-channel` 细腻度，其缩放因子直接使用$S_l^w=\frac{127}{max⁡(abs(per out channel))}$。对于激活量化使用的是 `per-layer(per-tensor)` 的细腻度，缩放因子的计算则是与前面说的NVIDIA使用的方法一样：  
<img alt="2.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/ncnn%20int8-img/2.png" width="538" height="325">  
是不是很简单粗暴呢？嘿嘿&#x1F600;。接着再来看看量化流程以及理解论文里常看到的 `Dequant` 和 `Requant` 有啥不一样：  
 <img alt="3.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/ncnn%20int8-img/3.png" width="538" height="325">  
以一个两层卷积网络举例子&#x1F330;，激活的量化是在推理时在线完成的（但其缩放系数是离线通过校准数据得到的），权重的量化通常是推理之前就量化好保存下来的。浮点型的输入通过量化得到int8型，与int8型的权重进行int8卷积得到int32的输出，经过 `去量化(Dequant)+量化(Quant)=Requant` 再转为int8作为下一层的输入，最后一层的输出经过一次 `去量化(Dequant)` 得到浮点型的输出作为结果。  

摘自[知乎- MegFlow 和 ncnn int8 简介](https://zhuanlan.zhihu.com/p/476605320)
`“对于输出通道为32的卷积层，ncnn int8的核心就是找到 32+1 个系数”`，其中 `32` 个系数就是 `per-channel weight quantization` 的缩放系数，`1` 个系数则是 `per-tensor activation quantization` 的缩放系数。

对于带有分支需要 `add` 或 `concat` 的网络就比较复杂，因为他们需要量化操作是受同样的缩放因子影响，因子再ncnn int8中选择先对两分支去量化以后再合并：  
 <img alt="4.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week2/ncnn%20int8-img/4.png" width="538" height="325">  

### 结语
呜呼~&#x1F916;，至此，咱已经了解了对称和非对称量化，也了解了NVIDIA TensorRT和ncnn int8中的量化细节，通过知乎大佬们的分享和源代码的学习，算是收获多多！这次分享属于学习路上的一片笔记，如有理解不对的地方，欢迎大家指导和讨论！（表情）最后，欢迎大家关注&#x2B50;：  
[MM Deploy]( https://github.com/open-mmlab/mmdeploy)  
[ncnn]( https://github.com/Tencent/ncnn)  
[LY-mmdeploy-summer-camp]( https://github.com/LiuYi-Up/mmdeploy-summer-camp)
