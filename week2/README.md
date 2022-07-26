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
$$ r = S(q-Z)   \tag{1} $$  
其中， `S` 与 `Z` 为量化参数，对激活和权重的每一个数组使用独立的参数。对于 `B-bit` 量化 `q` 就是 `B-bit` 的整数，当 `B==8` 时，对于偏置这一类参数通常是 `32-bit` 整数。 `S` 可以是任意正实数， `Z` 的数据类型与 `q` 相同。
- 整数矩阵乘法  

这一小节的目的主要是阐述两个问题，也是这篇文章的重难点：  

a. 如何使用上述公式（1）将实数（即 `r` ，浮点型）的运算转换为量化值（即 `q` ，整型）的计算？  
b. 由于 `S` 为浮点型，如何在后续计算中规避 `S` 浮点型的运算而转为定点运算？ 

假设两个 `N×N` 的矩阵相乘，公式（1）转变为（2），两矩阵乘法表示为（3）：  
$$ r_α^{(i,j)}=S_α (q_α^{(i,j)}-Z_α) \ \ \ \ \ \ \ α=1,2 \ or \ 3,1≤i,j≤N \tag{2}$$  
$$ S_3(q_3^{(i,j)}-Z_3)=∑_{j=1}^{N}S_1(q_1^{(i,j)}-Z_1)S_2(q_2^{(i,j)}-Z_2) \tag{3}  $$  
为了解决问题 `a` ，将公式（3）化简为（4）-（5）：  
$$ q_3^{(i,k)}=Z_3+M∑_{j=1}^{N}(q_1^{(i,j)} -Z_1)(q_2^{(j,k)}-Z_2)  \tag{4} $$  
$$ M≔\frac{S_1 S_2}{S_3}  \tag{5} $$
要将原始浮点型算术转为量化后整型算数，我们首先得到公式（4），其中除了 `M` 为浮点型，其他均为整型，看来我们就差一步之遥了。接着就遇到了问题 `b` ，要如何规避浮点型 $S_α$ 的计算呢？有了公式（5）能够看出M只和$S_1$ 、$S_2$和$S_3$有关，而且在量化模型后这三个参数是已知的常量，也就是说M能够被离线计算。作者在论文中说到由经验所得 `M` 的分布总是在(0,1)之间，那么将 `M` 表示为公式（6）：  
$$ M=2^{-n}M_0  \tag{6} $$  
难点来了，根据公式（6）要让浮点型的M通过整数或整数的运算来表示，首先 $ 2^{-n} $ 在计算机运算中可以直接通过位移来实现，这是非常有效且方便的，那么 $M_0$ 要取什么样的整数呢？个人理解为下面几步：  

i. $ M=2^{-n}M_0→M_0=2^{n}M $。  
ii. 对 `n` 进行枚举使得 $ M_0∈[0.5,1) $。  
iii. 将 $ M_0 $ 定点化即 $ M_0^{'}=⌊2^{b}×M_0⌉ $，这里的 `b` 根据硬件设备支持的最高位数设置，如32位的设备上， `b=31` ，这里就解释了在第ii步中为什么要使得 $ M_0∈[0.5,1) $。  
iv. 前三个步骤都是离线进行的，这样一来，当计算X×M时就可以近似为$ X×M≈(X×M_0^{'})>>2^{-n-b} $，此时公式里出现的参数都是整型表示，浮点乘法转换为了定点乘法和移位操作。  

经过上述公式（2）-（6）以及i-iv一系列操作，我们已经解决了问题 `a` 和 `b` 。  
- 高效处理零点  

在非对称量化中，除了缩放因子 `S` 这个参数以外，还有一个零点参数 `Z` 。回顾公式（4），对于矩阵的每一个元素 $ q_3^{(i,k)} $ 需要 $ 2N^2 $ 次减法，矩阵一共有 $ N^2 $ 个元素，因此共有 $ 2N^3 $ 次减法。若化简如下：  
$$ q_3^{(i,k)}=Z_3+M(NZ_1 Z_2-Z_1 ∑_{j=1}^{N}q_2^{(i,k)}-Z_2∑_{(j=1)}^{N}q_1^{(i,j)}+∑_{(j=1)}^{N}q_1^{(i,j)}q_2^{(i,k)} \tag{7} $$  
此时间复只需 $2N^2$ 次，其他没咋变化，也是减少了不少计算量。  
- 典型的融合层部署  

通常量化后的推理是uint8输入和uint8权重，根据之前的公式（7）能够看出卷积操作中有很多乘法和加法。两个uint8的数相乘为了防止溢出，至少需要16位寄存器存储结果；同理两个16位数相加（就相当于两个16位数乘2），多个16位数相加使用32位寄存器比较保险。因此乘法后的数据为int32，偏执在之前说过也是int32，两者再相加。卷积操作后获得了int32位结果，但为了后续的推理，对后续的网络层来说需要保持uint8的输入，因此还需要以下三步： 

    i. scale down（将int32的结果缩放为8bit）  
    ii.	cast down（接着转换到uint8）  
    iii. activation function（使用激活函数产生非线性的8-bit输出）  
#### 0x03 模拟量化训练  
    - 权重通常在与输入卷积之前量化，若有BN层，将会使用“BN折叠”操作将BN的计算融合到量化前的权重中。  
    - 激活通常在经过激活函数或全连接层之后量化，或者在与旁路’add’或’concatinate’之后量化。 
但是有了前文所述的量化推理，但是`S`、`Z`、`q`等参数怎么得到的呢？我们有公式组（8）：  
$$ clamp(r;a,b)≔min⁡(max⁡(x,a),b)  $$   
$$ s(a,b,B)≔\frac{b-a}{2^B-1} $$   
$$ q(r;a,b,B)≔⌊\frac{clamp(r;a,b)-a}{s(a,b,B)}⌉s(a,b,B)+a  \tag{8} $$  
以上就是量化方程的定义。  
- 学习量化范围  

故事越来越清晰，我们有了公式（8）的定义，问题就从 `S`、`Z`、`q` 等参数怎么得到转换为怎么求 `a`、`b`、`B`。首先对于 `B` 是量化等级所决定的，如int8的量化， `B=8` 。  
对于权重量化，通常设 $a≔min⁡ w,b≔max⁡w$，经微调是量化为int8的权重分布在[-127,127]范围内。
对激活量化，`a`、`b`的取值从训练数据中统计，使用的方法多样（此处不详细记录了）。
- BN折叠
BN折叠主要是以下公式表示：  
$$ BN(x)=\frac{γ(x-μ)}{\sqrt{σ^2+ϵ}}+β $$   
$$ y=BN(W:x)=\frac{γ(W:x-μ)}{\sqrt{σ^2+ϵ}}+β=\frac{γW}{\sqrt{σ^2+ϵ}}x+(β-\frac{γμ}{\sqrt{σ^2+ϵ}}) \tag{9} $$
其中 $W:x$ 表示卷积，`γ`、`β`、`ϵ`为超参，`μ`、`σ`分别为均值和方差。此时，第一项对 `x` 的系数就为折叠后的新权重，第二项位新偏执。