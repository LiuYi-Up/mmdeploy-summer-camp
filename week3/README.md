### 本周目标：
使用`MMDeploy` 的 `tools/deploy.py` 完成 `mmcls resnet` 模型的量化，主要包含以下步骤：  
>1.	安装MMDeploy;
>2.	安装量化工具PPQ；
>3.	模型量化；
>4.	测试；
>5.	理解tools/deploy.py的流程。
>6. 补充：算子不匹配问题与mmdeploy的重写机制

### 1.	安装MMDeploy
在上一周的学习，咱已经了解了ncnn量化的基本流程和一些细节，本次量化的推理引擎就选择ncnn。当然MMDeploy还支持ONNXRuntime、TensorRT等等其他推理引擎，供大家选择。MMDeploy安装步骤咱就参考[mmdeploy build 官方文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/01-how-to-build/linux-x86_64.md)，这里我只是充一些自己遇到的问题和解决办法：  
#### 安装构建和编译工具链   

这一步如果没有安装，就按照文档里的步骤进行；如果是已经安装了的，就需要确保版本对齐（我就是后者情况，但是一开始没有注意cmake版本，导致进行后面步骤时报错，又再重新安装解决的）：  
```
a. cmake version >= 3.14.0  # 通过终端命令 cmake –version 查看当前版本;
b. GCC 7+  # 通过终端命令 gcc –version 查看当前版本。
```

#### 安装依赖包  

- a. 安装MMDeploy Converter依赖，按照文档进行即可；
- b. 安装MMDeploy SDK依赖，本周学习只涉及模型的量化转换，不需要SDK，因此可以忽略此步骤；
- c. 安装推理引擎，MMDeploy为咱提供了很多种推理引擎，咱们这次选择的是ncnn：  

1) 编译ncnn：步骤和第一周一样，参考[ncnn 官方文档]( https://github.com/Tencent/ncnn/wiki/how-to-build)编译ncnn，注意&#x2728;这里编译时需要设置 `-DNCNN_PYTHON=ON` 并且 `make install` 一下，参考以下命令: 
```
cmake -CMAKE_BUILD_TYPE=Release -DNCNN_PYTHON=ON-DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
make install
```  
 接着进入ncnn根目录，将其写入环境变量：
```
cd path/to/ncnn
export NCNN_DIR=$(pwd)
```
注意&#x2728;，这里通过 `export` 添加的环境变量只在当前终端有用且周期伴随当终端的关闭而失效。要想永久有效，可参考文档里的`~/.bashrc`更改方法。  

2) 安装pyncnn，这一步其实比较容易出错，可以先按照文档里的命令行进行安装，然后试着在python中`import ncnn`一下，如果成功导入那么应该不会再有问题，如果报错可以尝试：
```
pip uninstall ncnn
cd ${NCNN_DIR}
pip install .
```
这个时候可能网络的好坏也会影响，如果网络失败可以多尝试几次（我是使用前者方法一直导不进`ncnn`后尝试后者成功的）。

#### 编译MMDeploy
>a. 导入环境变量；
>b. 编译安装Model Converter——ncnn自定义算子；
>c. 编译安装Model Converter——安装Model Converter。

至此就完成了MMDeploy的安装。

### 2.安装量化工具PPQ + 3.模型量化
咱们以 MMClassification 中[resnet50-cifar10](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)模型为例进行量化。参考[mmdeploy quantize 官方文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/quantize_model.md)，这里依旧是只说明一些需要补充的我遇到的问题和解决方法:
-	安装PPQ
-	克隆MMClassification
```
git clone https://github.com/open-mmlab/mmclassification.git
```
-	选择需要量化的网络，并添加至环境变量，这里选择 `resnet50-cifar10`，从[resnet-model](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnet)将对应的权重下载下来，接着：
```
cd path/to/mmclassification
export MM_CLS=$(pwd)
export MODEL_PATH=${MM_CLS}/configs/resnet/resnet18_8xb16_cifar10.py
export MODEL_CONFIG=the/path/to/downloaded/model/weight # 将刚刚下载的模型权重添加到MODEL_CONFIG环境变量
```
-	量化模型
建议量化之前，先观看 [deploy.py源码]( https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py)，了解每个参数代表的含义在进行选择，但是记得在那之前添加一下ncnn的环境变量：
```
export PATH=$PATH:${NCNN_DIR}/your/build/path/install/bin
```
文档里的 `--quant-image-dir /path/to/images `如果省略，则将会自动下载默认配置的校准数据集。此时可能会因为网络问题迟迟无法下载，那么可以手动下载对应数据集到`path/to/mmdeploy/data/cifar10`里。

### 4.测试
该步骤也参考[profile 官方文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/02-how-to-run/profile_model.md)进行。也记得首先阅读一下[tool/test.py 源码]( https://github.com/open-mmlab/mmdeploy/blob/master/tools/test.py)了解各个参数的作用哦&#x1F469;。例如：
```
python tools/test.py configs/mmcls/classification_ncnn-int8_static.py ${MODEL_CONFIG} --model path/to/end2end.bin path/to/end2end.param --speed-test --device cpu --metrics accuracy --out out.pkl
```

### 5.理解tools/deploy.py的流程

这一步可以通过调试了解大概过程，观看 [tool/deploy.py源码]( https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py)：  

- 首先将 `pytorch` 模型转换为 `onnx` 中间框架:pytorch-> end2end.onnx:  
- 接着将onnx转换为ncnn: end2end.onnx->end2end.bin + end2end.param  
- 调用ppq生成校准表：end2end.onnx->end2end.table+end2end_quant.onnx  
- 量化：end2end.bin + end2end.param+ end2end.table->end2end_int8.bin + end2end_int8.param  
- 最后就是输出pytorch与ncnn_int8量化前后的两个模型对测试图片的测试结果。   
fp32 result:  
<img alt="fp32.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week3/img/output_pytorch.jpg"> 
int8 result:  
<img alt="fp32.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week3/img/output_ncnn.jpg"> 
通过tools/test.py测试的结果如下：
fp32 result:  
<img alt="fp32.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week3/img/fp32.png"> 
int8 result:  
<img alt="fp32.png" src="https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week3/img/int8.png">   
可以看出，量化后的网络推理速度提升了很多，但同时精度也降了一些。

-------------------------------------------------------------分割线-------------------------------------------------

### 6.算子不匹配问题与mmdeploy的重写机制
前段时间放了暑假，然后就偷懒了&#x1F92D;，现在继续把一些想要补充的内容填上。主要补充模型从 `pytorch` 转到 `onnx` 再到 `ncnn` 过程中涉及到的算子的添加与重写。
#### 了解onnx及为pytorch添加onnx支持

实际上，在此之前我一直不懂 `onnx` 是什么，也不知道为啥量化部署要转中间表示，直到看见下面链接中的一系列文档（&#x2728;这里我必须要感谢mmlab的大佬们，他们是真的想教会我，这是我遇到的最棒的开源社区，来自菜鸡的痛哭感谢&#x1F44D;）：

[mmdeploy/tutorial文档](https://github.com/open-mmlab/mmdeploy/tree/master/docs/zh_cn/tutorial )

如果你和我一样不了解onnx，我非常建议你先阅读上面的一系列文档，再来看看文档第一篇中的这个图应该就会有新的理解了：  

<img alt='arch' src='https://user-images.githubusercontent.com/4560679/156556619-3da7a572-876b-4909-b26f-04e81190c546.png'>

如果要把模型从深度学习框架通过中间表示部署到推理引擎上，理论上就需要模型的每一个算子在三个阶段中都有一对一或者一对多的映射。

&#X1FA90;我们首先再次明确一点：中间表示只是一套标准，它定义了网络的计算图输入输出的格式、属性、名称序号等等内容。由于我们不需要执行 `onnxruntime.InferenceSession()` 来运行 `.onnx` 格式下的模型，当推理引擎拿到这套标准后只需要知道数据从那个算子（或模块）进和出、数据进出的格式有哪些，接着把推理引擎上对应的算子按照得到的进出顺序在搭建起来就能够还原了，因此并不在乎在这个算子（或模块）在中间表示中是怎么计算的（或者说在中间表示中根本不需要计算，只要有这个算子的定义规范好相应的输入输出等等信息就好）。
#### mmdeploy中的新增算子与重写机制

那么当网络从 `pytorch` 转到 `onnx` 时，如果遇到下面两种情况：

- pytorch代码中出现的算子在onnx找不到对应的算子；
- 将导出的 `.onnx` 文件放到Netron（开源的模型可视化工具）中观察到某些算子被撕裂成很多奇怪的算子。

这时候我们就需要自己为算子添加 `onnx` 支持并重写了。回想 [mmdeploy/tutorial文档](https://github.com/open-mmlab/mmdeploy/tree/master/docs/zh_cn/tutorial ) 里提到的三种在pytorch中支持更多算子的方法，我认为最主要的思想就是 `算子->符号函数->注册符号函数` 对吧？但是在 `mmdeploy` 中要怎么做呢？欸嘿&#x1F60F;，别担心，官方也给了教程文档（再次夸赞mmlab社区！）：

[mmdeploy/developer-guide](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/07-developer-guide/support_new_model.md )

现在咱们就可以看看&#x1F330;啦，参考：

[白牛知乎]( https://zhuanlan.zhihu.com/p/505481568 )

其中就讲了在部署ViT 到 `ncnn` 时遇到的一些算子不匹配，手动添加 `onnx` 支持并重写的过程。俺就以这个为例，这个例子中，作者分享了 `MHA`、`GeLu`、`LayerNorm` 三个算子在 `torch2onnx` 时不支持或者被撕裂了，但是这些算子在 `ncnn` 中都支持。

对[MHA]( https://github.com/tpoisonooo/mmdeploy/blob/cb730a04a5711e3d1797cf4d8f4c6013344be641/mmdeploy/codebase/mmcls/models/utils/attention.py#L58 )，作者先使用继承 `torch. autograd.Function` 的方法捏了这个算子使 `onnx` 支持：
```python
class MultiHeadAttentionop(torch.autograd.Function):
    """Create onnx::MultiHeadAttention op."""

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, q_weight: Tensor,
                q_bias: Tensor, k_weight: Tensor, k_bias: Tensor,
                v_weight: Tensor, v_bias: Tensor, o_weight: Tensor,
                o_bias: Tensor, embed_dims: int, num_heads: int) -> Tensor:
        return torch.rand_like(q)

    @staticmethod
    def symbolic(g, q: torch._C.Value, k: torch._C.Value, v: torch._C.Value,
                 q_weight: torch._C.Value, q_bias: torch._C.Value,
                 k_weight: torch._C.Value, k_bias: torch._C.Value,
                 v_weight: torch._C.Value, v_bias: torch._C.Value,
                 o_weight: torch._C.Value, o_bias: torch._C.Value,
                 embed_dims: int, num_heads: int):
        ...

        return g.op(
            'mmdeploy::MultiHeadAttention',
            q,
            k,
            v,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            embed_dim_i=embed_dims,
            num_heads_i=num_heads)
```
然后使用装饰器注册并重写前向过程：
```python
@FUNCTION_REWRITER.register_rewriter(
    func_name='mmcls.models.utils.attention.MultiheadAttention.forward',
    backend=Backend.NCNN.value)
def multiheadattention__forward__ncnn(ctx, self, qkv_input):

    ...

    out = MultiHeadAttentionop.apply(qkv_input, qkv_input, qkv_input, q_weight,
                                     q_bias, k_weight, k_bias, v_weight,
                                     v_bias, o_weight, o_bias, self.embed_dims,
                                     self.num_heads)
    return out
```
上面的操作是不是和 [文档](https://github.com/open-mmlab/mmdeploy/blob/master/docs/zh_cn/tutorial/04_onnx_custom_op.md) 里第三种方法很像。对于[LayerNorm]( https://github.com/tpoisonooo/mmdeploy/blob/support-vision-transformer/mmdeploy/pytorch/ops/layer_norm.py )，作者使用第二种写法，首先使用  `@parse_args` 装饰一个符号函数，然通过 `@SYMBOLIC_REWRITER.register_symbolic` 装饰器注册并重写。
```python
@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    """Symbolic function for `layer_norm`.
    PyTorch does not support export layer_norm to ONNX by default. We add the
    support here. `layer_norm` will be exported as ONNX node
    'mmdeploy::layer_norm'
    """
    weight.setDebugName('layernorm_weight')
    bias.setDebugName('layernorm_bias')
    return g.op(
        'mmdeploy::LayerNorm', input, weight, bias, affine_i=1, epsilon_f=eps)


@SYMBOLIC_REWRITER.register_symbolic(
    'layer_norm', is_pytorch=True, backend=Backend.NCNN.value)
def layer_norm__ncnn(ctx, *args):
    """Register default symbolic function for `layer_norm`.
    Add support to layer_norm to ONNX.
    """
    return layer_norm(*args)
```
同样的，对于[GeLu](https://github.com/tpoisonooo/mmdeploy/blob/support-vision-transformer/mmdeploy/pytorch/ops/gelu.py)，使用第一种方法，通过 `@SYMBOLIC_REWRITER.register_symbolic` 直接注册重写。
```python
@SYMBOLIC_REWRITER.register_symbolic(
    'gelu', is_pytorch=True, arg_descriptors=['v'], backend=Backend.NCNN.value)
def gelu__ncnn(ctx, g, self):
    """Support export GELU with ncnn backend."""
    return g.op('mmdeploy::Gelu', self)
```

这部分的理解还是要先看文档，至此，咱已经对 `mmdeploy` 的重写机制有了了解，那么当在终端敲下 `python3 tools/deploy.py ……` 时，上面所说的操作都是啥时候发生的呢？

很简单的流程在补充线的前一段已经说过，观察[源码]( https://github.com/open-mmlab/mmdeploy/blob/master/tools/deploy.py )   `line138` ：

- 调用 `torch2ir` 时会转向 `pytorch2onnx.torch2onnx()`;
- 接着在这里面调用 `export()` 转向 `export.export()`;
- 在这里面我们会看到熟悉的 `torch.onnx.export()`，在这一行的前面几行我们也会看到 `RewriterContext()`，这说明在将模型 `pytorch2onnx`前，我们已经调用重写机制进行添加算子或重写操作啦。

得到了正确的 `.onnx` 模型文件后，`mmdeploy` 会使用 `ppq` 工具来量化并导出到相应的推理引擎上。可以参考[ppq/inference_with_ncnn文档]( https://github.com/openppl-public/ppq/blob/master/md_doc/inference_with_ncnn.md)了解使用方法。以上就是俺补充的个人理解，很有趣，如果对 `mmdeploy` 部署想要有更深入的了解，非常建议观看官方更多文档和源码。

### 结语
至此，本周学习就到这了！这次分享属于学习路上的一片笔记，如有理解不对的地方，欢迎大家指导和讨论！最后，欢迎大家关注&#x2B50;：  
[MM Deploy]( https://github.com/open-mmlab/mmdeploy)  
[ncnn]( https://github.com/Tencent/ncnn)  
[LY-mmdeploy-summer-camp]( https://github.com/LiuYi-Up/mmdeploy-summer-camp)
