### 本周目标：
使用`MMDeploy` 的 `tools/deploy.py` 完成 `mmcls resnet` 模型的量化，主要包含以下步骤：  
>1.	安装MMDeploy;
>2.	安装量化工具PPQ；
>3.	模型量化；
>4.	测试；
>5.	理解tools/deploy.py的流程。

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
- 生成校准表：end2end.onnx->end2end.table+end2end_quant.onnx
- 量化：end2end.bin + end2end.param+ end2end.table->end2end_int8.bin + end2end_int8.param
- 最后就是输出pytorch与ncnn_int8量化前后的两个模型对测试图片的测试结果。
（图片）
通过tools/test.py测试的结果如下：

