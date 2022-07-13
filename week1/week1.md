### 本周目标
>1.环境配置  
    2.编译ncnn  
    3.使用 tool/quantize 工具量化 squeezenet_v1.1 模型  
    4.学习 image Conv2D 的实现，手写一份naive 卷集代码  
    
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
将 [测试图片](https://github.com/nihui/ncnn-android-squeezenet/blob/master/screenshot.png)  下载到  `${NCNN_DIR}/images/sceenshort.png`   
- 使用 vulkan  
```
cd ${NCNN_DIR}/build20220713  
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=ON -DNCNN_BUILD_EXAMPLES=ON ..  
make -j$(nproc)  
ln -sf ../examples/squeezenet_v1.1.param squeezenet_v1.1.param  
ln -sf ../examples/squeezenet_v1.1.bin squeezenet_v1.1.bin
./examples/squeezenet ../images/screenshrt.png
```
结果为：  
![w](https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/wovulkan.png)  
查看标签使用上图等号前的数字 `(128/143/98)+1` 找出 [GT表格](https://github.com/Tencent/ncnn/blob/master/examples/synset_words.txt) 对应行号的结果，此时 `128+1` 对应的标签为 `black stork, Ciconia nigra` 显然出了问题。
- 不使用 vulkan  
```
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..  
make -j$(nproc)  
./examples/squeezenet ../images/screenshrt.png  
```
结果为：  
![wo](https://github.com/LiuYi-Up/mmdeploy-summer-camp/blob/main/week1/results_img/wvulkan.png)  
`281+1` 对应的标签为 `tabby, tabby cat` ，结果正确。因此之后的操作咱都禁止 vulkan。  
### b.优化模型  
好，咱接着来，优化模型这一步在我的实验过程中似乎对网络的影响是负面的，咱先不管，跟着官方文档走一遍 &#x1F910; ：
```
./tools/ncnnoptimize ./squeezenet_v1.1.param ./squeezenet_v1.1.bin sqznet-opt.param sqznet-opt.bin 0  
```
显示如下结果：
```

```
