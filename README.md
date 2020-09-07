
百度论文复现3D-ResNets，原项目地址：[https://github.com/kenshohara/3D-resnets-pytorch](https://github.com/kenshohara/3D-resnets-pytorch)

AI Studio 地址：[https://aistudio.baidu.com/aistudio/projectdetail/735326](https://aistudio.baidu.com/aistudio/projectdetail/735326)
### 环境
* Python 3+
* paddlepaddle 1.8.0 +
```
pip install -r requirement.txt
```
### 使用方法
```
python train.py
```
可开启MIX-UP数据增强，效果未验证。
```
python train.py --mixup
```
### 实验结果
top-1准确率为93.55% 略高于论文中的92.9%。

