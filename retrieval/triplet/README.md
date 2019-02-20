# triplet with keras 

This is a Keras implementation of the triplet

## 兼容
The code is tested using keras  
python:3.6  
keras:2.2.4 
tensorflow-gpu:1.8.0  
torchvision:0.2.1

## 更新
| Date     | Update |
|----------|--------|
| 2019-01-30 | the first time I upload all files |
| 2019-02-01 | 添加了测试代码|
| 2019-02-18 | 基础网络冻结，修复keras checkpoint的bug|


## 数据
如果想训练你自己的数据，那么你的数据结构需要如下（文件夹格式）
- animals
  - birds
      - birds_1.jpg
      - ...
  - fish
      - fish_1.jpg
      - ...
- holidays
  - christmas
      - christmas_1.jpg
      - ...

## 训练
```
$ python train.py 文件主目录
```
## 测试
```
# type 指的是提取所有图像的特征，还是开始检索测试图片（1:先提取所有的特征，0:开始检索测试图片相似的）
$ python test.py 文件主目录 模型路径 测试图片 type
```

## 参考
  - [keras issue #9498](https://github.com/keras-team/keras/issues/9498)
  - [facenet-triplet-loss-with-keras](https://stackoverflow.com/questions/41075993/facenet-triplet-loss-with-keras)
  - [苏剑林大神博客](https://spaces.ac.cn/archives/4493)


## 讨论
- vgg16作为base_model,替换的很简单
- 去年实际自己就跑过keras下的triplet训练，但是无论我是三输入还是单输入，semi-hard采样或者常规采样, 训练损失始终都无限接近于margin，尝试打印输出向量，都全部接近于0。
- 这里我用的是余弦距离，margin设置为0.3，其他的参数没有过多的去微调
- 欧式距离的margin设置为1.0
- 如果自己的数据不是很多的话，建议不要直接微调整个网络，先训练最后一个embedding层更为好