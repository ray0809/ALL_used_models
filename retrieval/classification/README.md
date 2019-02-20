# 基于softmax的检索 

这是keras的一个基础例子
如何使用分类的方法进行图像检索

## 兼容
The code is tested using keras  
python:3.6  
keras:2.1.0  
tensorflow-gpu:1.8.0  
torchvision:0.2.1  

## 更新
| Date     | Update |
|----------|--------|
| 2019-01-30 | the first upload all files |
| 2019-02-01 | 增加测试的代码 |



## 训练数据
你的数据结构如下面所示：
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
### 第一步
创建训练/测试txt，每行的结构类似 “xx.jpg 1”,"xx.jpg 2"
每个图片对应一个类别
```
$ python utils.py your_pic_main_path
```
### 第二步
训练
```
$ python train.py your_pic_main_path your_train_txt your_test_txt your_class_nb
```
### 第三步
测试
```
# type 指的是提取所有图像的特征，还是开始检索测试图片（1:先提取所有的特征，0:开始检索测试图片相似的）
$ python test.py your_pic_main_path your_model_weights_path your_test_pic type
```
## 讨论
- 尝试了只训练后面的全连接，收敛速度很快
- 整体微调的速度会稍微慢点，精度会比上者更高点，实际情况还需要看数据
- 这里使用torchvision的预处理，只做了简单的旋转和resize，具体的额外操作可以看自己的数据本身类型
- 这里使用的resnet作为base_model,可以自己简单的更换