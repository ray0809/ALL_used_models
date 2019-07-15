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
| 2019-07-13 | 修改大部分结构，loss该用tf接口编写，模仿pytorch的方式重写dataloader|


## 数据
如果想训练你自己的数据，那么你的数据结构需要参考txt文件夹下的内容格式


## 训练
```
$ python train.py 保存目录 保存日期
```
## 测试
```
#测试数据集路径，参考：./txt/test.txt
#joblib路径：保存了每个测试样本的绝对路径，它的标签，它的特征向量
$ python test.py 模型目录 joblib路径 测试数据集路径
```

## 参考
  - [keras issue #9498](https://github.com/keras-team/keras/issues/9498)
  - [facenet-triplet-loss-with-keras](https://stackoverflow.com/questions/41075993/facenet-triplet-loss-with-keras)
  - [苏剑林大神博客](https://spaces.ac.cn/archives/4493)
  - [facenet](https://github.com/davidsandberg/facenet)



## 讨论
这里实验使用的是cifar10的数据集，10000作为测试，resnet38作为base_model，得到的mAP为93.5%，并未做data augmentation，只是直接的resize

