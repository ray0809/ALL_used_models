# 图像分割

这里的代码主要是用于图像分割

## 兼容
The code is tested using keras  
python:3.6  
keras:2.2.4 
tensorflow-gpu:1.8.0  
torchvision:0.2.1

## 更新
| Date     | Update |
|----------|--------|
| 2019-02-20 | 第一次上传 |



## 数据
如果想训练你自己的数据，那么你的数据结构需要如下（文件夹格式）
- imgs
  - xx1.jpg
  - xx2.jpg
  - ...
- masks
  - xx1.png
  - xx2.png
  - ...


## 训练
```
$ python train.py imgs主目录 masks主目录
```
训练之前会按照9:1自动划分训练和验证，并且会自动保存到txt文件夹内


## 参考
  - [qubvel]([https://github.com/keras-team/keras/issues/9498](https://github.com/qubvel/segmentation_models))：构建了几个分割模型库，方便直接调用，省去了手撸，添加了预训练权重（iamgenet），适当的可以微调
  - [keras issue#11796]([https://stackoverflow.com/questions/41075993/facenet-triplet-loss-with-keras](https://github.com/keras-team/keras/issues/11796))：keras训练过程保存model的一个问题
  - [albumentations](https://github.com/albu/albumentations)：一个强大数据增强库，适用于分类，分割，检测



## 讨论
- 借用了qubvel搭建的分割库，构建现有的model，而不需要自己去手撸一个
- 我尝试的是皮肤分割，一个二分类问题，直接使用BCE容易因为label不均造成了预测倾向于背景，这方面可以尝试增加class_weight，参照的是[Enet](https://github.com/TimoSaemann/ENet/blob/master/scripts/calculate_class_weighting.py)的做法，可以去作者github上看
- U-Net用于医学图像做二分类都倾向于使用iou重叠大小作为loss，因为病变区域本身就很小，所以不考虑背景部分
- keras的一个bug，涉及到hdf5 best模型覆盖保存问题，找到一个临时解决方法是保存之前先把原先的删掉，在utils文件内可以看到