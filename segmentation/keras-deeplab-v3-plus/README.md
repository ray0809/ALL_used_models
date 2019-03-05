# Deelplab V3+
deeplab v3+用于图像分割

## 兼容
python:3.6  
keras:2.2.4 
tensorflow-gpu:1.8.0  



## 数据
如果想训练你自己的数据，那么你的数据结构需要如下（文件夹格式）  

        DATA_DIR/
                |-- imgs/
                |    |-- xxx1.jpg
                |    |-- xxx2.jpg
                |    |-- ...
                |-- masks/
                |    |-- xxx1.png
                |    |-- xxx2.png
                |    |-- ...


## 训练
```
$ python train.py imgs主目录 masks主目录
```
训练之前会按照9:1自动划分训练和验证，并且会自动保存到txt文件夹内



## 参考
  - [keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)：作者直接将tf官方的模型转化成了keras版本
  - [albumentations](https://github.com/albu/albumentations)：一个强大数据增强库，适用于分类，分割，检测

