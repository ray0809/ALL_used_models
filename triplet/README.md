# Classification with keras 

This is a Keras implementation of the triplet with my pictures

## Compatibility
The code is tested using keras  
python:3.6  
keras:2.1.0  
tensorflow-gpu:1.8.0  
torchvision:0.2.1

## Update
| Date     | Update |
|----------|--------|
| 2019-01-30 | the first time I upload all files |



## Training data
The [pngimg](http://pngimg.com/) dataset has been used for training. This training set is collected for web design, here "selenium+python" we use to download all of the pictures 

if you want train with your own dataset,you may edit the files structure like this:
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





## Traing
here the based model in tripletnet must be trained with softmax befor  
it can see in [classification](https://github.com/ray0809/ALL_used_models/tree/master/classification)
```
$ python train.py your_pic_main_path
```

## Inspired
  - [keras issue #9498](https://github.com/keras-team/keras/issues/9498)
  - [facenet-triplet-loss-with-keras](https://stackoverflow.com/questions/41075993/facenet-triplet-loss-with-keras)
  - [苏剑林大神博客](https://spaces.ac.cn/archives/4493)


## Discuss
- resent50 is the base_model, it can replace simply
- I find that if traing with triplet directly,the loss will equal margin




## 笔记
- 去年实际自己就跑过keras下的triplet训练，但是无论我是三输入还是单输入，semi-hard采样或者常规采样, 训练损失始终都无限接近于margin，尝试打印输出向量，都全部接近于0。
- 这次我就先用classification训练base_model，然后再拿来做triplet的微调，结果发现能够收敛了，所以说刚开始训练数据对于网络来说完全没有区分能力(为什么其他人就能训练出来，代码基本一致，采样也类似的前提下)。
- 也有一个小发现，直接用triplet，余弦距离替换欧式，有稍微的下降，但还是在margin徘徊，有人建议试试降低学习率。