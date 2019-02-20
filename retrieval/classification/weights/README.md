# Classification with keras 

This is a Keras implementation of the classification with my pictures

## Compatibility
The code is tested using keras  
python:3.6  
keras:2.1.0  
tensorflow-gpu:1.8.0  
torchvision:0.2.1
## Update
| Date     | Update |
|----------|--------|
| 2019-01-30 | the first upload all files |



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
### First
create the train/text txt file
```
$ python utils.py your_pic_main_path
```
### Second
training
```
$ python train.py your_pic_main_path your_train_txt your_test_txt your_class_nb
```


## Discuss
- resent50 is the base_model, it can replace simply
- the pics is something different from nature pics
- fine-tune the full-connect layer can get 80% accuracy after 15min
- fine-tune full model has very low convergence, but it can get 85% accuracy after 2h
- I use the torchvision transform as my preprocess method


anyway,the repository is used for storing my code and ideas in the future


## 笔记
- keras的Sequence用于数据生成，pytorch的DataLoader跟它类似，听说是借鉴keras的，之前自己一直都没去注意和使用，这次借助这个分类任务，来做一个基础教程，用作自己的参考
- torchvision的预处理很方便，总比自己去写要的快，它经常跟DataLoader配合使用，放在这里跟keras也能完美衔接
- 个人比较喜欢fit_generator，有点强迫症
- 这个分类整体没什么，分类是为了做检索任务，至于提取最后一层做特征，这里就不加了，主要是一个基本的分类结构，以后可以套用