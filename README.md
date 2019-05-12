# 天池肺结节检测与分类
MyCode:肺结节检测论文中使用到的方法；
OtherCode:介绍了其他的一些肺结节检测算法的代码；
classification:介绍了一些分类算法的论文和代码；
detection：介绍了一些检测算法的论文和代码；
segmentation：介绍了一些分割算法的论文和代码；
现在主要介绍一下MyCode中的使用。
# 首先介绍一下使用的环境：
Keras 2.1.5+
TensorFlow 1.4.0+
Python 2.7+
Opencv 3
pandas
numpy
h5py
sklearn
Skimage
tqdm
matplotlib
ntpath
shutil
multiprocessing
glob
# 文件中主要包括检测部分、分割部分、分类部分、测试部分以及全局函数。
全局函数中定义了一些全局的方法、引入的模块以及文件路径；
分割部分包括了预训练的处理代码和使用U-Net分割网络和U-Net block分割网络；
检测部分是使用YOLOV3网络训练自己的数据集，主要是新建VOCdevkit的文件夹和修改yolov3-voc.cfg文件；
分类部分主要是先将肺结节切块送入分类网络，分类网络主要用到的是VGG网络、Inception网络和DenseNet网络；
测试部分主要是首先生成分类的概率，然后使用lung16比赛 的评分函数进行评价；
# 运行过程：
首先运行分割网络部分或者检测网络部分。分割网络部分按照A-Train-X中X的值从前往后运行；检测网络按照训练自己数据集的过程，然后训练。
其次，训练分类网络，先运行对肺部切块的代码，之后运行分类模型，调用了DenseNet模型和Inception模型。
之后使用测试集进行测试，运行测试部分代码，按照B-Test-x中X的值依次执行，生成结节概率文件；生成概率后按照lung16评分函数运行C部分代码生成评分。
三种分类网络相结合的是按生成的概率进行结合，求出新的分类概率值后进行评分。
