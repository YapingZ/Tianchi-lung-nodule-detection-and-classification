# 天池肺结节检测与分类
#Tianchi lung nodule detection and classification

MyCode:肺结节检测论文中使用到的方法；
MyCode:Methods used in lung nodule detection papers；

OtherCode:介绍了其他的一些肺结节检测算法的代码；
OtherCode:Introduced some other code for lung nodule detection algorithm；

classification:介绍了一些分类算法的论文和代码；
classification:Introduced papers and codes for some classification algorithms；

detection:介绍了一些检测算法的论文和代码；
detection:Introduced some papers and codes for detection algorithms；

segmentation:介绍了一些分割算法的论文和代码；
segmentation:Introduced papers and code for some segmentation algorithms；

现在主要介绍一下MyCode中的使用。
Now mainly introduce the use of MyCode.
## 使用的环境：
## Environment used:

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
## 文件中主要包括检测部分、分割部分、分类部分、测试部分以及全局函数。
## The file mainly includes the detection part, the division part, the classification part, the test part and the global function.

全局函数中定义了一些全局的方法、引入的模块以及文件路径；
Global methods, imported modules, and file paths are defined in global functions；

分割部分包括了预训练的处理代码和使用U-Net分割网络和U-Net block分割网络；
The segmentation section includes pre-trained processing code and U-Net segmentation network and U-Net block segmentation network；

检测部分是使用YOLOV3网络训练自己的数据集，主要是新建VOCdevkit的文件夹和修改yolov3-voc.cfg文件；
The detection part is to train your own data set using the YOLOV3 network, mainly to create a new VOCdevkit folder and modify the yolov3-voc.cfg file；

分类部分主要是先将肺结节切块送入分类网络，分类网络主要用到的是VGG网络、Inception网络和DenseNet网络；
The classification part mainly sends the lung nodule dicing to the classification network. The classification network mainly uses VGG network, Inception network and DenseNet network；

测试部分主要是首先生成分类的概率，然后使用lung16比赛 的评分函数进行评价；
The test part is mainly to first generate the probability of classification, and then use the scoring function of the lung16 game to evaluate；
## 运行过程：
## working process：

首先运行分割网络部分或者检测网络部分。分割网络部分按照A-Train-X中X的值从前往后运行；检测网络按照训练自己数据集的过程，然后使用./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74 -gpus 0,1 进行训练。
First run the split network part or detect the network part. The split network part runs from the back to the value of X in A-Train-X; the network is tested according to the process of training its own data set, and then uses ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv .74 -gpus 0,1 for training。

其次，训练分类网络，先运行对肺部切块的代码，之后运行分类模型，调用了DenseNet模型和Inception模型。
Secondly, train the classification network, first run the code for the lung dicing, then run the classification model, call the DenseNet model and the Inception model.

之后使用测试集进行测试，运行测试部分代码，按照B-Test-x中X的值依次执行，生成结节概率文件；生成概率后按照lung16评分函数运行C部分代码生成评分。
Then use the test set to test, run the test part of the code, execute in accordance with the value of X in B-Test-x, and generate the nodule probability file; after generating the probability, run the C part code to generate the score according to the lung16 scoring function.

三种分类网络相结合的是按生成的概率进行结合，求出新的分类概率值后进行评分。
The combination of the three classification networks is based on the generated probability, and the new classification probability value is obtained and scored.
