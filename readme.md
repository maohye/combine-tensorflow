Combine-Tensorflow: an improved Algorithm of Region Extraction and Multi-Scale Feature applied in Image Detection  
======

[![build](https://img.shields.io/badge/build-passing-green.svg)](https://img.shields.io/travis/maohye/combine-tensorflow)
[![author](https://img.shields.io/badge/author-maohye-blue.svg)](https://img.shields.io/travis/maohye/combine-tensorflow)
[![language](https://img.shields.io/badge/language-python-orange.svg)](https://img.shields.io/travis/maohye/combine-tensorflow)

环境要求
-----------------
* windows
* python(3.6+)
* google colaboratory

依赖包
-----------------
[![tensorflow](https://img.shields.io/badge/tensorflow-v1.13.0-red.svg)](https://img.shields.io/travis/maohye/combine-tensorflow)
[![cuda](https://img.shields.io/badge/cuda-10.0-red.svg)](https://img.shields.io/travis/maohye/combine-tensorflow)

文件夹结构和功能简介
-----------------

--`combine_pascal.py`
  pascal voc0712数据集训练代码，这里把pascal voc2007和2012数据集同时作为训练集。
  训练时必须修改：训练数据的tfrecords格式数据所在文件夹(dataset_dir)、网络参数权重ckpt文件所在文件夹(checkpoint_path)、
  网络模型作用域(checkpoint_model_scope)、模型权重文件和结果指标events文件保存地址(model_dir)，
  其他超参数(如学习率，迭代次数，批量大小等)根据实际情况修改。

--`combine_kitti.py` 
  kitti数据集训练代码
  训练时必须修改：训练数据的tfrecords格式数据所在文件夹(dataset_dir)、网络参数权重ckpt文件所在文件夹(checkpoint_path)、
  网络模型作用域(checkpoint_model_scope)、模型权重文件和结果指标events文件保存地址(model_dir)，
  其他超参数(如学习率，迭代次数，批量大小等)根据实际情况修改。

--`combine_test.py`
  测试代码
  测试时必须修改：数据名(dataset_name,本文是pascalvoc_2007或者kitti)、类别数(pascalvoc--21,kitti--4)、
  测试数据的tfrecords格式数据所在文件夹(dataset_dir)、
  训练好的网络参数权重ckpt文件所在文件夹(checkpoint_path)、模型权重文件和结果指标events文件保存地址(eval_dir)，
  其他超参数(如学习率，非极大值抑制阈值、目标阈值等)根据实际情况修改。

--`tf_utils.py`
  其他操作合集代码
  包含模型参数更新、学习率下降方式选择、权重文件导入指定参数权重等一些小的操作模块

--`tf_convert_data.py`
  数据集转tfrecords
  需要修改：数据名(dataset_name,pascalvoc或kitti)、数据原始图片格式所在文件夹(dataset_dir)、
  输出数据名(output_name,pascalvoc或kitti)、输出tfrecords数据地址(output_dir)

--`datasets`  数据集操作
  --dataset_factory.py  数据工厂，直接调用指定数据集的程序
  --dataset_utils.py  数据格式转换
  --pascalvoc_2007.py  pascal voc 2007数据集说明（数据类别，训练集测试集划分）
  --pascalvoc_2007_2012.py pascal voc 2007和2012数据集说明（数据类别，训练集测试集划分）
  --kitti.py  kitti数据集说明（数据类别，标注文件信息）
  --pascalvoc_common.py pascalvoc数据集整体说明（标注文件信息）
  --kitti_to_tfrecords.py  kitti数据集转tfrecords格式
  --pascalvoc_to_tfrecords.py  pascalvoc数据集转tfrecords格式
  --voc_eval.py  recall,precision等指标写入文件
  
--`nets`  网络结构
  --combine_net.py  融合模型的网络架构，处理pascalvoc数据集
  --combine_net_kitti.py  融合模型的网络架构，处理kitti数据集（相比pascalvoc数据集修改了候选框的尺寸设计）
  --custom_layers.py  一些常规层的设计，不能直接由tensorflow函数定义
  --nets_factory.py  网络工厂，直接调用指定网络架构的程序
  --np_methods.py  候选框的编码、选择、求交并比等
  --ssd_common.py  对真实目标进行预处理，使ground truth与预测结果对应

--`preprocessing` 预处理
  --pascal_preprocessing  pascalvoc数据集预处理，包括尺寸变换、图片翻转、色彩变换等
  --kitti_preprocessing  kitti数据集预处理，包括尺寸变换、图片翻转、色彩变换等
  --preprocessing_factory  预处理工厂，直接调用给定数据集的预处理函数
  --tf_image  图片预处理操作的具体函数定义
  --vgg_preprocessing  其他的一些预处理操作
  
--`tf_extended`  候选框处理
  --bboxes.py  候选框处理函数定义，包括候选框置信度排序、非极大值抑制、尺寸变换等
  --math.py  定义网络中的一些运算
  --metrics.py  计算precision,recall,ap,map指标的函数定义，对应tensorflow模型评估模块的度量(metric)
  --tensors.py  定义其他的张量运算

Output Example
------------------
Here are two examples of successful detection outputs: 

<div align='center'><img src="https://github.com/maohye/combine-tensorflow/blob/master/pictures/1.jpg">

![image](https://github.com/maohye/combine-tensorflow/blob/master/pictures/2.jpg)
