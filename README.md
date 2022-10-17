# U-Net-based radiotherapy assistance algorithm for gastric cancer (Pytorch)


## Table of Contents
* [Example](#example)
* [Task](#task)
* [Experiment](#experiment)
* [File Structure](#file-structure)
* [Dataset](#citation)

##Example
The red part represents the intestine, the blue part represents the stomach, and the picture is a medical scan image.
![Alt](.\Example.jpg)

##Task
Our task is to achieve the recognition of normal organs by using semantic segmentation techniques (FCN or U-Net) in deep learning.

##Experiment
###1.Training Loss
![Alt](.\result\train.jpg)

###2.Prediction
![Alt](.\Example1.jpg)

##File Structure
```angular2html
uwmgi-2-5d-infer-pytorch.ipynb         2.5d-U-Net demo
uwmgi-unet-train-pytorch.ipynb         Simple U-Net demo
Preparation_1.ipynb                    Reproduction work
```
##Dataset
You can download the data from this [link](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/data) 

