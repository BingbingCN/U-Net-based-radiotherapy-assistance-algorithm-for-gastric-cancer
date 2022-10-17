# U-Net-based radiotherapy assistance algorithm for gastric cancer (Pytorch)


## Table of Contents
* [Example](#example)
* [Task](#task)
* [Experiment](#experiment)
* [File Structure](#file-structure)
* [Dataset](#citation)



##Example
The red part represents the intestine, the blue part represents the stomach, and the picture is a medical scan image.
![Example](https://user-images.githubusercontent.com/94735262/196133795-e04ecb96-1e4c-4d90-89b6-0dd3c6eda59f.jpg)


##Task
Our task is to achieve the recognition of normal organs by using semantic segmentation techniques (FCN or U-Net) in deep learning.

##Experiment
###1.Training Loss
![train](https://user-images.githubusercontent.com/94735262/196133918-5b5ec323-dc65-4c2c-b5c8-842892fcc6e7.jpg)


###2.Prediction
![Example1](https://user-images.githubusercontent.com/94735262/196133975-373911b6-7f36-4e1e-97de-833a8c7a9731.jpg)


##File Structure
```angular2html
uwmgi-2-5d-infer-pytorch.ipynb         2.5d-U-Net demo
uwmgi-unet-train-pytorch.ipynb         Simple U-Net demo
Preparation_1.ipynb                    Reproduction work
```
##Dataset
You can download the data from this [link](https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/data) 

