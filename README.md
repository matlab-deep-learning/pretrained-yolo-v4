# Pretrained YOLO v4 Network For Object Detection

This repository provides a pretrained YOLO v4[1] object detection network for MATLAB&reg;. [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=matlab-deep-learning/pretrained-yolo-v4)

Requirements
------------  

- MATLAB&reg; R2021a or later
- Deep Learning Toolbox&trade;
- Computer Vision Toolbox&trade;

Overview
--------

Object detection is a computer vision technique used for locating instances of objects in images or videos. YOLO v4 is a popular single stage object detector that performs detection and classification using CNNs. The YOLO v4 network is composed of a backbone feature extraction network and detection heads for the localization of objects in an image. 

This repository implements two variants of the YOLO v4 object detectors:
- YOLOv4-coco: Standard YOLOv4 network trained on COCO dataset.
- YOLOv4-tiny-coco: Lightweight YOLOv4 network trained on COCO dataset.

The pretrained networks are trained to detect different object categories including person, car, traffic light, etc. These networks are trained using COCO 2017[2] which have 80 different object categories.

For more information about object detection, see [Getting Started with Object Detection Using Deep Learning](https://www.mathworks.com/help/vision/ug/getting-started-with-object-detection-using-deep-learning.html).

Getting Started
---------------

Download or clone this repository to your machine and open it in MATLAB&reg;.

### Setup
Add path to the source directory.

```
addpath('src');
```

### Download the pretrained network
Use the below helper to download the YOLO v4 pretrained models. Use "YOLOv4-coco" model name for selecting standard YOLO v4 pretrained network and "YOLOv4-tiny-coco" model name for tiny YOLO v4 network. 

```
modelName = 'YOLOv4-coco';
model = helper.downloadPretrainedYOLOv4(modelName);
net = model.net;
```

Detect Objects Using Pretrained YOLO v4 
---------------------------------------

```
% Read test image.
image = imread('visionteam.jpg');

% Get classnames of COCO dataset.
classNames = helper.getCOCOClassNames;

% Get anchors used in training of the pretrained model.
anchors = helper.getAnchors(modelName);

% Detect objects in test image.
executionEnvironment = 'auto';
[bboxes, scores, labels] = detectYOLOv4(net, image, anchors, classNames, executionEnvironment);

% Visualize detection results.
annotations = string(labels) + ": " + string(scores);
image = insertObjectAnnotation(image, 'rectangle', bboxes, annotations);

figure, imshow(image)
```
![alt text](images/result.png?raw=true)


Choosing a Pretrained YOLO v4 Object Detector
---------------------------------------------

| Model | Input image resolution | mAP  | Size (MB) | Classes |
| ------ | ------ | ------ | ------ | ------ |
| YOLOv4-coco | 608 x 608 | 44.2 | 229 | [coco class names](src/+helper/coco-classes.txt) |
| YOLOv4-tiny-coco | 416 x 416 | 19.7 | 21.5 | [coco class names](src/+helper/coco-classes.txt) |

- mAP for models trained on the COCO dataset is computed as average over IoU of .5:.95.

Train Custom YOLO v4 Using Transfer Learning
----------------------------------------------------
Transfer learning enables you to adapt a pretrained YOLO v4 network to your dataset. Create a custom YOLO v4 network for transfer learning with a new set of classes and train using the `yolov4TransferLearn.m` script.

Code Generation for YOLO v4
-----------------------------------
Code generation enables you to generate code and deploy YOLO v4 on multiple embedded platforms.

Run `codegenYOLOv4.m`. This script calls the `yolov4Predict.m` entry point function and generate CUDA code for YOLOv4-coco or YOLOv4-tiny-coco models. It will run the generated MEX and give output.  

| Model | Input image resolution | Speed(FPS) with Codegen| Speed(FPS) w/o Codegen | 
| ------ | ------ | ------ | ------ | 
| YOLOv4-coco | 608 x 608 | 14.025 | 1.18 |
| YOLOv4-tiny-coco | 416 x 416 | 49.309 | 9.46 |

- Performance (in FPS) is measured on a TITAN-RTX GPU.

For more information about codegen, see [Deep Learning with GPU Coder](https://www.mathworks.com/help/gpucoder/gpucoder-deep-learning.html)

YOLO v4 Network Details
-----------------------
YOLO v4 network architecture is comprised of three sections i.e. Backbone, Neck and Detection Head.

![alt text](images/network.png?raw=true)

- **Backbone:** CSP-Darknet53(Cross-Stage-Partial Darknet53) is used as the backbone for YOLO v4 networks. This is a model with a higher input resolution (608 x 608), a larger receptive field size (725 x 725), a larger number of 3 x 3 convolutional layers and a larger number of parameters. Larger receptive field helps to view the entire objects in an image and understand the contexts around those. Higher input resolution helps in detection of small sized objects. Hence, CSP-Darknet53 is a suitable backbone for detecting multiple objects of different sizes in a single image.

- **Neck:** This section comprised of many bottom-up and top-down aggregation paths. It helps to increase the receptive field further in the network and separates out the most significant context features and causes almost no reduction of the network operation speed. SPP (Spatial Pyramid Pooling) blocks have been added as neck section over the CSP-Darknet53 backbone. PANet (Path Aggregation Network) is used as the method of parameter aggregation from different backbone levels for different detector levels.

- **Detection Head**: This section processes the aggregated features from the Neck section and predicts the Bounding boxes, Objectness score and Classification scores. This follows the principle of one-stage anchor based object detector.    

References
-----------

[1] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 (2020).

[2] Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).

Copyright 2021 The MathWorks, Inc.
