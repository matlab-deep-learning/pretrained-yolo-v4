# Pretrained YOLO v4 Network For Object Detection
This repository provides a pretrained YOLO v4[1] object detection network for MATLAB&reg;. [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=matlab-deep-learning/pretrained-yolo-v4)

**Creator**: MathWorks Development


## Requirements
- MATLAB® R2022a or later
- Deep Learning Toolbox™
- Computer Vision Toolbox™
- Computer Vision Toolbox™ Model for YOLO v4 Object Detection

Note: Previous MATLAB® release users can use [this](https://github.com/matlab-deep-learning/pretrained-yolo-v4/tree/previous) branch to download the pretrained models.


## Getting Started
[Getting Started with YOLO v4](https://in.mathworks.com/help/vision/ug/getting-started-with-yolo-v4.html)


### Detect Objects Using Pretrained YOLO v4
Use to code below to perform detection on an example image using the pretrained model.

Note: This functionality requires Deep Learning Toolbox™ and the Computer Vision Toolbox™ for YOLO v4 Object Detection. You can install the Computer Vision Toolbox for YOLO v4 Object Detection from Add-On Explorer. For more information about installing add-ons, see [Get and Manage Add-Ons](https://in.mathworks.com/help/matlab/matlab_env/get-add-ons.html).

```
% Load pretrained detector
modelName = 'csp-darknet53-coco';
detector = yolov4ObjectDetector(name);

% Read test image.
image = imread('visionteam.jpg');

% Detect objects in the test image.
[boxes, scores, labels] = detect(detector, img);

% Visualize detection results.
img = insertObjectAnnotation(img, 'rectangle', bboxes, scores);
figure, imshow(img)
```
![alt text](images/result.png?raw=true)

### Choosing a Pretrained YOLO v4 Object Detector
You can choose the ideal YOLO v4 object detector for your application based on the below table:

| Model | Input image resolution | mAP  | Size (MB) | Classes |
| ------ | ------ | ------ | ------ | ------ |
| YOLOv4-coco | 608 x 608 | 44.2 | 229 | [coco class names](src/+helper/coco-classes.txt) |
| YOLOv4-tiny-coco | 416 x 416 | 19.7 | 21.5 | [coco class names](src/+helper/coco-classes.txt) |

- mAP for models trained on the COCO dataset is computed as average over IoU of .5:.95.

### Train Custom YOLO v4 Detector Using Transfer Learning
To train a YOLO v4 object detection network on a labeled data set, use the [trainYOLOv4ObjectDetector](https://in.mathworks.com/help/vision/ref/trainyolov4objectdetector.html) function. You must specify the class names for the data set you use to train the network. Then, train an untrained or pretrained network by using the [trainYOLOv4ObjectDetector](https://in.mathworks.com/help/vision/ref/trainyolov4objectdetector.html) function. The training function returns the trained network as a [yolov4ObjectDetector](https://in.mathworks.com/help/vision/ref/yolov4objectdetector.html) object.

For more information about training a YOLO v4 object detector, see [Object Detection using YOLO v4 Deep Learning Example](https://in.mathworks.com/help/vision/ug/object-detection-using-yolov4-deep-learning.html).

## Code Generation for YOLO v4
Code generation enables you to generate code and deploy YOLO v4 on multiple embedded platforms. For more information about generating CUDA® code using the YOLO v4 object detector see [Code Generation for Object Detection by Using YOLO v4](https://in.mathworks.com/help/gpucoder/ug/code-generation-for-object-detection-using-YOLO-v4.html)

## YOLO v4 Network Details
YOLO v4 network architecture is comprised of three sections i.e. Backbone, Neck and Detection Head.

![alt text](images/network.png?raw=true)

- **Backbone:** CSP-Darknet53(Cross-Stage-Partial Darknet53) is used as the backbone for YOLO v4 networks. This is a model with a higher input resolution (608 x 608), a larger receptive field size (725 x 725), a larger number of 3 x 3 convolutional layers and a larger number of parameters. Larger receptive field helps to view the entire objects in an image and understand the contexts around those. Higher input resolution helps in detection of small sized objects. Hence, CSP-Darknet53 is a suitable backbone for detecting multiple objects of different sizes in a single image.

- **Neck:** This section comprised of many bottom-up and top-down aggregation paths. It helps to increase the receptive field further in the network and separates out the most significant context features and causes almost no reduction of the network operation speed. SPP (Spatial Pyramid Pooling) blocks have been added as neck section over the CSP-Darknet53 backbone. PANet (Path Aggregation Network) is used as the method of parameter aggregation from different backbone levels for different detector levels.

- **Detection Head**: This section processes the aggregated features from the Neck section and predicts the Bounding boxes, Objectness score and Classification scores. This follows the principle of one-stage anchor based object detector.    

## References
[1] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 (2020).

[2] Lin, T., et al. "Microsoft COCO: Common objects in context. arXiv 2014." arXiv preprint arXiv:1405.0312 (2014).

Copyright 2021 - 2024 The MathWorks, Inc.
