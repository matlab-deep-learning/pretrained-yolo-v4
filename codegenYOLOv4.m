%% Code Generation for YOLO v4
% The following code demonstrates code generation for pre-trained YOLO v4 
% object detection network, trained on COCO 2017 dataset.

%% Setup
% Add path to the source directory.
addpath('src');

%% Download the Pretrained Network
% This repository uses two variants of YOLO v4 models.
% *YOLOv4-coco*
% *YOLOv4-tiny-coco*
% Set the modelName from the above ones to download that pretrained model.
modelName = 'YOLOv4-coco';
model = helper.downloadPretrainedYOLOv4(modelName);
net = model.net;

%% Read and Preprocess Input Image.
% Read input image.
image = imread('visionteam.jpg');

% Preprocess the image. 
inputSize = net.Layers(1).InputSize;
[I,imageScale] = helper.preprocess(image,inputSize);

% Provide location of the mat file of the trained network.
matFile = 'models/YOLOv4-coco.mat';

%% Run MEX code generation
% The yolov4Predict.m is the entry-point function that takes an input image 
% and give output for YOLOv4-coco or YOLOv4-tiny-coco models. The functions 
% uses a persistent object yolov4obj to load the dlnetwork object and reuses 
% that persistent object for prediction on subsequent calls.
%
% To generate CUDA code for the entry-point functions,create a GPU code 
% configuration object for a MEX target and set the target language to C++. 
% 
% Use the coder.DeepLearningConfig (GPU Coder) function to create a CuDNN 
% deep learning configuration object and assign it to the DeepLearningConfig 
% property of the GPU code configuration object. 
%
% Run the codegen command.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
args = {coder.Constant(matFile), I};
codegen -config cfg yolov4Predict -args args -report

%% Run Generated MEX
% Call yolov4Predict_mex on the input image.
outFeatureMaps = yolov4Predict_mex(matFile,I);

% Get classnames of COCO dataset.
classNames = helper.getCOCOClassNames;

% Get anchors used in training of the pretrained model.
anchors = helper.getAnchors(modelName);

% Visualize detection results.
[bboxes,scores,labels] = helper.postprocess(outFeatureMaps, anchors, inputSize, imageScale, classNames);
annotations = string(labels) + ": " + string(scores);
image = insertObjectAnnotation(image, 'rectangle', bboxes, annotations);

figure, imshow(image);

% Copyright 2021 The MathWorks, Inc.
