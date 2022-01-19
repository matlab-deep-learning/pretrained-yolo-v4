function [lgraph, networkOutputs, anchorBoxes, anchorBoxMasks] = configureYOLOv4(net, classNames, anchorBoxes, modelName)
% Configure the pretrained network for transfer learning.

% Copyright 2021 The MathWorks, Inc.

% Specify anchorBoxMasks to select anchor boxes to use in both the detection 
% heads. anchorBoxMasks is a cell array of [Mx1], where M denotes the number 
% of detection heads. Each detection head consists of a [1xN] array of row 
% index of anchors in anchorBoxes, where N is the number of anchor boxes to 
% use. 
% Select anchor boxes for each detection head based on size—use larger 
% anchor boxes at lower scale and smaller anchor boxes at higher scale.
if strcmp(modelName, 'YOLOv4-coco')
    area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
    [~, idx] = sort(area, 'ascend');
    anchorBoxes = anchorBoxes(idx, :);
    anchorBoxMasks = {[1,2,3]
        [4,5,6]
        [7,8,9]
        };
elseif strcmp(modelName, 'YOLOv4-tiny-coco')
    area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
    [~, idx] = sort(area, 'descend');
    anchorBoxes = anchorBoxes(idx, :);
    anchorBoxMasks = {[1,2,3]
        [4,5,6]
        };
end
 
% Specify the number of object classes to detect, and number of prediction 
% elements per anchor box. The number of predictions per anchor box is set 
% to 5 plus the number of object classes. "5" denoted the 4 bounding box 
% attributes and 1 object confidence.
if isrow(classNames)
    classNames = classNames';
end
numClasses = size(classNames, 1);
numPredictorsPerAnchor = 5 + numClasses;
 
% Modify the layegraph to train with new set of classes.
lgraph = layerGraph(net);

if strcmp(modelName, 'YOLOv4-coco')
    yoloModule1 = convolution2dLayer(1,length(anchorBoxMasks{1})*numPredictorsPerAnchor,'Name','yoloconv1');
    yoloModule2 = convolution2dLayer(1,length(anchorBoxMasks{2})*numPredictorsPerAnchor,'Name','yoloconv2');
    yoloModule3 = convolution2dLayer(1,length(anchorBoxMasks{3})*numPredictorsPerAnchor,'Name','yoloconv3');

    lgraph = replaceLayer(lgraph,'conv_140',yoloModule1);
    lgraph = replaceLayer(lgraph,'conv_151',yoloModule2);
    lgraph = replaceLayer(lgraph,'conv_162',yoloModule3);

    networkOutputs = ["yoloconv1"
        "yoloconv2"
        "yoloconv3"
        ];
elseif strcmp(modelName, 'YOLOv4-tiny-coco')
    yoloModule1 = convolution2dLayer(1,length(anchorBoxMasks{1})*numPredictorsPerAnchor,'Name','yoloconv1');
    yoloModule2 = convolution2dLayer(1,length(anchorBoxMasks{2})*numPredictorsPerAnchor,'Name','yoloconv2');

    lgraph = replaceLayer(lgraph,'conv_31',yoloModule1);
    lgraph = replaceLayer(lgraph,'conv_38',yoloModule2);

    networkOutputs = ["yoloconv1"
        "yoloconv2"
        ];
end    
end
