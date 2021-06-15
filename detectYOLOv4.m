function [bboxes, scores, labels] = detectYOLOv4(dlnet, image, anchors, classNames, executionEnvironment)
% detectYOLOv4 runs prediction on a trained yolov4 network.
%
% Inputs:
% dlnet                - Pretrained yolov4 dlnetwork.
% image                - RGB image to run prediction on. (H x W x 3)
% anchors              - Anchors used in training of the pretrained model.
% classNames           - Classnames to be used in detection.
% executionEnvironment - Environment to run predictions on. Specify cpu,
%                        gpu, or auto.
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h]) formatted as
%              NumDetections x 4.
% scores     - NumDetections x 1 classification scores.
% labels     - NumDetections x 1 categorical class labels.

% Copyright 2021 The MathWorks, Inc.

% Get the input size of the network.
inputSize = dlnet.Layers(1).InputSize;

% Apply Preprocessing on the input image.
[img, scale] = helper.preprocess(image, inputSize);

% Convert to dlarray.
dlInput = dlarray(img, 'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlInput = gpuArray(dlInput);
end

% Perform prediction on the input image.
outFeatureMaps = cell(length(dlnet.OutputNames), 1);
[outFeatureMaps{:}] = predict(dlnet, dlInput);

% Apply postprocessing on the output feature maps.
[bboxes,scores,labels] = helper.postprocess(outFeatureMaps, anchors, ...
    inputSize, scale, classNames);
end