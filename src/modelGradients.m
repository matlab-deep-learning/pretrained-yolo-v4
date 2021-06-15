function [gradients, state, info] = modelGradients(net, XTrain, YTrain, anchors, mask, penaltyThreshold, networkOutputs)
% The function modelGradients takes as input the dlnetwork object net, 
% a mini-batch of input data XTrain with corresponding ground truth boxes YTrain, 
% anchor boxes, anchor box mask, the specified penalty threshold, and the network 
% output names as input arguments and returns the gradients of the loss with respect 
% to the learnable parameters in net, the corresponding mini-batch loss, and 
% the state of the current batch. 
% 
% The model gradients function computes the total loss and gradients by performing 
% these operations.

% * Generate predictions from the input batch of images using the supporting 
% function yolov4Forward.
% * Collect predictions on the CPU for postprocessing.
% * Convert the predictions from the YOLO v4 grid cell coordinates to bounding 
% box coordinates to allow easy comparison with the ground truth data by using 
% the supporting functions generateTiledAnchors and applyAnchorBoxOffsets.
% * Generate targets for loss computation by using the converted predictions 
% and ground truth data. These targets are generated for bounding box positions 
% (x, y, width, height), object confidence, and class probabilities. See the helper 
% function generateTargets.
% * Calculates the mean squared error of the predicted bounding box coordinates 
% with target boxes. See the supporting function bboxOffsetLoss.
% * Determines the binary cross-entropy of the predicted object confidence score 
% with target object confidence score. See the supporting function objectnessLoss.
% * Determines the binary cross-entropy of the predicted class of object with 
% the target. See the supporting function classConfidenceLoss.
% * Computes the total loss as the sum of all losses.
% * Computes the gradients of learnables with respect to the total loss.

% Copyright 2021 The MathWorks, Inc.

inputImageSize = size(XTrain,1:2);

% Gather the ground truths in the CPU for post processing
YTrain = gather(extractdata(YTrain));

% Extract the predictions from the network.
[YPredCell, state] = helper.yolov4Forward(net,XTrain,networkOutputs,mask);

% Gather the activations in the CPU for post processing and extract dlarray data. 
gatheredPredictions = cellfun(@ gather, YPredCell(:,1:6),'UniformOutput',false); 
gatheredPredictions = cellfun(@ extractdata, gatheredPredictions, 'UniformOutput', false);

% Convert predictions from grid cell coordinates to box coordinates.
tiledAnchors = helper.generateTiledAnchors(gatheredPredictions(:,2:5),anchors,mask);
gatheredPredictions(:,2:5) = helper.applyAnchorBoxOffsets(tiledAnchors, gatheredPredictions(:,2:5), inputImageSize);

% Generate target for predictions from the ground truth data.
[boxTarget, objectnessTarget, classTarget, objectMaskTarget, boxErrorScale] = helper.generateTargets(gatheredPredictions, YTrain, inputImageSize, anchors, mask, penaltyThreshold);

% Compute the loss.
boxLoss = loss.bboxOffsetLoss(YPredCell(:,[2 3 7 8]),boxTarget,objectMaskTarget,boxErrorScale);
objLoss = loss.objectnessLoss(YPredCell(:,1),objectnessTarget,objectMaskTarget);
clsLoss = loss.classConfidenceLoss(YPredCell(:,6),classTarget,objectMaskTarget);
totalLoss = boxLoss + objLoss + clsLoss;

info.boxLoss = boxLoss;
info.objLoss = objLoss;
info.clsLoss = clsLoss;
info.totalLoss = totalLoss;

% Compute gradients of learnables with regard to loss.
gradients = dlgradient(totalLoss, net.Learnables);
end