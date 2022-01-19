function [bboxes,scores,labels] = postprocess(outFeatureMaps, anchors, netInputSize, scale, classNames)
% The postprocess function applies postprocessing on the generated feature
% maps and returns bounding boxes, detection scores and labels.

% Copyright 2021 The MathWorks, Inc.

% Get number of classes.
if isrow(classNames)
    classNames = classNames';
end
classNames = categorical(classNames);
numClasses = size(classNames,1);

% Get anchor boxes and anchor boxes masks.
anchorBoxes = anchors.anchorBoxes;
anchorBoxMasks = anchors.anchorBoxMasks;

% Postprocess generated feature maps.
outputFeatures = [];
for i = 1:size(outFeatureMaps,1)
    currentFeatureMap = outFeatureMaps{i};
    numY = size(currentFeatureMap,1);
    numX = size(currentFeatureMap,2);
    stride = max(netInputSize)./max(numX, numY);
    batchsize = size(currentFeatureMap,4);
    h = numY;
    w = numX;
    numAnchors = size(anchorBoxMasks{i},2);

    currentFeatureMap = reshape(currentFeatureMap,h,w,5+numClasses,numAnchors,batchsize);
    currentFeatureMap = permute(currentFeatureMap,[5,4,1,2,3]);
    
    [~,~,yv,xv] = ndgrid(1:batchsize,1:numAnchors,0:h-1,0:w-1);
    gridXY = cat(5,xv,yv);
    currentFeatureMap(:,:,:,:,1:2) = sigmoid(currentFeatureMap(:,:,:,:,1:2)) + gridXY;
    anchorBoxesCurrentLevel= anchorBoxes(anchorBoxMasks{i}, :);
    anchorBoxesCurrentLevel(:,[2,1]) = anchorBoxesCurrentLevel(:,[1,2]);
    anchor_grid = anchorBoxesCurrentLevel/stride;
    anchor_grid = reshape(anchor_grid,1,numAnchors,1,1,2);
    currentFeatureMap(:,:,:,:,3:4) = exp(currentFeatureMap(:,:,:,:,3:4)).*anchor_grid;
    currentFeatureMap(:,:,:,:,1:4) = currentFeatureMap(:,:,:,:,1:4)*stride;
    currentFeatureMap(:,:,:,:,5:end) = sigmoid(currentFeatureMap(:,:,:,:,5:end));

    if numClasses == 1
        currentFeatureMap(:,:,:,:,6) = 1;
    end
    currentFeatureMap = reshape(currentFeatureMap,batchsize,[],5+numClasses);
    
    if isempty(outputFeatures)
        outputFeatures = currentFeatureMap;
    else
        outputFeatures = cat(2,outputFeatures,currentFeatureMap);
    end
end

% Coordinate conversion to the original image.
outputFeatures = extractdata(outputFeatures);% [x_center,y_center,w,h,Pobj,p1,p2,...,pn]
outputFeatures(:,:,[1,3]) = outputFeatures(:,:,[1,3])*scale(2);% x_center,width
outputFeatures(:,:,[2,4]) = outputFeatures(:,:,[2,4])*scale(1);% y_center,height
outputFeatures(:,:,1) = outputFeatures(:,:,1) -outputFeatures(:,:,3)/2;%  x
outputFeatures(:,:,2) = outputFeatures(:,:,2) -outputFeatures(:,:,4)/2; % y
outputFeatures = squeeze(outputFeatures); % If it is a single image detection, the output size is M*(5+numClasses), otherwise it is bs*M*(5+numClasses)

if(canUseGPU())
    outputFeatures = gather(outputFeatures);
end

% Apply Confidence threshold and Non-maximum suppression.
confidenceThreshold = 0.5;
overlapThresold = 0.5;

scores = outputFeatures(:,5);
outFeatures = outputFeatures(scores>confidenceThreshold,:);

allBBoxes = outFeatures(:,1:4);
allScores = outFeatures(:,5);
[maxScores,indxs] = max(outFeatures(:,6:end),[],2);
allScores = allScores.*maxScores;
allLabels = classNames(indxs);

bboxes = [];
scores = [];
labels = [];
if ~isempty(allBBoxes)
    [bboxes,scores,labels] = selectStrongestBboxMulticlass(allBBoxes,allScores,allLabels,...
        'RatioType','Min','OverlapThreshold',overlapThresold);
end
end
