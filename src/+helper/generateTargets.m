function [boxDeltaTarget, objectnessTarget, classTarget, maskTarget, boxErrorScaleTarget] = generateTargets(YPredCellGathered, groundTruth, inputImageSize, anchorBoxes, anchorBoxMask, penaltyThreshold)
% generateTargets creates target array for every prediction element
% x, y, width, height, confidence scores and class probabilities.

% Copyright 2021 The MathWorks, Inc.

boxDeltaTarget = cell(size(YPredCellGathered,1),4);
objectnessTarget = cell(size(YPredCellGathered,1),1);
classTarget = cell(size(YPredCellGathered,1),1);
maskTarget = cell(size(YPredCellGathered,1),3);
boxErrorScaleTarget = cell(size(YPredCellGathered,1),1);

% Normalize the ground truth boxes w.r.t image input size.
gtScale = [inputImageSize(2) inputImageSize(1) inputImageSize(2) inputImageSize(1)];
groundTruth(:,1:4,:,:) = groundTruth(:,1:4,:,:)./gtScale;

scale_X_Y=[1.2 1.1 1.05];

for numPred = 1:size(YPredCellGathered,1)
    
    % Select anchor boxes based on anchor box mask indices.
    anchors = anchorBoxes(anchorBoxMask{numPred},:);
    scaleXY = scale_X_Y(numPred);
    
    bx = YPredCellGathered{numPred,2};
    by = YPredCellGathered{numPred,3};
    bw = YPredCellGathered{numPred,4};
    bh = YPredCellGathered{numPred,5};
    predClasses = YPredCellGathered{numPred,6};
    
    gridSize = size(bx);
    if numel(gridSize)== 3
        gridSize(4) = 1;
    end
    numClasses = size(predClasses,3)/size(anchors,1);
    
    % Initialize the required variables.
    mask = single(zeros(size(bx)));
    confMask = single(ones(size(bx)));
    classMask = single(zeros(size(predClasses)));
    tx = single(zeros(size(bx)));
    ty = single(zeros(size(by)));
    tw = single(zeros(size(bw)));
    th = single(zeros(size(bh)));
    tconf = single(zeros(size(bx)));
    tclass = single(zeros(size(predClasses)));
    boxErrorScale = single(ones(size(bx)));
    
    % Get the IOU of predictions with groundtruth.
    iou = getMaxIOUPredictedWithGroundTruth(bx,by,bw,bh,groundTruth);
    
    % Donot penalize the predictions which has iou greater than penalty
    % threshold.
    confMask(iou > penaltyThreshold) = 0;
    
    for batch = 1:gridSize(4)
        truthBatch = groundTruth(:,1:5,:,batch);
        truthBatch = truthBatch(all(truthBatch,2),:);
        
        % Get boxes with center as 0.
        gtPred = [0-truthBatch(:,3)/2,0-truthBatch(:,4)/2,truthBatch(:,3),truthBatch(:,4)];
        anchorPrior = [0-anchorBoxes(:,2)/(2*inputImageSize(2)),0-anchorBoxes(:,1)/(2*inputImageSize(1)),anchorBoxes(:,2)/inputImageSize(2),anchorBoxes(:,1)/inputImageSize(1)];
        
        % Get the iou of best matching anchor box.
        overLap = bboxOverlapRatio(gtPred,anchorPrior);
        [~,bestAnchorIdx] = max(overLap,[],2);
        
        % Select gt that are within the mask.
        index = ismember(bestAnchorIdx,anchorBoxMask{numPred});
        truthBatch = truthBatch(index,:);
        bestAnchorIdx = bestAnchorIdx(index,:);
        bestAnchorIdx = bestAnchorIdx - anchorBoxMask{numPred}(1,1) + 1;
        
        if ~isempty(truthBatch)
            % Convert top left position of ground-truth to centre coordinates.
            truthBatch = [truthBatch(:,1)+truthBatch(:,3)./2,truthBatch(:,2)+truthBatch(:,4)./2,truthBatch(:,3),truthBatch(:,4),truthBatch(:,5)];
            
            errorScale = 2 - truthBatch(:,3).*truthBatch(:,4);
            truthBatch = [truthBatch(:,1)*gridSize(2),truthBatch(:,2)*gridSize(1),truthBatch(:,3)*inputImageSize(2),truthBatch(:,4)*inputImageSize(1),truthBatch(:,5)];
            for t = 1:size(truthBatch,1)
                
                % Get the position of ground-truth box in the grid.
                colIdx = ceil(truthBatch(t,1));
                colIdx(colIdx<1) = 1;
                colIdx(colIdx>gridSize(2)) = gridSize(2);
                rowIdx = ceil(truthBatch(t,2));
                rowIdx(rowIdx<1) = 1;
                rowIdx(rowIdx>gridSize(1)) = gridSize(1);
                pos = [rowIdx,colIdx];
                anchorIdx = bestAnchorIdx(t,1);
                
                mask(pos(1,1),pos(1,2),anchorIdx,batch) = 1;
                confMask(pos(1,1),pos(1,2),anchorIdx,batch) = 1;
                
                % Calculate the shift in ground-truth boxes.
                tShiftX = truthBatch(t,1)-pos(1,2)+1;
                tShiftY = truthBatch(t,2)-pos(1,1)+1;
                tShiftW = log(truthBatch(t,3)/anchors(anchorIdx,2));
                tShiftH = log(truthBatch(t,4)/anchors(anchorIdx,1));
                
                % Update the target box.
                tx(pos(1,1),pos(1,2),anchorIdx,batch) = tShiftX;
                ty(pos(1,1),pos(1,2),anchorIdx,batch) = tShiftY;
                tw(pos(1,1),pos(1,2),anchorIdx,batch) = tShiftW;
                th(pos(1,1),pos(1,2),anchorIdx,batch) = tShiftH;
                boxErrorScale(pos(1,1),pos(1,2),anchorIdx,batch) = errorScale(t);
                tconf(rowIdx,colIdx,anchorIdx,batch) = 1;
                classIdx = (numClasses*(anchorIdx-1))+truthBatch(t,5);
                tclass(rowIdx,colIdx,classIdx,batch) = 1;
                classMask(rowIdx,colIdx,(numClasses*(anchorIdx-1))+(1:numClasses),batch) = 1;
            end
        end
    end
    boxDeltaTarget(numPred,:) = [{tx} {ty} {tw} {th}];
    objectnessTarget{numPred,1} = tconf;
    classTarget{numPred,1} = tclass;
    maskTarget(numPred,:) = [{mask} {confMask} {classMask}];
    boxErrorScaleTarget{numPred,:} = boxErrorScale;
end
end

function iou = getMaxIOUPredictedWithGroundTruth(predx,predy,predw,predh,truth)
% getMaxIOUPredictedWithGroundTruth computes the maximum intersection over
%  union scores for every pair of predictions and ground-truth boxes.

[h,w,c,n] = size(predx);
iou = zeros([h w c n],'like',predx);

% For each batch prepare the predictions and ground-truth.
for batchSize = 1:n
    truthBatch = truth(:,1:4,1,batchSize);
    truthBatch = truthBatch(all(truthBatch,2),:);
    predxb = predx(:,:,:,batchSize);
    predyb = predy(:,:,:,batchSize);
    predwb = predw(:,:,:,batchSize);
    predhb = predh(:,:,:,batchSize);
    predb = [predxb(:),predyb(:),predwb(:),predhb(:)];
    
    % Convert from center xy coordinate to topleft xy coordinate.
    predb = [predb(:,1)-predb(:,3)./2, predb(:,2)-predb(:,4)./2, predb(:,3), predb(:,4)];
    
    % Compute and extract the maximum IOU of predictions with ground-truth.
    try 
        overlap = bboxOverlapRatio(predb, truthBatch);
    catch me
        if(any(isnan(predb(:))|isinf(predb(:))))
            error(me.message + " NaN/Inf has been detected during training. Try reducing the learning rate.");
        elseif(any(predb(:,3)<=0 | predb(:,4)<=0))
            error(me.message + " Invalid predictions during training. Try reducing the learning rate.");
        else
            error(me.message + " Invalid groundtruth. Check that your ground truth boxes are not empty and finite, are fully contained within the image boundary, and have positive width and height.");
        end
    end
    
    maxOverlap = max(overlap,[],2);
    iou(:,:,:,batchSize) = reshape(maxOverlap,h,w,c);
end
end
