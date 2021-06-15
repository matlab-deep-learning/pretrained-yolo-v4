function predictions = extractPredictions(YPredictions, anchorBoxMask)
% Function extractPrediction extracts and rearranges the prediction outputs
% from YOLOv4 network.

% Copyright 2021 The MathWorks, Inc.

predictions = cell(size(YPredictions, 1),6);
for ii = 1:size(YPredictions, 1)
    % Get the required info on feature size.
    numChannelsPred = size(YPredictions{ii},3);
    numAnchors = size(anchorBoxMask{ii},2);
    numPredElemsPerAnchors = numChannelsPred/numAnchors;
    allIds = (1:numChannelsPred);
    
    stride = numPredElemsPerAnchors;
    endIdx = numChannelsPred;

    % X positions.
    startIdx = 1;
    predictions{ii,2} = YPredictions{ii}(:,:,startIdx:stride:endIdx,:);
    xIds = startIdx:stride:endIdx;
    
    % Y positions.
    startIdx = 2;
    predictions{ii,3} = YPredictions{ii}(:,:,startIdx:stride:endIdx,:);
    yIds = startIdx:stride:endIdx;
    
    % Width.
    startIdx = 3;
    predictions{ii,4} = YPredictions{ii}(:,:,startIdx:stride:endIdx,:);
    wIds = startIdx:stride:endIdx;
    
    % Height.
    startIdx = 4;
    predictions{ii,5} = YPredictions{ii}(:,:,startIdx:stride:endIdx,:);
    hIds = startIdx:stride:endIdx;
    
    % Confidence scores.
    startIdx = 5;
    predictions{ii,1} = YPredictions{ii}(:,:,startIdx:stride:endIdx,:);
    confIds = startIdx:stride:endIdx;
    
    % Accummulate all the non-class indexes
    nonClassIds = [xIds yIds wIds hIds confIds];
    
    % Class probabilities.
    % Get the indexes which do not belong to the nonClassIds
    classIdx = setdiff(allIds,nonClassIds);
    predictions{ii,6} = YPredictions{ii}(:,:,classIdx,:);
end
end