function [YPredCell, state] = yolov4Forward(net, XTrain, networkOutputs, anchorBoxMask)
% Predict the output of network and extract the confidence score, x, y,
% width, height, and class.

% Copyright 2021 The MathWorks, Inc.

YPredictions = cell(size(networkOutputs));
[YPredictions{:}, state] = forward(net, XTrain, 'Outputs', networkOutputs);
YPredCell = helper.extractPredictions(YPredictions, anchorBoxMask);

% Append predicted width and height to the end as they are required
% for computing the loss.
YPredCell(:,7:8) = YPredCell(:,4:5);

% Apply sigmoid and exponential activation.
YPredCell(:,1:6) = helper.applyActivations(YPredCell(:,1:6));
end