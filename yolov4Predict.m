function out = yolov4Predict(matFile, image)
%#codegen
% Copyright 2021 The MathWorks, Inc.

% Convert input to dlarray.
dlInput = dlarray(image, 'SSCB');

persistent yolov4Obj;

if isempty(yolov4Obj)
    yolov4Obj = coder.loadDeepLearningNetwork(matFile);
end

% Pass input.
out = cell(size(yolov4Obj.OutputNames,2),1);
[out{:}] = yolov4Obj.predict(dlInput);
