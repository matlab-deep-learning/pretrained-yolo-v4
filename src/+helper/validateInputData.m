function validateInputData(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

if(any(~validBoxes))
    imPaths = info(~validBoxes);
    str = strjoin(imPaths, '\n');
    boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        + str;
    
    msg = (msg + boxErrMsg + newline + newline);
end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'integer', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
try
    validateattributes(boxes, classes, attrs);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end