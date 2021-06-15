function [output, scale] = preprocess(image, netInputSize)
% The preprocess function applies preprocessing on the input image.

% Copyright 2021 The MathWorks, Inc.

inputSize = [size(image,1),size(image,2)];
scale = inputSize./netInputSize(1:2);

output = im2single(imresize(image,netInputSize(1:2)));
end