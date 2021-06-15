function YPredCell = applyActivations(YPredCell)
% Apply activation functions on YOLOv4 outputs.

% Copyright 2021 The MathWorks, Inc.

YPredCell(:,1:3) = cellfun(@ sigmoid, YPredCell(:,1:3), 'UniformOutput', false);
YPredCell(:,4:5) = cellfun(@ exp, YPredCell(:,4:5), 'UniformOutput', false);    
YPredCell(:,6) = cellfun(@ sigmoid, YPredCell(:,6), 'UniformOutput', false);
end