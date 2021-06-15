function tiledAnchors = applyAnchorBoxOffsets(tiledAnchors,YPredCell,inputImageSize)
% Convert grid cell coordinates to box coordinates.

% Copyright 2021 The MathWorks, Inc.

for i=1:size(YPredCell,1)
    [h,w,~,~] = size(YPredCell{i,1});  
    tiledAnchors{i,1} = (tiledAnchors{i,1}+YPredCell{i,1})./w;
    tiledAnchors{i,2} = (tiledAnchors{i,2}+YPredCell{i,2})./h;
    tiledAnchors{i,3} = (tiledAnchors{i,3}.*YPredCell{i,3})./inputImageSize(2);
    tiledAnchors{i,4} = (tiledAnchors{i,4}.*YPredCell{i,4})./inputImageSize(1);
end
end
