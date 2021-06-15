function anchors = getAnchors(modelName)
% getAnchors function returns the anchors used in the training of the 
% specified pretrained YOLO v4 model.

% Copyright 2021 The MathWorks, Inc.

if isequal(modelName, 'YOLOv4-coco')
    anchors.anchorBoxes = [16 12; 36 19; 28 40;...
                        75 36; 55 76; 146 72;...
                        110 142; 243 192; 401 459];
    anchors.anchorBoxMasks = {[1,2,3]
                            [4,5,6]
                            [7,8,9]};
elseif isequal(modelName, 'YOLOv4-tiny-coco')
    anchors.anchorBoxes = [82 81; 169 135; 319 344;...
                        27 23; 58 37; 82 81];
    anchors.anchorBoxMasks = {[1,2,3]
                            [4,5,6]};
end
end