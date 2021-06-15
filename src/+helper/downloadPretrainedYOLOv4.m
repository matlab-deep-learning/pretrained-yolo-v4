function model = downloadPretrainedYOLOv4(modelName)
% The downloadPretrainedYOLOv4 function downloads a YOLO v4 network 
% pretrained on COCO dataset.
%
% Copyright 2021 The MathWorks, Inc.

supportedNetworks = ["YOLOv4-coco", "YOLOv4-tiny-coco"];
validatestring(modelName, supportedNetworks);

dataPath = 'models';
netFileFullPath = fullfile(dataPath, [modelName, '.zip']);

if ~exist(netFileFullPath,'file')
    fprintf(['Downloading pretrained ', modelName ,' network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = ['https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/yolov4/',modelName,'.zip'];
    websave(netFileFullPath, url);
    fprintf('Done.\n\n');
    unzip(netFileFullPath, dataPath);
    model = load(['models/', modelName, '.mat']);
else
    fprintf(['Pretrained ', modelName, ' network already exists.\n\n']);
    unzip(netFileFullPath, dataPath);
    model = load(['models/', modelName, '.mat']);
end
end
