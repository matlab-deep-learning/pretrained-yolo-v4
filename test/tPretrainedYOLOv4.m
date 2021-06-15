classdef(SharedTestFixtures = {DownloadYolov4Fixture}) tPretrainedYOLOv4 < matlab.unittest.TestCase
    % Test for PretrainedYOLOv4
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture downloads the model. Here we check the
    % detections of each models.
    properties        
        RepoRoot = getRepoRoot;
    end
    
    properties(TestParameter)
        Model = iGetDifferentModels();
    end

    methods(Test)
        function exerciseDetection(test,Model)            
            detector = load(fullfile(test.RepoRoot,'models',Model.dataFileName));
            modelName = strsplit(Model.dataFileName,'.');
            image = imread('visionteam.jpg');
            classNames = helper.getCOCOClassNames;
            anchors = helper.getAnchors(modelName{1});
            [bboxes, scores, labels] = detectYOLOv4(detector.net, image, anchors, classNames, 'auto');
            test.verifyEqual(bboxes, Model.expectedBboxes,'AbsTol',single(1e-4));
            test.verifyEqual(scores, Model.expectedScores,'AbsTol',single(1e-4));
            test.verifyEqual(labels, Model.expectedLabels);            
        end       
    end
end

function Model = iGetDifferentModels()
% Load YOLOv4-coco
dataFileName = 'YOLOv4-coco.mat';

% Expected detection results.
expectedBboxes = single([591.50634  241.60702   55.68161   45.15553;...
    28.49908   47.95480  136.47926  368.61718;...
    157.16482   34.28966  101.99460  375.7619;...
    261.34636   39.16327  123.77390  345.26975;...
    387.99372   49.09709  111.1556  328.0656;...
    506.04692   51.24411  140.14930  351.3739;...
    655.68366   49.01465  134.45191  367.1318]);
expectedScores = single([0.8168; 0.9918; 0.9936; 0.9974; 0.9896; 0.9964; 0.9906]);
expectedLabels = categorical({'book';'person';'person';'person';'person';'person';'person'});
detectorYOLOv4Coco = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load YOLOv4-tiny-coco
dataFileName = 'YOLOv4-tiny-coco.mat';

% Expected detection results.
expectedBboxes = single([28.88783   47.35652  121.5511  337.3994;...
  146.31889   32.74744  120.68582  354.8534;...
  260.03797   34.20401  123.97517  341.538;...
  387.34500   47.32603  107.03614  321.148;...
  513.17959   69.06747  129.53408  326.8461;...
  652.33107   60.80466  131.63133  351.3047]);
expectedScores = single([0.9608; 0.9889; 0.9777; 0.9937; 0.9832; 0.9860]);
expectedLabels = categorical({'person';'person';'person';'person';'person';'person'});
detectorTinyYOLOv4Coco = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

 Model = struct(...
    'detectorYOLOv4Coco',detectorYOLOv4Coco,'detectorTinyYOLOv4Coco',detectorTinyYOLOv4Coco);  
end
