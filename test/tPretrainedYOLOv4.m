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
    157.16482   34.28966  101.99460  375.76175;...
    261.34636   39.16327  123.77390  345.26975;...
    387.99372   49.09721  111.15549  328.06540;...
    506.04692   51.24422  140.14930  351.37362;...
    655.68366   49.01477  134.45191  367.13157]);
expectedScores = single([0.8168; 0.9918; 0.9936; 0.9974; 0.9896; 0.9964; 0.9906]);
expectedLabels = categorical({'book';'person';'person';'person';'person';'person';'person'});
detectorYOLOv4Coco = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

% Load YOLOv4-tiny-coco
dataFileName = 'YOLOv4-tiny-coco.mat';

% Expected detection results.
expectedBboxes = single([28.88783   47.35689  121.5511  337.39872;...
  146.31889   32.74814  120.68582  354.85198;...
  260.03797   34.20447  123.97517  341.53702;...
  387.34500   47.32572  107.03614  321.14859;...
  513.17959   69.06781  129.53408  326.84541;...
  652.33107   60.80496  131.63133  351.30410]);
expectedScores = single([0.9608; 0.9889; 0.9777; 0.9937; 0.9832; 0.9860]);
expectedLabels = categorical({'person';'person';'person';'person';'person';'person'});
detectorTinyYOLOv4Coco = struct('dataFileName',dataFileName,...
    'expectedBboxes',expectedBboxes,'expectedScores',expectedScores,...
    'expectedLabels',expectedLabels);

 Model = struct(...
    'detectorYOLOv4Coco',detectorYOLOv4Coco,'detectorTinyYOLOv4Coco',detectorTinyYOLOv4Coco);  
end