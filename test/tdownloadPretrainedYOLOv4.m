classdef(SharedTestFixtures = {DownloadYolov4Fixture}) tdownloadPretrainedYOLOv4 < matlab.unittest.TestCase
    % Test for downloadPretrainedYOLOv4
    
    % Copyright 2021 The MathWorks, Inc.
    
    
    % The shared test fixture DownloadYolov4Fixture calls
    % downloadPretrainedYOLOv4. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'models');
    end
    
     
    properties(TestParameter)
        Model = {'YOLOv4-coco', 'YOLOv4-tiny-coco'};
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test,Model)
            dataFileName = [Model,'.mat'];
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,dataFileName),'file'),2));
        end
    end
end
