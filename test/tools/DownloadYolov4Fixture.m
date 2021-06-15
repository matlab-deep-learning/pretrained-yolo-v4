classdef DownloadYolov4Fixture < matlab.unittest.fixtures.Fixture
    % DownloadYolov4Fixture   A fixture for calling downloadPretrainedYOLOv4 
    % if necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        Yolov4DataDir = fullfile(getRepoRoot(),'models')
    end
    
    properties
        Yolov4CocoExist (1,1) logical
        TinyYolov4CocoExist (1,1) logical
    end
    
    methods
        function setup(this)            
            this.Yolov4CocoExist = exist(fullfile(this.Yolov4DataDir,'YOLOv4-coco.mat'),'file')==2;
            this.TinyYolov4CocoExist = exist(fullfile(this.Yolov4DataDir,'YOLOv4-tiny-coco.mat'),'file')==2;
            
            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.
            if ~this.Yolov4CocoExist
            	evalc('helper.downloadPretrainedYOLOv4(''YOLOv4-coco'');');                
            end
            if ~this.TinyYolov4CocoExist
            	evalc('helper.downloadPretrainedYOLOv4(''YOLOv4-tiny-coco'');');                
            end
        end
        
        function teardown(this)
            if ~this.Yolov4CocoExist
            	delete(fullfile(this.Yolov4DataDir,'YOLOv4-coco.mat'));
            end
            if ~this.TinyYolov4CocoExist
            	delete(fullfile(this.Yolov4DataDir,'YOLOv4-tiny-coco.mat'));
            end
        end
    end
end