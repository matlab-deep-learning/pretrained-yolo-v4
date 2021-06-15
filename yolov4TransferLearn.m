%% Transfer Learning Using Pretrained YOLO v4 Network
% The following code demonstrates how to perform transfer learning using the 
% pretrained YOLO v4 network for object detection. This script uses the 
% "configureYOLOv4" function to create a custom YOLO v4 network using the 
% pretrained model.

%% Setup
% Add path to the source directory.
addpath('src');

%% Download Pretrained Network
% This repository uses two variants of YOLO v4 models.
% *YOLOv4-coco*
% *YOLOv4-tiny-coco*
% Set the modelName from the above ones to download that pretrained model.
modelName = 'YOLOv4-coco';
model = helper.downloadPretrainedYOLOv4(modelName);
net = model.net;

%% Load Data
% This example uses a small labeled data set that contains 295 images. Many 
% of these images come from the Caltech Cars 1999 and 2001 data sets, available 
% at the Caltech Computational Vision <http://www.vision.caltech.edu/archive.html 
% website>, created by Pietro Perona and used with permission. Each image contains 
% one or two labeled instances of a vehicle. A small data set is useful for exploring 
% the YOLO v4 training procedure, but in practice, more labeled images are needed 
% to train a robust network.
% 
% Unzip the vehicle images and load the vehicle ground truth data. 
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% Add the full path to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd, vehicleDataset.imageFilename);

% *Note:* In case of multiple classes, the data can also organized as three 
% columns where the first column contains the image file names with paths, the 
% second column contains the bounding boxes and the third column must be a cell 
% vector that contains the label names corresponding to each bounding box.
% 
% All the bounding boxes must be in the form |[x y width height]|. This vector 
% specifies the upper left corner and the size of the bounding box in pixels.
% 
% Split the data set into a training set for training the network, and a test 
% set for evaluating the network. Use 60% of the data for training set and the 
% rest for the test set.
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx), :);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end), :);

% Create an image datastore for loading the images.
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);
 
% Create a datastore for the ground truth bounding boxes.
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

% Combine the image and box label datastores.
trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

% Use validateInputData to detect invalid images, bounding boxes or labels 
% i.e., 
%
% * Samples with invalid image format or containing NaNs
% * Bounding boxes containing zeros/NaNs/Infs/empty
% * Missing/non-categorical labels. 
%
% The values of the bounding boxes should be finite, positive, non-fractional, 
% non-NaN and should be within the image boundary with a positive height and width. 
% Any invalid samples must either be discarded or fixed for proper training.
helper.validateInputData(trainingData);
helper.validateInputData(testData);

%% Data Augmentation
% Data augmentation is used to improve network accuracy by randomly transforming 
% the original data during training. By using data augmentation, you can add more 
% variety to the training data without actually having to increase the number 
% of labeled training samples.
% 
% Use transform function to apply custom data augmentations to the training 
% data. The augmentData helper function, listed at the end of the example, applies 
% the following augmentations to the input data.
%
% * Color jitter augmentation in HSV space
% * Random horizontal flip
% * Random scaling by 10 percent
augmentedTrainingData = transform(trainingData, @helper.augmentData);
 
% Read the same image four times and display the augmented training data.

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)

%% Preprocess Training Data
% Specify the network input size. 
networkInputSize = net.Layers(1).InputSize;
 
% Preprocess the augmented training data to prepare for training. The preprocessData 
% helper function, listed at the end of the example, applies the following preprocessing 
% operations to the input data.
% 
% * Resize the images to the network input size
% * Scale the image pixels in the range |[0 1]|.
preprocessedTrainingData = transform(augmentedTrainingData, @(data)helper.preprocessData(data, networkInputSize));
 
% Read the preprocessed training data.
data = read(preprocessedTrainingData);

% Display the image with the bounding boxes.
I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Reset the datastore.
reset(preprocessedTrainingData);

%% Modify Pretrained YOLO v4 Network
% The YOLO v4 network uses anchor boxes estimated using training data to have 
% better initial priors corresponding to the type of data set and to help the 
% network learn to predict the boxes accurately.
% 
% First, use transform to preprocess the training data for computing the 
% anchor boxes, as the training images used in this example vary in size. 
%
% Specify the number of anchors as follows:
% 'YOLOv4-coco' model      - 9
% 'YOLOv4-tiny-coco' model - 6 
%
% Use the estimateAnchorBoxes function to estimate the anchor boxes. Note 
% that the estimation process is not deterministic. To prevent the estimated 
% anchor boxes from changing while tuning other hyperparameters set the 
% random seed prior to estimation using rng.
%
% Then pass the estimated anchor boxes to configureYOLOv4 function to arrange 
% them in correct order to be used in the training.
rng(0)
trainingDataForEstimation = transform(trainingData, @(data)helper.preprocessData(data, networkInputSize));
numAnchors = 9;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

% Specify the classNames to be used in the training.
classNames = {'vehicle'};

% Configure the pretrained model for transfer learning using configureYOLOv4
% function. This function will return the modified layergraph, networkoutput 
% names, reordered anchorBoxes and anchorBoxMasks to select anchor boxes to 
% use in the detection heads.
[lgraph, networkOutputs, anchorBoxes, anchorBoxMasks] = configureYOLOv4(net, classNames, anchorBoxes, modelName);

%% Specify Training Options
% Specify these training options.
% 
% * Set the number of epochs to be 90.
% * Set the mini batch size as 4. Stable training can be possible with higher 
% learning rates when higher mini batch size is used. Although, this should 
% be set depending on the available memory.
% * Set the learning rate to 0.001. 
% * Set the warmup period as 1000 iterations. It helps in stabilizing the 
% gradients at higher learning rates.
% * Set the L2 regularization factor to 0.0005.
% * Specify the penalty threshold as 0.5. Detections that overlap less than 
% 0.5 with the ground truth are penalized.
% * Initialize the velocity of gradient as []. This is used by SGDM to store 
% the velocity of gradients.
numEpochs = 90;
miniBatchSize = 4;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.001;
penaltyThreshold = 0.5;
velocity = [];

%% Train Model
% Train on a GPU, if one is available. Using a GPU requires Parallel Computing 
% Toolbox™ and a CUDA® enabled NVIDIA® GPU.
% 
% Use the minibatchqueue function to split the preprocessed training data 
% into batches with the supporting function createBatchData which returns the 
% batched images and bounding boxes combined with the respective class IDs. 
% For faster extraction of the batch data for training, dispatchInBackground 
% should be set to "true" which ensures the usage of parallel pool.
% 
% minibatchqueue automatically detects the availability of a GPU. If you do 
% not have a GPU, or do not want to use one for training, set the OutputEnvironment 
% parameter to "cpu". 
if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

mbqTrain = minibatchqueue(preprocessedTrainingData, 2,...
        "MiniBatchSize", miniBatchSize,...
        "MiniBatchFcn", @(images, boxes, labels) helper.createBatchData(images, boxes, labels, classNames), ...
        "MiniBatchFormat", ["SSCB", ""],...
        "DispatchInBackground", dispatchInBackground,...
        "OutputCast", ["", "double"]);

% To train the network with a custom training loop and enable automatic differentiation, 
% convert the layer graph to a dlnetwork object. Then create the training progress 
% plotter using supporting function configureTrainingProgressPlotter. 
% 
% Finally, specify the custom training loop. For each iteration:
% 
% * Read data from the minibatchqueue. If it doesn't have any more data, reset 
% the minibatchqueue and shuffle. 
% * Evaluate the model gradients using dlfeval and the modelGradients function. 
% The function modelGradients, listed as a supporting function, returns the 
% gradients of the loss with respect to the learnable parameters in net, the 
% corresponding mini-batch loss, and the state of the current batch.
% * Apply a weight decay factor to the gradients to regularization for more 
% robust training.
% * Determine the learning rate based on the iterations using the 
% piecewiseLearningRateWithWarmup supporting function.
% * Update the network parameters using the sgdmupdate function.
% * Update the state parameters of net with the moving average.
% * Display the learning rate, total loss, and the individual losses (box loss, 
% object loss and class loss) for every iteration. These can be used to interpret 
% how the respective losses are changing in each iteration. For example, a sudden 
% spike in the box loss after few iterations implies that there are Inf or NaNs 
% in the predictions.
% * Update the training progress plot.  

% The training can also be terminated if the loss has saturated for few epochs. 

% Convert layer graph to dlnetwork.
net = dlnetwork(lgraph);

% Create subplots for the learning rate and mini-batch loss.
fig = figure;
[lossPlotter, learningRatePlotter] = helper.configureTrainingProgressPlotter(fig);

iteration = 0;
% Custom training loop.
for epoch = 1:numEpochs
      
    reset(mbqTrain);
    shuffle(mbqTrain);
    
    while(hasdata(mbqTrain))
        iteration = iteration + 1;
       
        [XTrain, YTrain] = next(mbqTrain);
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients, state, lossInfo] = dlfeval(@modelGradients, net, XTrain, YTrain, anchorBoxes, anchorBoxMasks, penaltyThreshold, networkOutputs);

        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);

        % Determine the current learning rate value.
        currentLR = helper.piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
        
        % Update the network learnable parameters using the SGDM optimizer.
        [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);

        % Update the state parameters of dlnetwork.
        net.State = state;
        
        % Display progress.
        if mod(iteration,10)==1
            helper.displayLossInfo(epoch, iteration, currentLR, lossInfo);
        end
            
        % Update training plot with new points.
        helper.updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
    end
end

% Save the trained model with the anchors.
anchors.anchorBoxes = anchorBoxes;
anchors.anchorBoxMasks = anchorBoxMasks;

save('yolov4_trained', 'net', 'anchors');

%% Evaluate Model
% Computer Vision System Toolbox™ provides object detector evaluation functions 
% to measure common metrics such as average precision (evaluateDetectionPrecision) 
% and log-average miss rates (evaluateDetectionMissRate). In this example, the 
% average precision metric is used. The average precision provides a single number 
% that incorporates the ability of the detector to make correct classifications 
% (precision) and the ability of the detector to find all relevant objects (recall).
% 
% Following these steps to evaluate the trained dlnetwork object net on test data.
%
% * Specify the confidence threshold as 0.5 to keep only detections with confidence 
% scores above this value.
% * Specify the overlap threshold as 0.5 to remove overlapping detections.
% * Apply the same preprocessing transform to the test data as for the training 
% data. Note that data augmentation is not applied to the test data. Test data 
% must be representative of the original data and be left unmodified for unbiased 
% evaluation.
% * Collect the detection results by running the detector on testData. 
% Use the supporting function detectYOLOv4 to get the bounding boxes, object 
% confidence scores, and class labels.
% * Call evaluateDetectionPrecision with predicted results and preprocessedTestData 
% as arguments. 
confidenceThreshold = 0.5;
overlapThreshold = 0.5;

% Create a table to hold the bounding boxes, scores, and labels returned by
% the detector. 
numImages = size(testDataTbl, 1);
results = table('Size', [0 3], ...
    'VariableTypes', {'cell','cell','cell'}, ...
    'VariableNames', {'Boxes','Scores','Labels'});

% Run detector on images in the test set and collect results.
reset(testData)
while hasdata(testData)
    % Read the datastore and get the image.
    data = read(testData);
    image = data{1};
    
    % Run the detector.
    executionEnvironment = 'auto';
    [bboxes, scores, labels] = detectYOLOv4(net, image, anchors, classNames, executionEnvironment);
    
    % Collect the results.
    tbl = table({bboxes}, {scores}, {labels}, 'VariableNames', {'Boxes','Scores','Labels'});
    results = [results; tbl];
end

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, testData);

% The precision-recall (PR) curve shows how precise a detector is at varying 
% levels of recall. Ideally, the precision is 1 at all recall levels.

% Plot precision-recall curve.
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

%% Detect Objects Using Trained YOLO v4
% Use the network for object detection.
%
% * Read an image.
% * Convert the image to a dlarray and use a GPU if one is available..
% * Use the supporting function detectYOLOv4 to get the predicted bounding 
% boxes, confidence scores, and class labels. 
% * Display the image with bounding boxes and confidence scores.

% Read the datastore.
reset(testData)
data = read(testData);

% Get the image.
I = data{1};

% Run the detector.
executionEnvironment = 'auto';
[bboxes, scores, labels] = detectYOLOv4(net, I, anchors, classNames, executionEnvironment);

% Display the detections on image.
if ~isempty(scores)
    I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
end
figure
imshow(I)


%% References
% 1. Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "YOLOv4: 
% Optimal Speed and Accuracy of Object Detection." arXiv preprint arXiv:2004.10934 
% (2020).
% 
% Copyright 2021 The MathWorks, Inc.
