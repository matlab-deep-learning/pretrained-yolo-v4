classdef sliceLayer < nnet.layer.Layer
%#codegen
% Custom layer used for channel grouping.

% Copyright 2021 The MathWorks, Inc.

    properties
        connectID
        groups 
        group_id 
    end
    
    methods
        function layer = sliceLayer(name,con,groups,group_id)
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            text = [ num2str(groups), ' groups,group_id: ', num2str(group_id), ' sliceLayer '];
            layer.Description = text;

            % Set layer type.
            layer.Type = 'sliceLayer';

            % Set other properties.
            layer.connectID= con;
            layer.groups= groups;
            layer.group_id= group_id;
            assert(group_id>0,'group_id must great zero! it must start index from 1');
        end
        
        function Z = predict(layer, X)
            X = reshape(X,[size(X),1]);
            channels = size(X,3);
            deltaChannels = channels/layer.groups;
            selectStart = (layer.group_id-1)*deltaChannels+1;
            selectEnd = layer.group_id*deltaChannels;
            Z = X(:,:,selectStart:selectEnd,:);
        end       
    end
end
