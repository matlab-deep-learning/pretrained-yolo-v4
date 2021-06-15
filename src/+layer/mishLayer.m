classdef mishLayer < nnet.layer.Layer
%#codegen
% Custom layer for Mish activation function. 

% Copyright 2021 The MathWorks, Inc.

    methods
        function layer = mishLayer(name)
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "mish activation layer";
            
            % Set layer type.
            layer.Type = 'mishLayer';
        end
        
        function Z = predict(~, X)
            Z1 = max(X,0) + log(1 + exp(-abs(X)));
            Z = X.*tanh(Z1);
        end
    end
end
