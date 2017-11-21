close all;
addpath('src'); 
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
% param = config();
param = config1();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
% use 6 for mannequin 

% imgInds
% imgInds = [270:300, 401:419];   % the images for test
imgInds = 1:419;    % total set. But we would better not use this one 
% imgInds = 271:280;   % small set test. 
% modelInds = [1,6:9];

modelInds = [6];
fdName= 'BW3SC0_2_2017.3.31N';    % last character can be N,E,S,W
% only test the Nth direction 
for i = 1:length(modelInds);
% benchmark_modelID = 1;
    benchmark_modelID = modelInds(i);
    makeFigure = 0;
%     prediction_file = run_benchmark2(param, 'LSP', benchmark_modelID, makeFigure,imgInds);
    prediction_fileName = run_benchmarkMANNE(param,benchmark_modelID,makeFigure, imgInds,fdName);

    fprintf('model %d evaluated',i);
end

