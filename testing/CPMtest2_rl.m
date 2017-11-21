%% demo of CPM on Mannequin in-bed pose
close all;
addpath('src'); 
addpath('util');
addpath('util/ojwoodford-export_fig-5735e6d/');
param = config1();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);


% test_image = 'sample_image/t2N.jpg';
test_image = 'sample_image/WIN_20170331_18_18_28_Pro.jpg';

interestPart = 'Lwri'; % to look across stages. check available names in config.m

flgBBdetection = 1;

I = imread(test_image);
[m,n,c]  =size(I); % xmin ymin width height 
rectangle = [1,1,n-1,m-1]; % just give the whole image 
Igray = rgb2gray(I);
I_RGB = cat(3,Igray,Igray,Igray);

figure(1);
imshow(I);
hold on;
if flgBBdetection;
	Ibw = im2bw(I);
	rectangle = GetBB(Ibw);
else 
	rectangles = [1,1,n-1,m-1]; % just give the whole image 
end
tic;
[heatMaps, prediction] = applyModelIm(I_RGB, param, rectangle);
elap(1) = toc;
display(['prediction time cost is',num2str(elap(1))]);

%% visualize, or extract variable heatMaps & prediction for your use
visualize(test_image, heatMaps, prediction, param, rectangle, interestPart);
elap(2)=toc;
display(['prediction with visulization time cost is', num2str(elap(2))]);

