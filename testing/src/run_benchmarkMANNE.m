function prediction_file=run_benchmarkMANNE(param,benchmark_modelID,makeFigure, imgInds,fdName)
% give the model param and model ID, if make figure, and imgInds.
% save the prediciton and evaluation out. Prediction is for further
% evaluations.  BB, bodysize, imgInds, joint_gt_coord, predictions
% evaluations.
% file name rule: model_fdName
% Author: Shuangjun Liu

model = param.model(benchmark_modelID);
scale_search = 0.7:0.1:1.3; % fit training
boxsize = model.boxsize;    % 368 the input size
np = model.np;      % 14 body parts
obj = zeros(1,np);
detected = zeros(1,np);
thresh = 0.4 ;      % for BW BB detection
evaluation = [];
allHits = [];
correctR = [];
flgEdge=0;    % 1 edge 0 for threshold;
flgBdSz = 1;    % if use body part normal distance anyway, this only affect evaluation,
% we need prediction to re-evaluate. not very important
threshRt = 0.2; % the threshold ratio for the correct detection

fprintf('load inbed mat file....\n');

% get gt data transfer to 14x2xN format joints_gt
gt = load(fullfile('../dataset',fdName,'labelMANNE_GRAY_SC0_2.mat'));
% crafted to have only the indices joint.
try
    joints_gt = gt.joints_gt(1:2,:,imgInds);   % 2x14xN labels, 
catch
    error('index out of label matrix boundary');
end
%     joint_gt=joint_gt'; % trans to 14x2 coords
joints_gt = permute(joints_gt,[2,1,3]);
testLength = size(joints_gt,3); % Here adjust joint_gt_W according to your data
% third dimension nImgs
order_to_lsp = [14 13 9 8 7 10 11 12 3 2 1 4 5 6];
target_dist = 0.8;

net = caffe.Net(model.deployFile, model.caffemodel, 'test');
center_map = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, model.sigma);

fprintf('Running inference on %s using model %s, %d scales for each sample.\n', fdName, model.description, length(scale_search));
prediction_all = zeros(np, 2, testLength);
BBs = [];       % to save bounding box.
for i = 1:testLength
    fprintf('image %d/%d', i, testLength);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         imagePath = sprintf('../dataset/inbed/labeled_image_W/%06d.jpg', i);
    %         imagePath = sprintf('../dataset/inbed/labeled_image_W/%06d.jpg', imgInds(i)); % read in img inds
%     imagePath = sprintf('../dataset/MANNE_GRAY_SC0_2/images/%06d.jpg', imgInds(i)); % read in img inds
    imagePath = sprintf('../dataset/%s/images/%06d.jpg',fdName, imgInds(i)); % read in img inds

    try
        oriImg = imread(imagePath);
    catch
        error('image cannot be loaded, make sure you have %s', imagePath);
    end
    
    % Bound box with the edge method
    % edge based BB
    % Iedge = edge (rgb2gray(oriImg),'sobel',0.1);
    % bEd = GetBB(Iedge);
    
    % thresh based BB choose bb method
    if flgEdge
        Ibw = edge(rgb2gray(oriImg),'sobel',0.1);
    else
        Ibw = im2bw(oriImg,thresh);
    end
    bEd = GetBB(Ibw);
    
    BBs(i,:) = bEd;     %   bounding box save 
    oriImg = oriImg(bEd(2):bEd(2)+bEd(4)-1, bEd(1):bEd(1)+bEd(3)-1, :);
    
    center = [size(oriImg,2), size(oriImg,1)]/2;
    scale_provided = size(oriImg,1)/model.boxsize; % something prop to image height
    scale0 = target_dist/scale_provided;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         joint_gt(:,:,i) = gt.joint_gt_W(1:2,1:14,i)';% Here adjust joint_gt_W according to your data
    % cut data out
    multiplier = scale_search;  % 0.7:0.1:1.3
    score = cell(1, length(multiplier)); % {nScal} w, h, 14 for one single image
    pad = cell(1, length(multiplier));
    
    for m = 1:length(multiplier)
        scale = scale0 * multiplier(m);
        imageToTest = imresize(oriImg, scale);
        
        center_s = center * scale;
        [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s);   % pad square image from center
        
        imageToTest = preprocess(imageToTest, 0.5, center_map);
        
        score{m} = applyDNN(imageToTest, net);
        pool_time = size(imageToTest,1) / size(score{m},1);
        score{m} = imresize(score{m}, pool_time);
        score{m} = resizeIntoScaledImg(score{m}, pad{m});
        score{m} = imresize(score{m}, [size(oriImg,2), size(oriImg,1)]);
    end
    
    % summing up scores
    final_score = zeros(size(score{1,1}));
    for m = 1:size(score,2)
        final_score = final_score + score{m};   % all m score sum together 
    end
    final_score = permute(final_score, [2 1 3]); % to r,c for find purpose?
    % generate prediction
    prediction = zeros(np,2);
    for j = 1:np
        [prediction(j,2), prediction(j,1)] = findMaximum(final_score(:,:,j));
        prediction(j,:) = prediction(j,:) + [bEd(1) bEd(2)]; % tack to BB
    end
    
    prediction(order_to_lsp,:) = prediction;    % [x1,y1;x2,y2; ....];
    final_score(:,:,order_to_lsp) = final_score(:,:,1:np);
    allHits =[];
    correctR =[];
    
    bodysize(i) = util_get_bodysize_size(joints_gt(:,:,i)); %465.2068
    if flgBdSz  % if use body size
        normDis = bodysize(i);
    else
        normDis = max(bEd(4),bEb(3));
    end
    for j = 1:np
        if(makeFigure)
            max_value = max(max(final_score(:,:,j)));
            imToShow = single(oriImg)/255 * 0.5 + mat2im(final_score(:,:,j), jet(100), [0 max_value])/2;
            imToShow = insertShape(imToShow, 'FilledCircle', [prediction(j,:) 2], 'Color', 'w');
            imToShow = insertShape(imToShow, 'FilledCircle', [joints_gt(j,1:2,i) 2], 'Color', 'g');
            imToShow = insertShape(imToShow, 'FilledRectangle', [center 3 3], 'Color', 'c');
            
            figure(j); imshow(imToShow);
            title('paused, click to resume');
            name = sprintf('%d_mypredicts_%d.fig',i,j);
            saveas(figure(j),name);
            pause;
        end
        %             fprintf('joint_gt size');
        %             size(joint_gt)
        %             fprintf('i is %d',i);
        %prediction(j,:), % 956   2421
        %joint_gt(j,1:2,i), % 1026.9    1809.7
        
        error_dist = norm(prediction(j,:) - joints_gt(j,1:2,i));
        %             hit = error_dist <= bodysize*0.2; % bodypart standard
%         hit = error_dist<= normDis*0.2;  % the 0.2 bg
        hit = error_dist<= threshRt*normDis;
        
        obj(j) = obj(j) + 1;    % each body part
        if(hit)
            detected(j) = detected(j) + 1;
        end
        fprintf(' %d', hit);    % display result on screen
        allHits = [allHits hit];    % [1 1 0 0 ...1] how many hits
    end
    
    for j = 1:np
        fprintf(' %.3f', detected(j)/obj(j));
        correctR = [correctR detected(j)/obj(j)];   % correct rate [0.7, 0.8...]
    end
    fprintf(' |%.4f\n', sum(detected)/sum(obj));
    sumOfcorr = sum(detected)/sum(obj); % all cc rate
    
    prediction_all(:,:,i) = prediction;
    evaluation(i,:) = [allHits correctR sumOfcorr]; % [1 14 1] 16
end

% prediction_file = sprintf('predicts/inbed_prediction_model_%s.mat', model.description_short);
% save(prediction_file, 'prediction_all','BBs','imgInds');    % 14x3xN joints, save BBs for further detection
% 
% evaluation_file = sprintf('predicts/inbed_evaluation_model_%s.mat', model.description_short);
% save(evaluation_file, 'evaluation');    % hit corr_each .... corr_all
% new name rule 

joints_gt_coord = joints_gt(1:14,:,:);  % 14x2xN;
prediction_file = sprintf('predicts/prediction_%s_%s.mat', model.description_short,fdName);
save(prediction_file, 'prediction_all','BBs','imgInds','bodysize','joints_gt_coord');    % 14x3xN joints, save BBs for further detection

evaluation_file = sprintf('predicts/evaluation_%s_%s.mat', model.description_short,fdName);
save(evaluation_file, 'evaluation');    % hit corr_each .... corr_all


function img_out = preprocess(img, mean, center_map)
    img = double(img) / 256;
    
    img_out = double(img) - mean;
    img_out = permute(img_out, [2 1 3]);
    img_out = img_out(:,:,[3 2 1]);
    img_out(:,:,4) = center_map{1};
    
function scores = applyDNN(images, net)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    scores = net.forward(input_data);
    scores = scores{1};
    
function [img_padded, pad] = padAround(img, boxsize, center)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:)*0, [pad(1) 1 1])+128;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:)*0, [1 pad(2) 1])+128;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:)*0, [pad(3) 1 1])+128;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:)*0, [1 pad(4) 1])+128;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];
    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
    
function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    score = permute(score, [2 1 3]);
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
    score = permute(score, [2 1 3]);

% function headSize = util_get_head_size(rect)
%     SC_BIAS = 0.6; % 0.8*0.75
%     headSize = SC_BIAS * norm([rect.x2 rect.y2] - [rect.x1 rect.y1]);
    
function bodysize = util_get_bodysize_size(rect)
    bodysize = norm(rect(10,:) - rect(3,:)); % following evalLSP_official

function label = produceCenterLabelMap(im_size, x, y, sigma) %this function is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);
