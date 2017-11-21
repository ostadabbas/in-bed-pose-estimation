function [heatMaps, prediction] = applyModel(test_image, param, rectangle)

%% Select model and other parameters from param
model = param.model(param.modelID);
boxsize = model.boxsize;    % 368 in config
np = model.np;  % what is np? 
nstage = model.stage;   % how many stages 
oriImg = imread(test_image);

%% Apply model, with searching thourgh a range of scales
octave = param.octave;
% set the center and roughly scale range (overwrite the config!) according to the rectangle
x_start = max(rectangle(1), 1); % the max is the whole image, not outside. 
x_end = min(rectangle(1)+rectangle(3), size(oriImg,2));
y_start = max(rectangle(2), 1);
y_end = min(rectangle(2)+rectangle(4), size(oriImg,1));
center = [(x_start + x_end)/2, (y_start + y_end)/2];    % bd box center 

% determine scale range
middle_range = (y_end - y_start) / size(oriImg,1) * 1.2;    % draw height/ image height /1.2
starting_range = middle_range * param.start_scale;  % from 0.8 0.8*1.2 =0.96
ending_range = middle_range * param.end_scale;  % 1.2*1.2 bounding from 0.96 to 1.44 actually larger bb
% range comes from i, that's the reason why vertical can't gives good
% result .

starting_scale = boxsize/(size(oriImg,1)*ending_range); % box 368  368/imageH* 1.2 
ending_scale = boxsize/(size(oriImg,1)*starting_range);
multiplier = 2.^(log2(starting_scale):(1/octave):log2(ending_scale));
% from start to end if increase 1 means the scale increase 2, and need 6
% octave. from 0.96 to 1.44 perhaps for the octave 6 step, only 3 steps 

% data container for each scale and stage
score = cell(nstage, length(multiplier)); % nSta * nScal 
pad = cell(1, length(multiplier));  % 1x nScale
ori_size = cell(1, length(multiplier));  %1 x nScale

net = caffe.Net(model.deployFile, model.caffemodel, 'test');
% caffe.Net(deployFileName, modelFileName, 'test);  

% change outputs to enable visualizing stagewise results
% note this is why we keep out own copy of m-files of caffe wrapper

colors = hsv(length(multiplier));   % nScale color map each scale a color
for m = 1:length(multiplier)
    scale = multiplier(m);
    
    imageToTest = imresize(oriImg, scale);  % Iout = scale * oriImg. first 0.96 smaller almost the boxSize
    ori_size{m} = size(imageToTest);    % larger image
    center_s = center * scale;  % scaled bdbox center make it no larger than 
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s, model.padValue); % into boxsize, which is multipler of 4
    % imageToTest is padded one. 
    % plot bbox indicating what actually goes into CPM
    % pad keep the up ,down left right margin size
    figure(1);
    pad_current = pad{m};
    x = [0-pad_current(2), size(oriImg,2)*scale + pad_current(4)]/scale;
    y = [0-pad_current(1), size(oriImg,1)*scale + pad_current(3)]/scale;
    plot([x(1) x(1) x(2) x(2) x(1)], [y(1) y(2) y(2) y(1) y(1)], 'Color', colors(m,:)); % draw bounding box 
    drawnow;     % updat figure window  on original bounding box, draw different colors 
    % figure(m+2); imshow(imageToTest);
    
    imageToTest = preprocess(imageToTest, 0.5, param);  % normalize to [-1/2,1/2], 4 channel gaussian center
    % box size 
    fprintf('Running FPROP for scale #%d/%d....', m, length(multiplier));
    tic;
    score(:,m) = applyDNN(imageToTest, net, nstage, np);    % processed image, cnnNet, how many stage, how many parts 
    time = toc;
    fprintf('done, elapsed time: %.3f sec\n', time);
    
    pool_time = size(imageToTest,1) / size(score{1,m},1); % stride-8
    % make heatmaps into the size of original image according to pad and scale
    % this part can be optimizied if needed
    score(:,m) = cellfun(@(x) imresize(x, pool_time), score(:,m), 'UniformOutput', false);  % resize the score to testImage size 
    score(:,m) = cellfun(@(x) resizeIntoScaledImg(x, pad{m}), score(:,m), 'UniformOutput', false); % padded part bg. not any joint
    score(:,m) = cellfun(@(x) imresize(x, [size(oriImg,2) size(oriImg,1)]), score(:,m), 'UniformOutput', false); % back to original
    
    %figure(m+2); imagesc(score{end,m}(:,:,1)');
    
end

%% summing the heatMaps results 
heatMaps = cell(1, nstage);
final_score = cell(1, nstage);
for s = 1:nstage
    final_score{s} = zeros(size(score{1,1}));
    for m = 1:size(score,2)
        final_score{s} = final_score{s} + score{s,m};
    end
    heatMaps{s} = permute(final_score{s}, [2 1 3]);
    heatMaps{s} = heatMaps{s} / size(score,2);
end

%% generate prediction from last-stage heatMaps (most refined)
prediction = zeros(np,2);
for j = 1:np
    [prediction(j,1), prediction(j,2)] = findMaximum(final_score{end}(:,:,j));
end


function img_out = preprocess(img, mean, param)
% mius mean and intensity norm to -1/2 to 1/2 I think.  
% change channel order, change x,y indexes. That's for opencv I think. 
% 4th dimension for the center 
    img_out = double(img)/256;  
    img_out = double(img_out) - mean;
    img_out = permute(img_out, [2 1 3]);    % why permute maybe saved in different color 
    
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.model(param.modelID).boxsize;   % 368 
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, param.model(param.modelID).sigma);
    % sigma is 21  [368 368] 184, 184, sigma
    % only a gaussian map located at center 
    img_out(:,:,4) = centerMapCell{1};  
    
function scores = applyDNN(images, net, nstage, np)
    input_data = {single(images)};  % put in a cell with single format 
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    net.forward(input_data);    % the image in there  the result will be in the net
    scores = cell(1, nstage);   % 1x6 6 stage 
    %     res = net.forward({data});    % tutorial use res to get them out 
    % prob = res{1};
    
    for s = 1:nstage
        string_to_search = sprintf('stage%d', s);
        blob_id_C = strfind(net.blob_names, string_to_search); % find stage i name indexes
        blob_id = find(not(cellfun('isempty', blob_id_C))); % k=[1 2 3] indeces not cell array
        blob_id = blob_id(end); % last one is outupt layer like state2 Mconv5 (last) 
        scores{s} = net.blob_vec(blob_id).get_data();   % each stage data
        if(size(scores{s}, 3) ~= np+1)      % the scores are less then 14+1 =15 what's the conv5_2_CPM
            string_to_search = 'conv5_2_CPM';   % check that's still 15 output MPII one, maybe the FLIC one has only 10 parts 
            blob_id_C = strfind(net.blob_names, string_to_search);
            blob_id = find(not(cellfun('isempty', blob_id_C)));
            blob_id = blob_id(end);
            scores{s} = net.blob_vec(blob_id).get_data();
        end
    end
    
function [img_padded, pad] = padAround(img, boxsize, center, padValue)
% pad keep the up ,down left right margin size
% repmat will map empty [] if the indices is negative. 
% std box measure the distance from the BB center to pad. 
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up   if box 184 - center or halfH =center
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue; % times 0 constant pad neg will be empty
    img_padded = [pad_up; img]; 
    pad_left = repmat(img_padded(:,1,:), [1 pad(2) 1])*0 + padValue;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:), [1 pad(4) 1])*0 + padValue;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];    % add left margin and up margin 

    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
    
function score = resizeIntoScaledImg(score, pad)
% get ride of the original  padding part, or simply add 0 or 1 to there 
    np = size(score,3)-1;   % only 14 parts, the other is background 
    score = permute(score, [2 1 3]);    % back to matlab format. 
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up put constant up, recover to same result 
    else
        score(1:pad(1),:,:) = []; % crop up remove the padded part 
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
    
function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));   % each row same x value, each column same y value
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);