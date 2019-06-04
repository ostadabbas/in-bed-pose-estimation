% s_gen2endHogFts
% generate and save the two end HOG features with 4 additional rotation
% augmentation. Original not in a dataset format, save it dsFmt X N x m,  Y
% N x 1 
% save to all hogs to ort Data only the idxsTrn, no test in here

clc;clear; 
imgRt = 'S:\ACLab\datasets_prep\BW3MltSC0_2_170331';  
imgRtTest = 'S:\ACLab\datasets_prep\BW3SC0_2_20170331_rndOrt'
imgSet =imageSet( imgRt, 'recursive');  % 1x4 struct
rstFd = 'hgFts'
if 7 ~= exist(rstFd)
    mkdir(rstFd)
end

% parameters 
if_aug = 1;     % if augment the data with random rotation
if_shf  = 0;
scale = 0.5;        % original 0.1 too small I think 
flgSave = 1;
if_show = 0;

% E,N,S,W
ortInd  = [2,3];   % only test north and south
imgBB ={};   % stdx,stdy,w,h, X imgInd X {nCategories}
whRt = {}; % nIm x {nCat}

hog2x2Fts = {}; % {nCat}x nIm x nFt
ortLabel = '';

sz_batch = 5;       % each batch 5 including one original, 4 augmentated 
n_aug = 4; 
augDeg = 15;        % 10 degress augmentations 
nImgs = 419 ;
nTrn = 370; 
% idxTrn = [1:270, 301:400];    % hard seg 
% idxTrn = 1:419;     % total 419 images 
if if_shf   % shuffled segmentation
    if exist(fullfile(imgRtTest, 'idxsSeg.mat'))    % shuffled version 
        dtIn = load(fullfile(imgRtTest, 'idxsSeg.mat')); % read in test index
        idxsTrn = dtIn.idxsTrn;
        idxsTest = dtIn.idxsTest;
    else
        fprintf('no train test segmentation exists, random generate one\n');
        p = randperm(nImgs);
        idxsTrn = p(1:370);
        idxsTest = p(371:419);
        save(fullfile(imgRtTest, 'idxsSeg'), 'idxsTrn','idxsTest');
    end
else
   idxsTrn = [1:270, 301:400];   % hard seg 
end
fts_hogAll = []; 
labels_all = []; 

for i = 1:length(ortInd)    % uncomment for iteration
    imgBBCatTemp = [];
    whRtCatTemp =[];
%     for j = 1:imgSet(drctInd(i)).Count  % idxImg 
    fts_hogTmp = [];
    labels_tmp = []; 
    if 1 == i 
%         label = 'y';  % north is 1  
        label = 1 ;
    else 
%         label = 'n' ;
        label = 0;
    end
    for j = 1:length(idxsTrn)
        orts= [0, augDeg*(rand(1,n_aug)*2-1)];  % each time new rotation
        I = imresize(read(imgSet(ortInd(i)),idxsTrn(j)),scale);   % resized to scale
        [I_pad, stR, endR, stC, endC ] = padImgSqrt(I);
        if if_aug == 1
            idx_augEnd = length(orts);
        else
            idx_augEnd = 1;
        end
        for k =1:idx_augEnd
            fprintf('processing ds %d, frame %d, aug %d \n', i, j, k);
            if ~if_aug
                idx_ent = j;
            else
                idx_ent = (j-1)*sz_batch + k ;
            end
            I_rtd = imrotate(I_pad, orts(k),'crop'); 
%             Ithresh = 0.4 * max(I_rtd(:))/max(I_rtd(:));   % 40% ratio threshold
            Ibw= im2bw(I_rtd,0.4);
            % method 2 the edge method
            % Iedge = edge (rgb2gray(I),'sobel',0.1);
    %         if flagPlot && 0
    %             figure(2);imshow(Iedge);
    %         end
    %         Ibw= Iedge;
        
            [validR,validC]  = find (Ibw);
            startX = min(validC);
            startY = min(validR);
            width = max(validC)- startX+1;
            height =max(validR)-startY+1;
            imgBBCatTemp(idx_ent,:) = [startX,startY,width,height];
            whRtCatTemp(idx_ent) = double(width)/height;
            keyCentTemp = GetVnCenters(imgBBCatTemp(idx_ent,:),width,2);    % bg found 
            % hog features at given points
            [hog1,valid, ptVis] = extractHOGFeatures(I_pad,keyCentTemp,'CellSize',round([width/2 width/2]));
            hogT = hog1';       % empty!!! 
            fts_hogTmp = [ fts_hogTmp; transpose(hogT(:))];   % add new underearth 
            labels_tmp = [labels_tmp; label];   % keep add predefined label
            % show the bounding box
            if if_show
                figure(3); cla;imshow(I_pad);
                hold on;
                %         rectangle ('Position',imgBBCatTemp(j,:),'EdgeColor','g');
                plot(keyCentTemp(:,1),keyCentTemp(:,2),'y*');
                plot(ptVis,'color','g');
            end
        end
    end
    fts_hogAll= [fts_hogAll; fts_hogTmp]; 
    labels_all = [labels_all; labels_tmp]; 
%     ortLabel(i) = imgSet(ortInd(i)).Description;    % cell is better options, description problems 
end

% save all the data
fileNm = '2endHogs';
if if_aug
    fileNm = [fileNm, 'Aug'];
end
if if_shf 
    fileNm = [fileNm, 'Shf'];
end
if flgSave

    svNm = sprintf('%s%.1f.mat',fileNm,scale);
    fprintf('save all features and labels to %s\n', svNm);  
%     save(sprintf('%s%.1f.mat',fileNm,scale), 'fts_hogAll', 'labels_all');
    save(fullfile(rstFd,svNm), 'fts_hogAll', 'labels_all');
end
