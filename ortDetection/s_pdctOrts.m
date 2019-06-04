% s_pdctOrts 
% read in image, find minimal bb, find orientation, make it vertical, 
% keep the rotation ,
% then classify the updown with our method,  if upside down, rotate another
% 180, the predicted orientation is -rt, cycle back to -180 to 180 

imgRtTest = 'S:\ACLab\datasets_prep\BW3SC0_2_20170331_rndOrt';
mdlFd = 'models';
mdlNm = 'ortMdlAug0.5.mat';
% mdlNm = 'ortMdl0.5.mat';
[fd, name, ext  ]= fileparts(mdlNm);
scale =str2num(name(end-2:end));
if_svImg = 0;
if_showImg = 0; 
% if_shfVer = 1; 
if contains(mdlNm, 'Shf')
    if_shfVer = 1;
else
    if_shfVer =0;
end

dtIn = load(fullfile(imgRtTest, 'ortVec.mat'));
ortVec = dtIn.ortVec;   % gt rotations 
gt = zeros(size(ortVec));    
gt(abs(ortVec)<90)=1;   
degs_rt = [];
if 7 ~= exist('rstImgs')
    mkdir('rstImgs');   % to save the result images         
end

% idxs_test = [271:300, 401:419]; % for test parts  % hard seg 

if if_shfVer == 1
    dtIn = load(fullfile(imgRtTest, 'idxsSeg.mat'));  % shuffled selected version
    idxs_test = dtIn.idxsTest;   % get in test index 
%     mdlNm = 'ortMdlAugShf.mat';
else
    idxs_test = [271:300, 401:419];
%     mdlNm = 'ortMdlAug.mat';
end

% exist('models/ortMldAug.mat')
% dtIn = load('models\ortMdlAugShf.mat'); % shuffle version mdl
dtIn = load(fullfile(mdlFd, mdlNm));
ortMdlAug = dtIn.ortMdlAug;
idx_sgl_test = 10;

fts_test = [] ;
if if_svImg
    idx_end = 10;
else
    idx_end = length(idxs_test);
end
for i =1:idx_end % length(idxs_test)
    I = imread(fullfile(imgRtTest, 'images', sprintf('%06d.jpg', idxs_test(i)))); % padded already must minimize it!
    I_scal = imresize(I, scale);
    fprintf('processing image %d in test segment\n', idxs_test(i));
%     imshow(I);
    % find bw, get minBB, get nearest orientation (-90 to 90), counter
    I_bw = im2bw(I_scal,0.4);    
    [idxs_r, idxs_c ] = find(I_bw);
    bb = minBoundingBox([idxs_c, idxs_r ]');
%     plot(bb(1,[1:end 1]),bb(2,[1:end 1]),'g')
    deg_rt = getBBrtFromV(bb);
    % rotate it rt 
    I_scal_rtd = imrotate(I_scal, -deg_rt, 'crop');
    I_rtd = imrotate(I, -deg_rt, 'crop');  % raw rotated image
    bb_rot = transpose(GetCoordImRot(bb', size(I_scal), -deg_rt));
   if if_showImg
    figure(1); imshow(I); hold on; 
    plot(bb(1,[1:end 1]),bb(2,[1:end 1]),'g');
    figure(2); imshow(I_rtd);hold on; 
    plot(bb_rot(1,[1:end 1]),bb_rot(2,[1:end 1]),'g');
    pause();
   end
   if i < 10 && if_svImg
       I_bwRaw = im2bw(I,0.4);  
       [idxs_r, idxs_c ] = find(I_bwRaw);
       bb = minBoundingBox([idxs_c, idxs_r ]');
%      plot(bb(1,[1:end 1]),bb(2,[1:end 1]),'g')
       deg_rt = getBBrtFromV(bb);
       bb_rot = transpose(GetCoordImRot(bb', size(I), -deg_rt));
      
       
        figure(1); imshow(I,'Border','tight');hold on;         
         plot(bb(1,[1:end 1]),bb(2,[1:end 1]),'g');
%          rectangle('Position',bTh,'EdgeColor','g');
       set(gca,'position',[0 0 1 1],'units','normalized')
        figure(2); imshow(I_rtd,'Border','tight');hold on;    
        plot(bb_rot(1,[1:end 1]),bb_rot(2,[1:end 1]),'g');
%          rectangle('Position',bTh,'EdgeColor','g');
       set(gca,'position',[0 0 1 1],'units','normalized')
             
       print(figure(1),'-r600','-dpng',sprintf('rstImgs/ori%06d', idxs_test(i)));
       print(figure(2),'-r600','-dpng',sprintf('rstImgs/rtd%06d', idxs_test(i)));
    
   end
   % get 2end Hog fts 
%    I_rtd = imrotate(I_rtd,180, 'crop');
   ft_hog = getNendHog(I_scal_rtd);
   fts_test(i,:) = ft_hog';
%    [label, score] = predict(ortMdlAug, ft_hog')   
%    pause();
    % find upside down, rotate another 180 
end 

if ~if_svImg
    [pdcts, scores]=  predict(ortMdlAug, fts_test);
    compares = [gt(idxs_test)', pdcts];
    confMat= confusionmat(gt(idxs_test), pdcts)
end
% confusionchart(confMat);
