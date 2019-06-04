% s_genOrtMdlAug
% generate orientation model with augmented data 
clear;
mkdir('models');
% dtMat = load('2endHogs.mat');
ftFileNm = '2endHogsAug0.5.mat';
% ftFileNm = '2endHogs';
mdlNm = 'ortMdl';
mdlFd = 'models';
if 7 ~= exist(mdlFd)
    mkdir(mdlFd)
end

if strfind(ftFileNm, 'Aug')
    if_aug = 1;
    mdlNm = strcat(mdlNm,'Aug');
else
    if_aug = 0;
end
% determine the shf part 
if strfind(ftFileNm, 'Shf');
    if_shf = 1;
    mdlNm = strcat(mdlNm, 'Shf');
else
    if_shf = 0; 
end
[fd, name, ext  ]= fileparts(ftFileNm);
scale =str2num(name(end-2:end));
mdlNm = sprintf(strcat(mdlNm, '%.1f.mat'), scale);
% ftMatNm = '2endHogsShf.mat';  % use the 
dtMat = load(fullfile('hgFts',ftFileNm));
% [fd, name, ext] = fileparts(dtMat);

fts_hogAll= dtMat.fts_hogAll;
labels_all = dtMat.labels_all;

ortMdlAug = fitcsvm(fts_hogAll,labels_all); 
ortMdlAugCross = crossval(ortMdlAug);
classLoss = kfoldLoss(ortMdlAugCross)   % default classification error 

fprintf('save modesl to %s\n', mdlNm);
save(fullfile(mdlFd, mdlNm), 'ortMdlAug','ortMdlAugCross');
% save model and cross validation model
