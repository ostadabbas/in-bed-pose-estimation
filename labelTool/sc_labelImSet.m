% sc_genImgSet
% call function to label image set 
% imSetFd = 'S:\ACLab\datasets_prep\ac2d_00_lbTest';
% LabelImSet(imSetFd);
clc;
% label the PM IR data 
% s drive version 
% dsFd = 'S:\ACLab\datasetPM\simLab\00003';
ifMac = 0;
dsFd = 'S:\ACLab\datasetPM\simLab\00007test';
align_PTr_IR = readNPY(fullfile(dsFd,'align_PTr_IR.npy'));
align_PTr_RGB = readNPY(fullfile(dsFd,'align_PTr_RGB.npy'));
align_PTr_IR2RGB = inv(align_PTr_RGB) * align_PTr_IR;
align_PTr_RGB2IR = inv(align_PTr_IR)* align_PTr_RGB;

if ~ifMac
    subFd = 'RGB\uncover';
else
    subFd = 'RGB/uncover';  % for mac version 
end
% local temp version 
% imSetFd = 'E:\dataset\dsPM_temp';
% subFd = 'uncover';
angRt = 0; % clock wise 
imgFmt = 'png';
% tnorm = maketform('projective',align_PTr_RGB2IR');
tform = maketform('projective',align_PTr_IR2RGB');

if ~ifMac
    subFd_tar = 'IR\uncover';
else
    subFd_tar = 'IR/uncover';
end


% LabelImSet_map(dsFd, subFd,subFd_tar, tnorm, angRt,imgFmt );
% LabelImSet_map(dsFd, subFd_tar, subFd, tform, angRt,imgFmt );
LabelImSet(dsFd, subFd, angRt,imgFmt);









