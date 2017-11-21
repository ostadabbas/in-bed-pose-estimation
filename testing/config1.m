function param = config1()
%% set this part
% 6 to 9 all our test result. 

% CPU mode (0) or GPU mode (1)
% friendly warning: CPU mode may take a while
param.use_gpu = 1;

% GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0;

% Select model (default: 1)
% 1: MPII+LSP(PC) 6-stage CPM
% 2: MPII 6-stage CPM
% 3: LSP(PC) 6-stage CPM
% 4: FLIC 4-stage CPM (upper body only)
% 5: MPII 6-stage CPM VGG-pretrained
% 6: MANNE_GRAY AS C3S2 fine tuning 2000
% 7: MANNE_GRAY S1 S6 
% 8: MANNE_GRAY all S
% 9: MANNE_GRAY AS C3S2 it 200
% 10: S6 

param.modelID = 6;

% Scaling paramter: starting and ending ratio of person height to image
% height, and number of scales per octave
% warning: setting too small starting value on non-click mode will take
% large memory
param.octave = 6;
param.start_scale = 0.8;
param.end_scale = 1.2;


% Path of caffe. You can change to your own caffe just for testing
% caffepath = '../caffe/matlab/'; % set the desired command line
caffepath = '/shared/apps/caffe_sep_2015/caffe-master/matlab/';
%caffepath = textread('../caffePath.cfg', '%s', 'whitespace', '\n\t\b ');
%caffepath= [caffepath{1} '/matlab/'];
fprintf('You set your caffe in caffePath.cfg at: %s\n', caffepath);
addpath(caffepath);
caffe.reset_all();      % how can matlab know caffe at this point 
if(param.use_gpu)
    fprintf('Setting to GPU mode, using device ID %d\n', GPUdeviceNumber);
    caffe.set_mode_gpu();
    caffe.set_device(GPUdeviceNumber);
else
    fprintf('Setting to CPU mode.\n');
    caffe.set_mode_cpu();
end


%% don't edit this part
param.click = 1;

param.model(1).caffemodel = '../model/_trained_MPI/pose_iter_985000_addLEEDS.caffemodel';
param.model(1).deployFile = '../model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(1).description = 'MPII+LSP 6-stage CPM';
param.model(1).description_short = 'MPII_LSP_6s';
param.model(1).boxsize = 368;
param.model(1).padValue = 128;
param.model(1).np = 14;
param.model(1).sigma = 21;
param.model(1).stage = 6;
param.model(1).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(1).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(2).caffemodel = '../model/_trained_MPI/pose_iter_630000.caffemodel';
param.model(2).deployFile = '../model/_trained_MPI/pose_deploy_centerMap.prototxt';
param.model(2).description = 'MPII 6-stage CPM';
param.model(2).description_short = 'MPII_6s';
param.model(2).boxsize = 368;
param.model(2).padValue = 128;
param.model(2).np = 14;
param.model(2).sigma = 21;
param.model(2).stage = 6;
param.model(2).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(2).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(3).caffemodel = '../model/_trained_LEEDS_PC/pose_iter_395000.caffemodel';
param.model(3).deployFile = '../model/_trained_LEEDS_PC/pose_deploy_centerMap.prototxt';
param.model(3).description = 'LSP (PC) 6-stage CPM';
param.model(3).description_short = 'LSP_6s';
param.model(3).boxsize = 368;
param.model(3).np = 14;
param.model(3).sigma = 21;
param.model(3).stage = 6;
param.model(3).padValue = 128;
param.model(3).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(3).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(4).caffemodel = '../model/_trained_FLIC/pose_iter_40000.caffemodel';
param.model(4).deployFile = '../model/_trained_FLIC/pose_deploy.prototxt';
param.model(4).description = 'FLIC (upper body only) 4-stage CPM';
param.model(4).description_short = 'FLIC_4s';
param.model(4).boxsize = 368;
param.model(4).np = 9;
param.model(4).sigma = 21;
param.model(4).stage = 4;
param.model(4).padValue = 128;
param.model(4).limbs = [1 2; 2 3; 4 5; 5 6];
param.model(4).part_str = {'Lsho', 'Lelb', 'Lwri', ...
                           'Rsho', 'Relb', 'Rwri', ...
                           'Lhip', 'Rhip', 'head', 'bkg'};

param.model(5).caffemodel = '../model/_trained_MPI/pose_iter_320000.caffemodel';
param.model(5).deployFile = '../model/_trained_MPI/pose_deploy_resize.prototxt';
param.model(5).description = 'MPII 6-stage CPM';
param.model(5).description_short = 'MPII_VGG_6s';
param.model(5).boxsize = 368;
param.model(5).padValue = 128;
param.model(5).np = 14;
param.model(5).sigma = 21;
param.model(5).stage = 6;
param.model(5).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(5).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};

param.model(6).caffemodel = '../model/MANNE_GRAY_SC0_2/caffemodel/pose_iter_2000AS_C3S2.caffemodel';
param.model(6).deployFile = '../model/MANNE_GRAY_SC0_2/pose_deploy.prototxt';
param.model(6).description = 'MANNE_GRAY all S S2C3 Fine tuned it2000';
param.model(6).description_short = 'MANNE_AS_S2C3_2000';
param.model(6).boxsize = 368;
param.model(6).np = 14;
param.model(6).sigma = 21;
param.model(6).stage = 6;
param.model(6).padValue = 128;
param.model(6).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(6).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(7).caffemodel = '../model/MANNE_GRAY_SC0_2/caffemodel/pose_iter_200S1_6.caffemodel';
param.model(7).deployFile = '../model/MANNE_GRAY_SC0_2/pose_deploy.prototxt';
param.model(7).description = 'MANNE_GRAY S1 S6 fine tuned';
param.model(7).description_short = 'MANNE_S1S6';
param.model(7).boxsize = 368;
param.model(7).np = 14;
param.model(7).sigma = 21;
param.model(7).stage = 6;
param.model(7).padValue = 128;
param.model(7).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(7).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};
                     
param.model(8).caffemodel = '../model/MANNE_GRAY_SC0_2/caffemodel/pose_iter_200S1T6.caffemodel';
param.model(8).deployFile = '../model/MANNE_GRAY_SC0_2/pose_deploy.prototxt';
param.model(8).description = 'MANNE_GRAY all stage fine tuned';
param.model(8).description_short = 'MANNE_AS';
param.model(8).boxsize = 368;
param.model(8).np = 14;
param.model(8).sigma = 21;
param.model(8).stage = 6;
param.model(8).padValue = 128;
param.model(8).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(8).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};         
                     
param.model(9).caffemodel = '../model/MANNE_GRAY_SC0_2/caffemodel/pose_iter_200S1T6_C3S2.caffemodel';
param.model(9).deployFile = '../model/MANNE_GRAY_SC0_2/pose_deploy.prototxt';
param.model(9).description = 'MANNE_GRAY all stage C3S2 fine tuned it200';
param.model(9).description_short = 'MANNE_AS_C3S2_200';
param.model(9).boxsize = 368;
param.model(9).np = 14;
param.model(9).sigma = 21;
param.model(9).stage = 6;
param.model(9).padValue = 128;
param.model(9).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(9).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};        

param.model(10).caffemodel = '../model/MANNE_GRAY_SC0_2/caffemodel/pose_iter_200S6.caffemodel';
param.model(10).deployFile = '../model/MANNE_GRAY_SC0_2/pose_deploy.prototxt';
param.model(10).description = 'MANNE_GRAY only stage 6 fine tuned it200';
param.model(10).description_short = 'MANNE_S6_200';
param.model(10).boxsize = 368;
param.model(10).np = 14;
param.model(10).sigma = 21;
param.model(10).stage = 6;
param.model(10).padValue = 128;
param.model(10).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(10).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};    
                                       
param.model(11).caffemodel = '../model/IRS_REAL_SC0_2/caffemodel/IRS_REAL_AS_S2C3_200.caffemodel';
param.model(11).deployFile = '../model/IRS_REAL_SC0_2/pose_deploy.prototxt';
param.model(11).description = 'IRS_REAL last layer of all stages and stage 2 mix convolutional layer 3 with iteration200';
param.model(11).description_short = 'IRS_REAL_AS_S2C3_200';
param.model(11).boxsize = 368;
param.model(11).np = 14;
param.model(11).sigma = 21;
param.model(11).stage = 6;
param.model(11).padValue = 128;
param.model(11).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(11).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'}; 
                     
param.model(12).caffemodel = '../model/IRS_REAL_SC0_2/caffemodel/IRS_REAL_S6.caffemodel';
param.model(12).deployFile = '../model/IRS_REAL_SC0_2/pose_deploy.prototxt';
param.model(12).description = 'IRS_REAL with only last layer of stage 6 iter200';
param.model(12).description_short = 'IRS_REAL_S6';
param.model(12).boxsize = 368;
param.model(12).np = 14;
param.model(12).sigma = 21;
param.model(12).stage = 6;
param.model(12).padValue = 128;
param.model(12).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(12).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'}; 
                     
param.model(13).caffemodel = '../model/IRS_REAL_SC0_2/caffemodel/IRS_REAL_AS_S2C3_2000.caffemodel';
param.model(13).deployFile = '../model/IRS_REAL_SC0_2/pose_deploy.prototxt';
param.model(13).description = 'IRS_REAL with only last layer of stage 6 iter2000';
param.model(13).description_short = 'IRS_REAL_AS_S2C3_2000';
param.model(13).boxsize = 368;
param.model(13).np = 14;
param.model(13).sigma = 21;
param.model(13).stage = 6;
param.model(13).padValue = 128;
param.model(13).limbs = [1 2; 3 4; 4 5; 6 7; 7 8; 9 10; 10 11; 12 13; 13 14];
param.model(13).part_str = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                         'Lsho', 'Lelb', 'Lwri', ...
                         'Rhip', 'Rkne', 'Rank', ...
                         'Lhip', 'Lkne', 'Lank', 'bkg'};