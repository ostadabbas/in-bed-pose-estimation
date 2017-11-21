function genJSON(dataset)
% use genJSON('MANNE_GRAY');
% only saved to joint_all to a JASON 
% take indexed pictures as training. 
% save joint position to the json file 

    addpath('util/jsonlab/');

    
        if(strcmp(dataset, 'MANNE_GRAY')) % generate the mannequin JASON files
        % in cpp: real scale = param_.target_dist()/meta.scale_self = (41/35)/scale_input
        targetDist = 41/35; % in caffe cpp file 41/35
        % oriTrTe = load('../dataset/LEEDS/lsp_dataset/joints.mat');
        % extTrain = load('../dataset/LEEDS/lspet_dataset/joints.mat');
        
        % MANNE data
        % oriTrTe = load('../dataset/MANNE_GRAY/Data_labeled(419_3d).mat')
         oriTrTe = load('../dataset/MANNE_GRAY_SC0_2/label419sc0_2.mat')
        

        ordering = [1 2 3, 4 5 6, 15 16, 13 14, 7 8 9, 10 11 12]; % should follow MPI 16 parts..?
        oriTrTe.Label_data_3d(:,[15 16],:) = 0;    % 3 coord x 16 joint x N subjects
        oriTrTe.Label_data_3d = oriTrTe.Label_data_3d(:,ordering,:);    % change to MPII format
        oriTrTe.Label_data_3d(3,:,:) = 1 - oriTrTe.Label_data_3d(3,:,:);
        oriTrTe.Label_data_3d = permute(oriTrTe.Label_data_3d, [2 1 3]);  % joints a structure joints inside 14x3xN
        
        % extTrain.joints([15 16],:,:) = 0;
        % extTrain.joints = extTrain.joints(ordering,:,:);

        count = 1;
       
       % original LEEDS settings 
        % path = {'lspet_dataset/images/im%05d.jpg', 'lsp_dataset/images/im%04d.jpg'};
        % local_path = {'../dataset/LEEDS/lspet_dataset/images/im%05d.jpg', '../dataset/LEEDS/lsp_dataset/images/im%04d.jpg'};
        % num_image = [10000, 1000]; %[10000, 2000];

        path = 'images/%06d.jpg';
%         local_path = '../dataset/MANNE_GRAY/images/%06d.jpg';
        local_path = '../dataset/MANNE_GRAY_SC0_2/images/%06d.jpg';
%         imgIndices = [1:100,201:300];

        % SC0.2 edition 
        imgIndices = [1:270,301:400];   % 10:1 ratio training /testing  
        num_image = length(imgIndices);
        %num_image = 419; %[10000, 2000];        
       
            for i = 1:num_image
                im = imgIndices(i);
                % trivial stuff for LEEDS
                joint_all(count).dataset = 'MANNE_GRAY';
                joint_all(count).isValidation = 0;
                joint_all(count).img_paths = sprintf(path, im);
                joint_all(count).numOtherPeople = 0;
                joint_all(count).annolist_index = count;
                joint_all(count).people_index = 1;
                % joints and w, h
           
                    joint_this = oriTrTe.Label_data_3d(:,:,im);
           
                    % joint_this = oriTrTe.joints(:,:,im);
            
                path_this = sprintf(local_path, im);
                [h,w,~] = size(imread(path_this));

                joint_all(count).img_width = w;
                joint_all(count).img_height = h;
                joint_all(count).joint_self = joint_this;   % x,y ,visibility 1
                % infer objpos
                invisible = (joint_all(count).joint_self(:,3) == 0);
                % if(dataset == 1) %lspet is not tightly cropped
                    joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
                    joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;

                count = count + 1;
                fprintf('processing %s\n', path_this);
            end
        

        joint_all = insertMPILikeScale(joint_all, targetDist);
        
   
%         opt.FileName = 'json/MANNE_GRAY_annotations.json';
        opt.FileName = 'json/MANNE_GRAYSC0_2_annotations.json';
        opt.FloatFormat = '%.3f';
        savejson('root', joint_all, opt);

        % ************************
         elseif(strcmp(dataset, 'IRS_REAL')) % generate the mannequin JASON files
       
         oriTrTe = load('../dataset/IRS_REAL_MltSC0_2_170518N/joints_gt.mat')
        

        ordering = [1 2 3, 4 5 6, 15 16, 13 14, 7 8 9, 10 11 12]; % should follow MPI 16 parts..?
        oriTrTe.joints_gt(:,[15 16],:) = 0;    % 3 coord x 16 joint x N subjects
        oriTrTe.joints_gt = oriTrTe.joints_gt(:,ordering,:);    % change to MPII format
        oriTrTe.joints_gt(3,:,:) = 1 - oriTrTe.joints_gt(3,:,:);
        oriTrTe.joints_gt = permute(oriTrTe.joints_gt, [2 1 3]);  % joints a structure joints inside 14x3xN
        
        % extTrain.joints([15 16],:,:) = 0;
        % extTrain.joints = extTrain.joints(ordering,:,:);

        count = 1;       


        path = 'images/%06d.jpg';
%         local_path = '../dataset/MANNE_GRAY/images/%06d.jpg';
        local_path = '../dataset/IRS_REAL_MltSC0_2_170518N/images/%06d.jpg';
%         imgIndices = [1:100,201:300];

        % SC0.2 edition 
        % imgIndices = [1:270,301:400];   % 10:1 ratio training /testing  
        % choose Fenghui for test,  87 to  113  from 223 
        imgIndices = [1:86,114:223]
        num_image = length(imgIndices);
        %num_image = 419; %[10000, 2000];        
       
            for i = 1:num_image
                im = imgIndices(i);
                % trivial stuff for LEEDS
                joint_all(count).dataset = 'IRS_REAL';
                joint_all(count).isValidation = 0;
                joint_all(count).img_paths = sprintf(path, im);
                joint_all(count).numOtherPeople = 0;
                joint_all(count).annolist_index = count;
                joint_all(count).people_index = 1;
                % joints and w, h
           
                    joint_this = oriTrTe.joints_gt(:,:,im);
           
                    % joint_this = oriTrTe.joints(:,:,im);
            
                path_this = sprintf(local_path, im);
                [h,w,~] = size(imread(path_this));

                joint_all(count).img_width = w;
                joint_all(count).img_height = h;
                joint_all(count).joint_self = joint_this;   % x,y ,visibility 1
                % infer objpos
                invisible = (joint_all(count).joint_self(:,3) == 0);
                % if(dataset == 1) %lspet is not tightly cropped
                    joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
                    joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;
         
                count = count + 1;
                fprintf('processing %s\n', path_this);
            end
        

        joint_all = insertMPILikeScale(joint_all, targetDist);
%         opt.FileName = 'json/MANNE_GRAY_annotations.json';
         opt.FileName = 'json/IRS_REAL_SC0_2_annotations.json';
        opt.FloatFormat = '%.3f';
        savejson('root', joint_all, opt);
    
    end
    
function out = parseFile(in)
    % out is like /media/posenas1//141215/141215_Pose1/Kinect//KINECTNODE1-December-15-2014-13-22-33/color/color-2561794.394.png
    % in should be /media/posenas1/Captures/poseMachineKinect/141215/141215_Pose1/Kinect//KINECTNODE1-December-15-2014-13-22-33/color/color-2561794.394.png
    out = strrep(in, '/media/posenas1/', '/media/posenas1/Captures/poseMachineKinect');


function joint_all = insertMPILikeScale(joint_all, targetDist)
    % calculate scales for each image first
    joints = cat(3, joint_all.joint_self);  % jointself is  14x3 to 14x3xN
    joints([7 8],:,:) = []; % delete 7 and 8?  left 14 columns 
    pa = [2 3 7, 5 4 7, 8 0, 10 11 7, 13 12 7];
    %cur=[1 2 3, 4 5 6, 7 8, 9  10 11,12 13 14]; 
    x = permute(joints(:,1,:), [3 1 2]);    % original 14x3xN to N x14 x1 as x
    y = permute(joints(:,2,:), [3 1 2]);
    vis = permute(joints(:,3,:), [3 1 2]);
    validLimb = 1:14-1;

    x_diff = x(:, [1:7,9:14]) - x(:, pa([1:7,9:14])); % N x 14
    y_diff = y(:, [1:7,9:14]) - y(:, pa([1:7,9:14]));
    limb_vis = vis(:, [1:7,9:14]) .* vis(:, pa([1:7,9:14]));    % 2 jonits visible
    l = sqrt(x_diff.^2 + y_diff.^2);    % all limb NX14 length 

    for p = 1:14-1 % for each limb. reference: 7th limb, which is 7 to pa(7) (neck to head)
        valid_compare = limb_vis(:,7) .* limb_vis(:,p); % head body visible
        ratio = l(valid_compare==1, p) ./ l(valid_compare==1, 7); % 14 part to head ratio
        r(p) = median(ratio(~isnan(ratio), 1)); % median value of 
    end

    numFiles = size(x_diff, 1); %  Nimgs
    all_scales = zeros(numFiles, 1);

    boxSize = 368;
    psize = 64;
    nSqueezed = 0;
    
    for file = 1:numFiles %numFiles
        l_update = l(file, validLimb) ./ r(validLimb);
        l_update = l_update(limb_vis(file,:)==1); % 1x N_visi
        distToObserve = quantile(l_update, 0.75); 
        scale_in_lmdb = distToObserve/35; % can't get too small. 35 is a magic number to balance to MPI
        scale_in_cpp = targetDist/scale_in_lmdb; % can't get too large to be cropped

        visibleParts = joints(:, 3, file);
        visibleParts = joints(visibleParts==1, 1:2, file);
        x_range = max(visibleParts(:,1)) - min(visibleParts(:,1));
        y_range = max(visibleParts(:,2)) - min(visibleParts(:,2));
        scale_x_ub = (boxSize - psize)/x_range;
        scale_y_ub = (boxSize - psize)/y_range;

        scale_shrink = min(min(scale_x_ub, scale_y_ub), scale_in_cpp);
        
        if scale_shrink ~= scale_in_cpp
            nSqueezed = nSqueezed + 1;
            fprintf('img %d: scale = %f %f %f shrink %d\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub), nSqueezed);
        else
            fprintf('img %d: scale = %f %f %f\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub));
        end
        
        joint_all(file).scale_provided = targetDist/scale_shrink; % back to lmdb unit prop to x,y range
    end
    
    fprintf('total %d squeezed!\n', nSqueezed);
