function x_new = GetCoordImRot(x_old, sizeImg,angRot)
% after imrotate, calculated the new coordinates of the original point. 
% only works for the crop method, 
% keep whole image will result failure of this method 
% input x_old : [x1,y1; x2,y2; x3,y3;];
% otuput x_new: [x1new, y1new; x2new, y2new ...];

assert(size(x_old,2) ==2);  % only 2 coordinates 
% orig_x = orig_x - sizeImg(2);
x_ori = [x_old(:,1) - sizeImg(2)/2, x_old(:,2)-sizeImg(1)/2];
% orig_y = orig_y - sizeImg(1);
rot_mat=[cosd(angRot), sind(angRot); -sind(angRot) ,cosd(angRot)];
 new_oriR =x_ori*rot_mat';
%  new_orig(1) = new_orig(1) + sz(2);
% new_orig(2) = new_orig(2) + sz(1);
x_new = [new_oriR(:,1)+sizeImg(2)/2, new_oriR(:,2)+sizeImg(1)/2];