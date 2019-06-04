function deg_rt = getBBrtFromV(bb)
% get bounding box rotation fom vertical position 
% bb 2x4 matrix contains the bb corners 
% deg_rt how many degs rotated from the vertical bb  
% note that the y is pointing down and the clockwise is positive direction 

diffs_bb  = diff(bb,1, 2);    % 1rd diff along row  
for i = 1:size(diffs_bb,2)
    lens_edge(i) = norm(diffs_bb(:,i));
end
% lens_edge = vecnorm(diffs_bb);  % only compare first 2 edge length  
% lens_edge   % show the edge lengths 
if lens_edge(1) <= lens_edge(2) % take the shorter edge 
    diff_edge = diffs_bb(:,1);  
else
    diff_edge = diffs_bb(:,2);  
end

if diff_edge(1) == 0 
    deg_rt = 90;
else
    deg_rt =-rad2deg(atan(diff_edge(2)/diff_edge(1)));   % atan(Y/X) 
end



    
