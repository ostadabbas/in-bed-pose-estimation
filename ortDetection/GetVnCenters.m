function centers = GetVnCenters(bb,blkWidth,n)
% put in the bouding box, and blkWidth of the block, get the n centers
% along the vertical
% this is a simplified version, can be extended in future. 
% output: centers nx2  [x1,y1; x2,y2;....xn,yn];

x1 = bb(1)+bb(3)/2;     % hori center
y1 = bb(2)+ blkWidth/2;

yn = bb(2)+bb(4)-blkWidth/2;

yVec = linspace(y1,yn,n);
xVec = x1* ones(1,n);

centers =single( [xVec;yVec]'); 
