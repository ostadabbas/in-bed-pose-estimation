function fts_hog = getNendHog(I)
% get n end hog features from the image 
I_bw = im2bw(I, 0.4);

[validR,validC]  = find (I_bw);
startX = min(validC);
startY = min(validR);
width = max(validC)- startX+1;
height =max(validR)-startY+1;

imgBB = [startX,startY,width,height];
 keyCentTemp = GetVnCenters(imgBB,width,2);
[hog1,valid, ptVis] = extractHOGFeatures(I,keyCentTemp,'CellSize',round([width/2 width/2]));
hogT = hog1';
fts_hog = hogT(:);


            
           