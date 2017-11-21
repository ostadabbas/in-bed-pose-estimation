function bb = GetBB(Ibw)
% get a bounding box from bw images. 
   [validR,validC]  = find (Ibw);
        startX = min(validC);
        startY = min(validR);
        width = max(validC)- startX+1;
        height =max(validR)-startY+1;
        
        bb= [startX,startY,width,height];
        