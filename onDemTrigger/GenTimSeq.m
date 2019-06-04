function timSeq = GenTimSeq(vToggle, len, initialSt)
% generate a time sequence with len length, with the toggle point indicated
% by the vToggle. Result will toggle between 0 and 1. default intial value
% is intialSt default 0 
% this version only handle even toggle point.
if nargin<3
    initialSt =0;
end

stPt = vToggle(1:2:length(vToggle));
endPt = vToggle(2:2:length(vToggle));
if length(stPt)> length(endPt)
    stPt(end)=[];
end

if initialSt
    timSeq = ones(1,len);
    toggledVal = 0;
else
    timSeq = zeros(1,len);
    toggledVal = 1;
end

for i=1:length(stPt)
    timSeq(stPt(i):endPt(i))=toggledVal;
end
