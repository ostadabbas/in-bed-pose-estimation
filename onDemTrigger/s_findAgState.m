% s_findAgState 
%% only record the necessary image, save the memory cost. Record reserved version is in ex1


% all to gray images 
clear;clc;
% control parameters 
flgSave = 1;
nmFigPdf = 'triggerEstimation';

% other parameters
set(0,'DefaultAxesXGrid','on','DefaultAxesYGrid','on');
set(0,'DefaultLineLineWidth',2); % plot properties
% vdRt = '../..';
vdRt = '.'
% frRate = 11.28; 
nmRslt = 'pose1Trigger.mat';
gtTrigger = [53,139,208,370,447,535,624,781]; 
% read frame from 
v = VideoReader(fullfile(vdRt,'pose1.mov'));
frRate = v.FrameRate;
% area = v.Width* v.Height; 
% nChannel  = v.BitsPerPixel/8;    % 3 channel
maxIntense = 255; 
% frame rate is  
Nbf = 30;   % about 1.77s back window 
len = v.NumberOfFrames;
if Nbf>v.NumberOfFrames
    error('too large backward window');
end
% Ibuf =uint8(zeros(v.Height,v.Width,v.NumberOfFrames));
Dbuf =zeros(1,Nbf);  % difference buff 
% initialize Ibuf 
% Ibuf(:,:,1) =rgb2gray(read(v,1));   % actually, we don't have to keep all the image information in the queue. 
% but just in case new difference method on the whole data is needed. 
% agPre = 0;
% agCur = 0; 
agStates = zeros(1,v.NumberOfFrames);   % keep agitation state 
triggerStates = zeros(1,v.NumberOfFrames); % keep the trigger infor 
threshold = 0.003;    % tentative trials 
diffs = zeros(1,v.NumberOfFrames-1);    % record whole state 

for i =2:Nbf    
    Ipre = read(v,i-1);
    Icur = read(v,i);
    Dtemp = abs(Icur - Ipre);   
    Dbuf(i-1)= mean(Dtemp(:))/255;  % normalize to 0 and 1
end
% find difference 
for i = Nbf:v.NumberOfFrames
    Ipre = read(v,i-1);    % get in last iteration I;
    Icur =read(v,i);       
    % queue in 
%     Ibuf(:,:,1:end-1)= Ibuf(:,:,2:end); 
%     Ibuf(:,:,end) = Icur;
    Dbuf(1:end-1) = Dbuf(2:end);
    Dtemp =abs(Icur - Ipre);
    Dbuf(end) = mean(Dtemp(:))/255;     
    diffs(i)=  mean(Dtemp(:))/255;
    if max(Dbuf)>threshold;
        agStates(i) = 1;
    else
        agStates(i) = 0; 
    end
    if agStates(i)-agStates(i-1)<0;    % drop back to stable then capture
        triggerStates(i) = 1;
    end
end
gtStates = GenTimSeq(gtTrigger,v.NumberOfFrames);
timeRange = (1:len)/frRate; % second indexed 
if flgSave
    save(nmRslt,'triggerStates','agStates','gtStates','frRate','len','timeRange');
end

figure(1);
plot(timeRange,gtStates,'g--'); % gt green dash line 
hold on;
plot(timeRange,agStates,'b-.'); % blue predictied line 
stem(timeRange(find(triggerStates)),1.5*ones(1,length(find(triggerStates))),'-.or');
axis([1,timeRange(end),0,2]);
xlabel('time(s)');
legend({'ground truth aggitation','detected aggitation','trigger signal'},'FontSize',12);

% save figure to pdf test
set(figure(1),'Units','Inches');
pos = get(figure(1),'Position');
set(figure(1), 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
print(figure(1),nmFigPdf,'-dpdf','-r0');

% test trigger signal minus default trigger

% plot(diffs);
% plot(agStates);

% video = read(v,20);
% imshow(video)

% test the implay to capture the transition point 
% implay(fullfile(vdRt,'pose1.mov'));
% labeled data 
