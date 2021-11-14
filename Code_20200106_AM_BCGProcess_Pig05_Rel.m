for eveNo = [1:5]
close all
clearvars -except eveNo
% Pig 5
% Note 1: P1is greater than P2, so I flipped the signals: P1 --> Femoral and P2 -->
% aortic arch
% Note 2: DP2 is much lower than DP1! --> makes Kwave amplitude much larger
% than Jwave amplitude in the constructed BCG!
%% Load 
% Intervention timing
addpath('C:\Users\sumbu\Documents\Research\GUI files')
load eventbeats
eventbeats(:,{'events'})    % Name of the interventions
eventbeats(:,{'pig5'})  % beat numbers
pigNo = 5;
if eveNo ~= 19
baseBt = [eventbeats{eveNo,6}.(['pig',num2str(pigNo)]),eventbeats{eveNo+1,6}.(['pig',num2str(pigNo)])];   % start and end beats in baseline 1 - relative or Absolute
end
% Load data in Pigs
addpath('C:\Users\sumbu\Documents\Research\data')
load(['pig',num2str(pigNo),'_relative'])
% load(['pig',num2str(pigNo),'_absolute'])
%% make ECG and BPs the same length as biopac ECG
% Filter BP
if eveNo == 5
baseBt = [eventbeats{eveNo,6}.(['pig',num2str(pigNo)]),eventbeats{eveNo+1,6}.(['pig',num2str(pigNo)])];   % start and end beats in baseline 1 - relative
baseBt(2) = length(new_beats_b);
end
% For Absolute
if eveNo == 10
    baseBt = [eventbeats{eveNo,6}.(['pig',num2str(pigNo)]),eventbeats{eveNo+7,6}.(['pig',num2str(pigNo)])];   % start and end beats in baseline 1 - relative or Absolute
end
if eveNo == 19
    baseBt = eventbeats{eveNo,6}.(['pig',num2str(pigNo)]);   % start and end beats in baseline 1 - relative
    baseBt(2) = length(new_beats_b);
end
fs = 2000;  %[Hz]
[u,v]=butter(1,20/(fs/2));
P2(isnan(P2)==1)=0;              % substitute NAN values with Zero
P1Filtered=filtfilt(u,v,P2);        % Aortic Arch (labels are flipped)
P2Filtered=filtfilt(u,v,P1);        % Femoral Artery
stBt = baseBt(1);
ndBt = baseBt(2);
if eveNo == 19
    ndBt = baseBt(2)-1;
end
figure
axt(1) = subplot(2,1,1);
plot(100*ECG_b(new_beats_b(stBt):new_beats_b(ndBt)))
hold on
grid on
plot(ECG1(new_beats_t(stBt):new_beats_t(ndBt)))
b_interval = new_beats_b(stBt):new_beats_b(ndBt);
t_interval = new_beats_t(stBt):new_beats_t(ndBt);

t_diff = diff(floor(new_beats_t));
b_diff = diff(floor(new_beats_b));
sbt = b_diff-t_diff;
ECG_t_copy = [];    % Adjusted length ECG
for ii = stBt:ndBt-1
    if sbt(ii) == 0
        ECG_t_copy = [ECG_t_copy;ECG1(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1)];
    elseif sbt(ii)>0
        ECG_t_copy = [ECG_t_copy;ECG1(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1);ECG1(floor(new_beats_t(ii+1)))*ones(floor(sbt(ii)),1)];
    else
        ECG_t_copy = [ECG_t_copy;ECG1(floor(new_beats_t(ii)):floor(new_beats_t(ii+1)-1+sbt(ii)))];
    end
end
ECG_t_copy = [ECG_t_copy;ECG_t_copy(end)];
plot(ECG_t_copy)

% Adjust P1
P1Filtered_copy = [];    % Adjusted length P1
for ii = stBt:ndBt-1
    if sbt(ii) == 0
        P1Filtered_copy = [P1Filtered_copy;P1Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1)];
    elseif sbt(ii)>0
        P1Filtered_copy = [P1Filtered_copy;P1Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1);P1Filtered(floor(new_beats_t(ii+1)))*ones(floor(sbt(ii)),1)];
    else
        P1Filtered_copy = [P1Filtered_copy;P1Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1)-1+sbt(ii)))];
    end
end
P1Filtered_copy = [P1Filtered_copy;P1Filtered_copy(end)];
axt(2) = subplot(2,1,2);
plot(P1Filtered(floor(new_beats_t(stBt)):floor(new_beats_t(ndBt))))
hold on
plot(P1Filtered_copy)
% Adjust P2
P2Filtered_copy = [];    % Adjusted length P2
for ii = stBt:ndBt-1
    if sbt(ii) == 0
        P2Filtered_copy = [P2Filtered_copy;P2Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1)];
    elseif sbt(ii)>0
        P2Filtered_copy = [P2Filtered_copy;P2Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1))-1);P2Filtered(floor(new_beats_t(ii+1)))*ones(floor(sbt(ii)),1)];
    else
        P2Filtered_copy = [P2Filtered_copy;P2Filtered(floor(new_beats_t(ii)):floor(new_beats_t(ii+1)-1+sbt(ii)))];
    end
end
P2Filtered_copy = [P2Filtered_copy;P2Filtered_copy(end)];
plot(P2Filtered(floor(new_beats_t(stBt)):floor(new_beats_t(ndBt))))
plot(P2Filtered_copy)
grid on
legend('P1','P1_a','P2','P2_a')
linkaxes(axt,'x')

%% Estimate BCG
global N
% Time
btStart = baseBt(1);
btEnd = baseBt(2);
b_interval = [new_beats_b(btStart):new_beats_b(btEnd)];
t_interval = [new_beats_t(btStart):new_beats_t(btEnd)];
timesig = [b_interval(1):b_interval(end)]./fs;
% Filter 
[u,v]=butter(1,[10 30]/(fs/2));
ECGFiltered_b = filtfilt(u,v,ECG_b);
ECG1(isnan(ECG1)) = 0; % replace Nan values
ECGFiltered_t = filtfilt(u,v,ECG1);
subplot(2,1,1)
plot(ECGFiltered_t(floor(new_beats_t(stBt)):floor(new_beats_t(ndBt))))
plot(100*ECGFiltered_b(floor(new_beats_b(stBt)):floor(new_beats_b(ndBt))))
plot(new_beats_b(stBt:ndBt)-new_beats_b(stBt)+1,100*ECG_b(floor(new_beats_b(stBt:ndBt))),'*')
plot(floor(new_beats_b(stBt:ndBt-1)-new_beats_b(stBt))+1,ECG_t_copy(floor(new_beats_b(stBt:ndBt-1)-new_beats_b(stBt))+1),'ok')

legend('ECG_b','ECG_t','ECG_a_d_j','ECG_t_F_i_l','ECG_b_F_i_l')
N = 4;
[u,v]=butter(1,[N,10]/(fs/2));   % Azin
sigDis = AX_1;
BCGFiltered=filtfilt(u,v,sigDis);
sigMat = [AX_1,AX_2,AY_1,AY_2,AZ_1,AZ_2];
BCGMatFilt = filtfilt(u,v,sigMat);
BCGMat = BCGMatFilt(b_interval,:);
BCGsig = BCGFiltered(b_interval);
BCGCopysig = BCGFiltered(b_interval);
ECGMaxLoc = floor(new_beats_b(stBt:ndBt-1)-new_beats_b(stBt))+1;    % This term changed
startIdx = 1;
BCGMA = expMA(BCGsig,BCGCopysig,ECGMaxLoc,startIdx,timesig,fs);
BCGInt = expMAInt(BCGsig,BCGCopysig,ECGMaxLoc,startIdx,timesig,fs);     % Double integrated BCG + Exp MA
BCGMatMA = expMA(BCGMat,BCGMat,ECGMaxLoc,startIdx,timesig,fs);
BCGMatInt = expMAInt(BCGMat,BCGMat,ECGMaxLoc,startIdx,timesig,fs);
%% wavelet analysis
[psig,f] = pwelch(sigDis,10000,1000,10000,fs);
figure
plot(f,psig*1,'linewidth',2)
[pwVal,pwInd] = findpeaks(psig,'minpeakdistance',5);
hold on
plot(f(pwInd),pwVal,'*')
axis([0 10 0 5e-7])
xlabel('Hz')
%% BCG Model
% Calculate P0 - with Time Delay
tauA = 0.015;        %Time delay b/w P0 and P1[s]
P0 = P1Filtered_copy(floor(tauA*fs):end);
% Calculate P2 - with Time Delay
tauD = 0.03;        % Time Delay b/w P2 and femoral BP measurement location
P2 = P2Filtered_copy(floor(tauD*fs):end);
L = min(length(P0),length(P2));
% Calculate BCG from BP
DA = 0.025;
AA = pi/4*DA^2;
DD = 0.01;
AD = pi/4*DD^2;
BCG_Des = AD.*(P1Filtered_copy(1:L)-P2(1:L));
BCG_Asc = AA*(P1Filtered_copy(1:L)-P0(1:L));
BCG_est = BCG_Des+BCG_Asc;
% BCG with P2 at measurement site
BCG_Des_site = AD.*(P1Filtered_copy(1:L)-P2Filtered_copy(1:L));
BCG_est_site = BCG_Des_site+BCG_Asc;
% Plot
figure('units','normalized','position',[0 0 1 1])
ax(1) = subplot(7,1,1);
plot(t_interval,ECG1(t_interval),'b')
hold on
plot(new_beats_t(btStart:btEnd),ECG1(floor(new_beats_t(btStart:btEnd))),'o')
s = plot(t_interval,ECGFiltered_t(t_interval),'g');
ylabel ECG_t
grid on
plot(b_interval(1:end-1),ECG_t_copy(1:length(b_interval(1:end-1))),'k')
plot(floor(new_beats_b(stBt:ndBt-1)),ECG_t_copy(floor(new_beats_b(stBt:ndBt-1)-new_beats_b(stBt))+1),'*')

ax(2) = subplot(7,1,2);
plot(b_interval,ECG_b(b_interval),'k')
hold on
plot(new_beats_b(btStart:btEnd),ECG_b(floor(new_beats_b(btStart:btEnd))),'o')
ylabel ECG_b
plot(new_beats_b(btStart:btEnd),ECG_b(floor(new_beats_b(btStart:btEnd))),'*')
s = plot(b_interval,5*ECGFiltered_b(b_interval),'g');
grid on
ax(3) = subplot(7,1,3);
plot(b_interval,BCGMA)
hold on
plot(b_interval,BCGInt)
ylabel BCG
legend('BCG Acc','BCG Disp')
grid on
ax(4) = subplot(7,1,4);
plot(b_interval(1:L),BCG_est(1:L))
hold on
plot(b_interval(1:L),BCG_est_site(1:L))
legend('w/ P2 Dly','w/ P2 Mes Site')
ylabel BCG_E_s_t
grid on
ax(5) = subplot(7,1,5);
plot(b_interval(1:L),P1Filtered_copy(1:L))
hold on
plot(b_interval(1:L),P2Filtered_copy(1:L))
plot(b_interval(1:L),P0(1:L))
plot(b_interval(1:L),P2(1:L))
legend('P1','P2 Mes','P0','P2 Dly')
ylabel BP
grid on
ax(6) = subplot(7,1,6);
plot(b_interval(1:L),BCG_Des(1:L))
hold on
plot(b_interval(1:L),BCG_Des_site(1:L))
legend('w/ P2 Dly','w/ P2 Mes Site')
ylabel BCG_D
grid on
ax(7) = subplot(7,1,7);
plot(b_interval(1:L),BCG_Asc(1:L))
ylabel BCG_A
linkaxes(ax,'x')
grid on
% axis([2.84e6 2.8535e6 -inf inf])
% axis auto
for hh = 1:7
    h{hh} = get(ax(hh),'position');
    h{hh} = h{hh}+[0.05 -0.02*floor((hh-1)/2) -0.05 0.05];
    set(ax(hh),'position',h{hh})
end
figure
ax2(1) = subplot(2,1,1);
plot(P1)
hold on
plot(P2)
legend('P1','P2')
ax2(2) = subplot(2,1,2);
plot(P1Filtered)
hold on
plot(P2Filtered)
linkaxes(ax2,'x')

%% Representative beat
beatNo = btStart:btEnd-1;
diffBt = diff(new_beats_b);
maxBtLg = max(diffBt);
saveBCGMAbt = nan(maxBtLg,length(beatNo));
saveBCGIntbt = nan(maxBtLg,length(beatNo));
figure
subplot(2,1,1)
for ii = beatNo
    hold on
    btRange = floor(new_beats_b(ii):new_beats_b(ii+1));
    btRange = btRange- floor(new_beats_b(btStart))+1;
    plot(BCGMA(btRange),'color',[0.5 0.5 0.5])
    saveBCGMAbt(1:length(btRange),ii) = BCGMA(btRange);
end
title(['pig: ',num2str(pigNo),'  Rel: ',num2str(eveNo)])
ylabel('BCG Acc')

subplot(2,1,2)
for ii = beatNo
    hold on
    btRange = floor(new_beats_b(ii):new_beats_b(ii+1));
    btRange = btRange- floor(new_beats_b(btStart))+1;
    plot(BCGInt(btRange),'color',[0.5 0.5 0.5])
    saveBCGIntbt(1:length(btRange),ii) = BCGInt(btRange);
end
ylabel('BCG Disp')
saveas(gcf,['C:\Users\sumbu\Documents\Research\GUI files\Fig_Pig_',num2str(pigNo),'_Rel_',num2str(eveNo),'.png'])
%% Save signals in structure
data.FreqPW = [pwVal; pwInd];
data.accel = 1;
data.fs = fs;
data.time = transpose(0:1/fs:length(b_interval)/fs-1/fs);
data.Peakt = new_beats_t;
data.BeatNo = [btStart:btEnd];
data.PeakSam = floor(new_beats_b(btStart:btEnd)); data.Peakb = data.PeakSam-data.PeakSam(1)+1;
data.ECGt = ECG1(t_interval);
data.ECGt_Adj = ECG_t_copy(1:length(b_interval)); 
data.ECGb = ECG_b(b_interval);
data.AX1 = AX_1(b_interval);
data.AXFiltered1 = BCGsig;
data.BCGInt1 = BCGInt;
data.BCG1 = BCGMA;
data.MatAcc = sigMat(b_interval,:);
data.MatFilt = BCGMat;
data.MatBCG = BCGMatMA;
data.MatBCGInt = BCGMatInt;
data.BP1 = P1Filtered_copy;
data.BP2 = P2Filtered_copy;
data.PPG1 = PPG_1(b_interval);
data.PPG2 = PPG_2(b_interval);
data.FreqContent = [f, psig];
data.Event = ['Pig_',num2str(pigNo),'Rel_',num2str(eveNo)];
% data.Event = ['Pig_',num2str(pigNo),'Abs_',num2str(eveNo)];
save(['C:\Users\sumbu\Documents\Research\data\pig05Data_Rel',num2str(eveNo)],'data')
% save(['D:\Project-AHA\Data For GUI\pig05Data_Abs',num2str(eveNo)],'data')
end
%%
function BCGsig=expMA(BCGsig,BCGCopysig,ECGMaxLoc,startIdx,time,fs)

% 10-beat exponential moving average filter for BCG signal

M=10;alpha=2/(M+1);
% startIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'startIdx');
% currentWindowStartIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'currentWindowStartIdx');
% currentWindowEndIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'currentWindowEndIdx');
% ECGMaxLoc = getappdata(mPlots.Analysis.(eventTag).Parent,'ECGMaxLoc');
% BCGsig = getappdata(mPlots.Analysis.(eventTag).Parent,'BCGsig');
% BCGCopysig = getappdata(mPlots.Analysis.(eventTag).Parent,'BCGCopysig');
ECGMaxDiff=diff(ECGMaxLoc);
nInt=round(nanmedian(ECGMaxDiff)*1.2);
maxPeakIdx=length(ECGMaxLoc);
while (ECGMaxLoc(maxPeakIdx)-(startIdx-1)+nInt-1>length(BCGsig))
    maxPeakIdx = maxPeakIdx-1;
end
% disp(maxPeakIdx)
if maxPeakIdx<M
    fprintf('Error: Not enough beats for BCG exponential moving averaging!\n')
else
    for peakIdx = M:maxPeakIdx
        tempNum=0;tempDen=0;
        for beatIdx = 1:M
            start = ECGMaxLoc(peakIdx-beatIdx+1)-(startIdx-1);
            if ~isnan(start)
%                 [nInt,start,start+nInt,length(BCGsig)]
                tempBCGsig = BCGCopysig(start:start+nInt-1);
                tempAmp(beatIdx,1) = max(tempBCGsig)-min(tempBCGsig);
                isValid(beatIdx,1) = true;
            else
                tempAmp(beatIdx,1) = NaN;
                isValid(beatIdx,1) = false;
            end
        end
%         tempAmp
%         isValid
        isIncluded = double((tempAmp<=2*nanmedian(tempAmp) & isValid));
        for beatIdx = 1:M
            start = ECGMaxLoc(peakIdx-beatIdx+1)-(startIdx-1);
            if isIncluded(beatIdx,1) == 1
                tempNum=tempNum+(1-alpha)^(beatIdx-1)*BCGCopysig(start:start+nInt-1);
                tempDen=tempDen+(1-alpha)^(beatIdx-1);
            end
        end
        newstart = ECGMaxLoc(peakIdx)-(startIdx-1);
        if ~isnan(newstart)
            BCGsig(newstart:newstart+nInt-1)=tempNum/tempDen;
        end
    end
end
% % Integrate- Azin
% % time = getappdata(mPlots.Analysis.EventFiltered.Parent,'timesig');
% % load mparams
% % fs=mParams.Constant.fs; % sampling frequency
% % ts = 1/fs;
% % t1=0:ts:ts*(length(ECG_T3)-1);
% int1 = cumtrapz(time(startIdx:startIdx+length(BCGsig)-1)',BCGsig);
% int2 = cumtrapz(time(startIdx:startIdx+length(BCGsig)-1)',int1);
% global N
% [u,v]=butter(1,[N,50]/(fs/2));
% % [u,v]=butter(1,[0.5,50]/(fs/2));   % Azin
% BCGsig=filtfilt(u,v,int2)*1e3;
% 
% % setappdata(mPlots.Analysis.(eventTag).Parent,'BCGsig',BCGsig)
% % set(mPlots.Analysis.(eventTag).BCG,'YData',BCGsig(currentWindowStartIdx:currentWindowEndIdx));
fprintf(['BCG has been updated!\n'])

end

function BCGsig=expMAInt(BCGsig,BCGCopysig,ECGMaxLoc,startIdx,time,fs)

% 10-beat exponential moving average filter for BCG signal

M=10;alpha=2/(M+1);
% startIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'startIdx');
% currentWindowStartIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'currentWindowStartIdx');
% currentWindowEndIdx = getappdata(mPlots.Analysis.(eventTag).Parent,'currentWindowEndIdx');
% ECGMaxLoc = getappdata(mPlots.Analysis.(eventTag).Parent,'ECGMaxLoc');
% BCGsig = getappdata(mPlots.Analysis.(eventTag).Parent,'BCGsig');
% BCGCopysig = getappdata(mPlots.Analysis.(eventTag).Parent,'BCGCopysig');
ECGMaxDiff=diff(ECGMaxLoc);
nInt=round(nanmedian(ECGMaxDiff)*1.2);
maxPeakIdx=length(ECGMaxLoc);
while (ECGMaxLoc(maxPeakIdx)-(startIdx-1)+nInt-1>length(BCGsig))
    maxPeakIdx = maxPeakIdx-1;
end
% disp(maxPeakIdx)
if maxPeakIdx<M
    fprintf('Error: Not enough beats for BCG exponential moving averaging!\n')
else
    for peakIdx = M:maxPeakIdx
        tempNum=0;tempDen=0;
        for beatIdx = 1:M
            start = ECGMaxLoc(peakIdx-beatIdx+1)-(startIdx-1);
            if ~isnan(start)
%                 [nInt,start,start+nInt,length(BCGsig)]
                tempBCGsig = BCGCopysig(start:start+nInt-1);
                tempAmp(beatIdx,1) = max(tempBCGsig)-min(tempBCGsig);
                isValid(beatIdx,1) = true;
            else
                tempAmp(beatIdx,1) = NaN;
                isValid(beatIdx,1) = false;
            end
        end
%         tempAmp
%         isValid
        isIncluded = double((tempAmp<=2*nanmedian(tempAmp) & isValid));
        for beatIdx = 1:M
            start = ECGMaxLoc(peakIdx-beatIdx+1)-(startIdx-1);
            if isIncluded(beatIdx,1) == 1
                tempNum=tempNum+(1-alpha)^(beatIdx-1)*BCGCopysig(start:start+nInt-1);
                tempDen=tempDen+(1-alpha)^(beatIdx-1);
            end
        end
        newstart = ECGMaxLoc(peakIdx)-(startIdx-1);
        if ~isnan(newstart)
            BCGsig(newstart:newstart+nInt-1)=tempNum/tempDen;
        end
    end
end
% Integrate- Azin
% time = getappdata(mPlots.Analysis.EventFiltered.Parent,'timesig');
% load mparams
% fs=mParams.Constant.fs; % sampling frequency
% ts = 1/fs;
% t1=0:ts:ts*(length(ECG_T3)-1);
int1 = cumtrapz(time(startIdx:startIdx+length(BCGsig)-1)',BCGsig);
int2 = cumtrapz(time(startIdx:startIdx+length(BCGsig)-1)',int1);
global N
[u,v]=butter(1,[N,50]/(fs/2));
% [u,v]=butter(1,[0.5,50]/(fs/2));   % Azin
BCGsig=filtfilt(u,v,int2)*1e3;

% setappdata(mPlots.Analysis.(eventTag).Parent,'BCGsig',BCGsig)
% set(mPlots.Analysis.(eventTag).BCG,'YData',BCGsig(currentWindowStartIdx:currentWindowEndIdx));
fprintf(['BCG Integral has been updated!\n'])

end