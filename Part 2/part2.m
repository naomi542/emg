%Example: https://www.mathworks.com/help/signal/ug/classify-arm-motions-using-emg-signals-and-deep-learning.html
%Create Datastore to Read Signal and Label Data
fs = 3000;

%localfile = matlab.internal.examples.downloadSupportFile("SPT","MyoelectricData.zip");
%datasetFolder = fullfile(fileparts(localfile),"MyoelectricData");
%unzip(localfile,'MyoelectricData.zip')


sds1 = signalDatastore('C:\Users\nkothiya\Desktop\assignment4\part2\Data',IncludeSubFolders=true,SampleRate=fs);
p = endsWith(sds1.Files,"d.mat");
sdssig = subset(sds1,p);

sds2 = signalDatastore('C:\Users\nkothiya\Desktop\assignment4\part2\Data',SignalVariableNames=["motion";"data_indx"],IncludeSubfolders=true);
p = endsWith(sds2.Files,"i.mat");
sdslbl = subset(sds2,p);

signal = preview(sdssig);

for i = 1:8
    ax(i) = subplot(4,2,i);
    plot(signal(:,i))
    title("Channel"+i)
end

linkaxes(ax,"y")

%Create ROI Table
lbls = {};

i = 1;
while hasdata(sdslbl)

    label = read(sdslbl);
    
    idx_start = label{2}(2:end-1)';
    idx_end = [idx_start(2:end)-1;idx_start(end)+(3*fs)];
    
    val = categorical(label{1}(2:end-1)',[1 2 3 4 5 6 7], ...
          ["HandOpen" "HandClose" "WristFlexion" "WristExtension" "Supination" "Pronation" "Rest"]);
    ROI = [idx_start idx_end];

    % In some cases, the number of label values and ROIs are not equal.
    % To eliminate these inconsistencies, remove the extra label value or ROI limits. 
    if numel(val) < size(ROI,1)
        ROI(end,:) = [];
    elseif numel(val) > size(ROI,1)
        val(end) = [];
    end

    lbltable = table(ROI,val);
    lbls{i} = {lbltable};

    i = i+1;
end

%Prepare Datastore
lblDS = signalDatastore(lbls);
lblstable = preview(lblDS);
lblstable{1}

DS = combine(sdssig,lblDS);
combinedData = preview(DS)

figure
msk = signalMask(combinedData{2});
plotsigroi(msk,combinedData{1}(:,1))

%Preprocess Data
tDS = transform(DS,@preprocess);
transformedData = preview(tDS);

%Divide Data into Training and Testing Sets
rng default
[trainIdx,~,testIdx] = dividerand(30,0.8,0,0.2);

trainIdx_all = {};
m = 1;

for k = trainIdx
    
    if k == 1
       start = k;
    else
       start = ((k-1)*24)+1;
    end
    l = start:k*24;
    trainIdx_all{m} = l;
    m = m+1;
end

trainIdx_all = cell2mat(trainIdx_all)';
trainDS = subset(tDS,trainIdx_all);

testIdx_all = {};
m = 1;

for k = testIdx
    if k == 1
       start = k;
    else
       start = ((k-1)*24)+1;
    end
    l = start:k*24;
    testIdx_all{m} = l;
    m = m+1;
end

testIdx_all = cell2mat(testIdx_all)';
testDS = subset(tDS,testIdx_all);


% Train Network
layers = [ ...
    sequenceInputLayer(8)
    lstmLayer(80,OutputMode="sequence")
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    MaxEpochs=100, ...
    MiniBatchSize=32, ...
    Plots="training-progress",...
    InitialLearnRate=0.001,...
    Verbose=0,...
    Shuffle="every-epoch",...
    GradientThreshold=1e5,...
    DispatchInBackground=true);

traindata = readall(trainDS,"UseParallel",true);
rawNet = trainNetwork(traindata(:,1),traindata(:,2),layers,options);

%Classify Testing Signals
testdata = readall(testDS);
predTest = classify(rawNet,testdata(:,1),MiniBatchSize=32);
figure()
confusionchart([testdata{:,2}],[predTest{:}],Normalization="column-normalized")