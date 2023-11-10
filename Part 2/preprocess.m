function Tsds = preprocess(inputDS)

sig = inputDS{1};
roiTable = inputDS{2};

% Remove first and last rest periods from signal
sig(roiTable.ROI(end,2):end,:) = [];
sig(1:roiTable.ROI(1,1),:) = [];

% Shift ROI limits to account for deleting start and end of signal
roiTable.ROI = roiTable.ROI-(roiTable.ROI(1,1)-1);

% Create signal mask
m = signalMask(roiTable);
L = length(sig);

% Obtain sequence of category labels and remove pronation, supination, and rest motions
mask = catmask(m,L);
idx = ~ismember(mask,{'Pronation','Supination','Rest'});
mask = mask(idx);
sig = sig(idx,:);

% Create new signal mask without pronation and supination categories
m2 = signalMask(mask);
m2.SpecifySelectedCategories = true;
% m2.SelectedCategories = [1 2 3 4 7];
m2.SelectedCategories = [1 2 3 4];
mask = catmask(m2);

% Filter and downsample signal data
sigfilt = bandpass(sig,[10 400],3000);
downsig = downsample(sigfilt,3);

% Downsample label data
downmask = downsample(mask,3);

targetLength = 12000;
% Get number of chunks
numChunks = floor(size(downsig,1)/targetLength);

% Truncate signal and mask to integer number of chunks
sig = downsig(1:numChunks*targetLength,:);
mask = downmask(1:numChunks*targetLength);

% Create a cell array containing signal chunks
sigOut = {};
step = 0;

for i = 1:numChunks
    sigOut{i,1} = sig(1+step:i*targetLength,:)';
    step = step+targetLength;
end

% Create a cell array containing mask chunks
lblOut = reshape(mask,targetLength,numChunks)';
lblOut = num2cell(lblOut,2);
    
% Output a two-column cell array with all chunks
Tsds = [sigOut,lblOut];
end
