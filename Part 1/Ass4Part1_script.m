% SYDE 544 Assignment 4 Part 1 Script

clear all
close all

load Ass4Part1_data;

% simulation duration
duration = 10;
% Problem 1a begins:
% generate the recruitment thresholds (RTE) (Equation 1,2)
RR = 30;
a = log(RR) / 120;
for i = 1:120
RTE(i) = exp(a * i); end
% normalize, so the maximal RTE is 1
RTE = RTE/30;
%Plot RTE (include labels and title)
figure()
plot(1:120, RTE)
title('RTE vs MU #')
xlabel('MU #')
ylabel('Normalized RTE')
% Problem 1a ends


% Problem 1b begins:
% calculate the firing rate gain 
% minimal firing rate is 8 Hz, maximal rate is 35 Hz
g = (35-8)./(1.2-RTE); 
%Plot g
figure()
plot(1:120,g)
title('g vs MU #')
xlabel('MU #')
ylabel('g')
% Problem 1b ends


% Problem 2 begins
% Generate the firing timings for all units at the 9 neural drive levels
%E is the neural drive levels normalized to the maximal RTE
E = [3,6,9,12,15,18,21,24,27]./30;

for i = 1:length(RTE) %for each MU
    for j = 1:length(E) 
        if RTE(i) <= E(j)
            firingRates(i,j) = g(i)*(E(j) - RTE(i)) + 8;
            mean= 1/firingRates(i,j);
            std = 0.15 * mean;
            
            % firing times
            firingTimes(i,j,1) = normrnd(1/firingRates(i,j), std);
            k = 1;
            while firingTimes(i,j,k) < duration
                k = k+1;
                norm= normrnd(1/firingRates(i,j), std);
                firingTimes(i,j,k) = firingTimes(i,j,k-1) + norm;
            end
        else
            firingRates(i,j) = 0;
        end;
    end;
end;
%Plot mean firing rates
figure()
for i = 1:size(firingRates,2) 
    subplot(1,9,i);
    plot(1:120, firingRates(:,i));
    ylim([0,30])
    title(sprintf('Neural Drive Level %d',i))
    xlabel('MU #')
    ylabel('Mean Firing Rate')
end
% Problem 2 ends

% Problem 3 begins
% plot MUAPs for quick investigation
freq = 4096;
MUAPtimes = 0:1/freq:(size(MUAPs,1)-1)/freq;
figure()
plot(MUAPtimes, MUAPs(:,:))
title('MUAPs')
xlabel('Time(s)')
ylabel('Action Potential Signal Strength')

% creating the MUAP trains
% timestamps of firing impulse trains
t = 0:1/4096:10; % time array 4096 hz for 10 seconds
u = zeros(size(RTE,2),size(firingTimes,2),length(t)); % empty array for series of impulse (120 MU and 9 Neural drive levels)
MUAPtrain = zeros(size(u)); % empty array for series of impulse (120 MU and 9 Neural drive levels)

for i=1:size(firingTimes,1) %for each activated MU
    for j=1:size(firingTimes,2) % for each neural drive
        for k =1:size(firingTimes,3)% for each firing time
            %for each firing time, find time that is closest and set
            %impulse to 1 (true)
            [M,index] = (min(abs(t-firingTimes(i,j,k)))); 
            u(i,j,index) = 1; 
        end
    end
end

% generating MUAP trains
% take the convolution of the MUAP with the series of impulse for each
% 120MU and 9 neural drive levels
MUAPtrains = zeros(length(RTE),length(E),length(t)); %create empty trains
for muID = 1:120
    for drive = 1:9
        MUAPtrains(muID,drive,:) = conv(squeeze(u(muID,drive,:)), squeeze(MUAPs(:,muID)), 'same'); %squeeze to remove first dimension
   
    end
end

% simulated EMG signals as the summation of all MUAP trains
EMG = squeeze(sum(MUAPtrains,1));

% plotting the EMG signal at each neural drive level
figure;
for i = 1:size(EMG,1) 
    subplot(1,9,i);
    plot(t,EMG(i,:))
    ylim([-2,2])
    title(sprintf('Neural Drive Level %d',i))
    xlabel('Time (s)')
    ylabel('EMG Signal Strength')
end
% Problem 3 ends

% Problem 4 begins
% plot force twitches for quick investigation

%  F= force twitches of the 120 MUs (each column for a MUAP). 
% For each MU, a firing will generate one force twitch. 
% The sampling rate is also 4096 sample per second.
times = 0:1/freq:(size(F,1)-1)/freq; 
figure()
plot(times, F(:,:))
ylim([0,100])
title('Force Twitches')
xlabel('Time(s)')
ylabel('Force Twitch Strength')

% creating force twitches
for muID = 1:120
    for drive = 1:9
        % find force output of each MU at each neural drive level by taking 
        % convolution of impulses with the force twitch
        %forceTwt(muID,drive,:) =  conv(u(muID,drive,:), F(:,muID)', 'same');
        forceTwt(muID,drive,:) = conv(squeeze(u(muID,drive,:)), squeeze(F(:,muID)), 'same');
    end
end
% overall force output is the sum of 120 MU force outputs
Force = squeeze(sum(forceTwt,1));
% plotting the force output at each neural drive level
figure;
for i = 1:size(Force,1) 
  subplot(1,9,i);
    plot(t,Force(i,:))
    ylim([0,7500])
    title(sprintf('Neural Drive Level %d',i))
    xlabel('Time (s)')
    ylabel('Force Signal Strength') 
end
% Problem 4 ends       


