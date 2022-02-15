%% Dataset C
% Preprocess data and convert format from .mat to .xls.
% Most of MATLAB functions are available in BBCI toolbox, visit BBCI toolbox (https://github.com/bbci/bbci_public)
% fNIRS-T uses preprocessed data, fNIRS-PreT uses raw data (without band-pass filtering and baseline correction)

% Original code is written by Jaeyoung Shin (jyshin34@wku.ac.kr) 08. Sep. 2019
% Modified by Zenghui Wang, February 15, 2022.

%% 1. Run BBCI toolbox
format compact
disp('1. run BBCI toobox')

% set paths
WorkingDir = pwd;
MyToolboxDir = fullfile(WorkingDir,'BBCI'); % toolbox path
MyDatDir  = fullfile(WorkingDir,'data');
MyMatDir  = fullfile(MyDatDir,'matdata'); % fNIRS dataset path
MySaveDir  = fullfile(MyDatDir,'predata'); % preprocessed data path

% run BBCI toolbox
addpath(MyToolboxDir);
startup_bbci_toolbox('DataDir', MyDatDir, 'MatDir', MyMatDir,...
    'TmpDir','/tmp/', 'History', 0);

disp(BTB.MatDir);

for t = 1:30
    %% 2. Load example fNIRS dataset
    disp('2. load example fNIRS dataset')
    % file name: e.g.) fNIRS 04.mat
    num = num2str(t);
    if t <= 9
        file = strcat('fNIRS 0', num, '.mat');
    else
        file = strcat('fNIRS',32, num, '.mat');
    end
    
    % load fNIRS dataset file to MATLAB workspace
    [cntHb, mrk, mnt] = file_loadMatlab(file);

    %% 3. Band-pass filtering to eliminate physiological noises  
    disp('3. band-pass filtering')
    
    % zero-order band-pass filtering using [ord]-order Butterworth IIR filter
    % with passpand of [band]
    ord = 3;
    band = [0.01 0.1]/cntHb.fs*2;
    [b, a] = butter(ord, band, 'bandpass');
    cntHb = proc_filtfilt(cntHb, b, a); 

    %% 4. Segmentation
    disp('4. segmentation')
    
    % segment cntHb into epochs ranging [ival_epo]
    ival_epo = [-1 25]*1000; % msec
    epo = proc_segmentation(cntHb, mrk, ival_epo);
    
    %% 5. Baseline correction
    disp('5. baseline correction')

    % baseline correction using reference interval of [ival_base]
    ival_base = [-1 0]*1000; % msec
    epo = proc_baseline(epo, ival_base);

    %% 6. Save preprocessed data
    disp('6. Save preprocessed data')
    mkdir(strcat(MySaveDir,'\', num));
    path = strcat(MySaveDir,'\', num ,'\',num);
    for i=1:75
        xlswrite(path, epo.x(:, :, i), i, 'A1');
    end
    path = strcat(MySaveDir,'\', num ,'\',num, '_', 'desc');
    xlswrite(path, epo.event.desc);
end

disp('Finish');
% remove toolbox path
rmpath(MyToolboxDir);
