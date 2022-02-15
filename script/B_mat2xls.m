%% Dataset B
% Preprocess data and convert format from .mat to .xls.
% Most of MATLAB functions are available in BBCI toolbox, visit BBCI toolbox (https://github.com/bbci/bbci_public)
% specify your nirs data directory (NirsMyDataDir), temporary directory (TemDir) and working directory (WorkingDir)
% Save preprocessed data directory (MySaveDir)
% fNIRS-T uses preprocessed data, fNIRS-PreT uses raw data (without filtering and baseline correction)

% Modified by Zenghui Wang, February 15, 2022.

clear all; close all;  clc; 
% specify your own directory name 
WorkingDir = pwd;
MyToolboxDir = fullfile(WorkingDir,'bbci_public-master');
NirsMyDataDir = fullfile(WorkingDir,'NIRS');
TemDir = fullfile(WorkingDir,'temp');
MySaveDir  = fullfile(WorkingDir,'predata'); % preprocessed data path

cd(MyToolboxDir);
startup_bbci_toolbox('DataDir',NirsMyDataDir,'TmpDir',TemDir, 'History', 0);
cd(WorkingDir);

for idx = 1 : 29
    filename = num2str(idx);
    if idx <= 9
        file = strcat('subject 0', filename);
    else
        file = strcat('subject',32, filename);
    end
    disp(file);
    subdir_list = {file};
    %subdir_list = {'subject 01'}; % for subject 1
    basename_list = {'motor_imagery1','mental_arithmetic1','motor_imagery2','mental_arithmetic2','motor_imagery3','mental_arithmetic3'}; % task type: motor imagery / recording session: 1 - 3

    % load nirs data
    loadDir = fullfile(NirsMyDataDir, subdir_list{1});
    cd(loadDir);
    load cnt; load mrk; load mnt;% load continous eeg signal (cnt), marker (mrk) and montage (mnt)

    cd(WorkingDir);

    % temporary variable
    cnt_temp = cnt; mrk_temp = mrk;
    clear cnt mrk;

    % Merge cnts
    [cnt.ment, mrk.ment] = proc_appendCnt({cnt_temp{2}, cnt_temp{4}, cnt_temp{6}}, {mrk_temp{2}, mrk_temp{4}, mrk_temp{6}}); % for mental arithmetic

    % MBLL
    cnt.ment = proc_BeerLambert(cnt.ment);

    % filtering
    [b, a] = butter(3, [0.01 0.1]/cnt.ment.fs*2);
    cnt.ment = proc_filtfilt(cnt.ment, b, a);

    % divide into HbR and HbO, cntHb uses same structure with cnt
    cntHb.ment.oxy   = cnt.ment; 
    cntHb.ment.deoxy = cnt.ment; 

    % replace data
    cntHb.ment.oxy.x = cnt.ment.x(:,1:end/2); 
    cntHb.ment.oxy.clab = cnt.ment.clab(:,1:end/2);
    cntHb.ment.oxy.clab = strrep(cntHb.ment.oxy.clab, 'oxy', ''); % delete 'oxy' in clab
    cntHb.ment.oxy.signal = 'NIRS (oxy)';

    cntHb.ment.deoxy.x = cnt.ment.x(:,end/2+1:end); 
    cntHb.ment.deoxy.clab = cnt.ment.clab(:,end/2+1:end);
    cntHb.ment.deoxy.clab = strrep(cntHb.ment.deoxy.clab, 'deoxy', ''); % delete 'deoxy' in clab
    cntHb.ment.deoxy.signal = 'NIRS (deoxy)'; 

    % epoching
    ival_epo = [-10 25]*1000; % from -10000 to 25000 msec relative to task onset (0 s)
    epo.ment.oxy   = proc_segmentation(cntHb.ment.oxy, mrk.ment, ival_epo);
    epo.ment.deoxy = proc_segmentation(cntHb.ment.deoxy, mrk.ment, ival_epo);

    % baseline correction
    ival_base = [-5 -2]*1000;
    epo.ment.oxy   = proc_baseline(epo.ment.oxy, ival_base);
    epo.ment.deoxy = proc_baseline(epo.ment.deoxy, ival_base);

    %% save preprocessed data
    num = num2str(idx);
    mkdir(strcat(MySaveDir, '\',num));
    % save HbO
    path = strcat(MySaveDir,'\', num ,'\',num, '_', 'oxy');
    for i=1:60
        xlswrite(path, epo.ment.oxy.x(:, :, i), i, 'A1');
    end

    % save HbR
    path = strcat(MySaveDir,'\',num ,'\',num, '_', 'deoxy');
    for i=1:60
        xlswrite(path, epo.ment.deoxy.x(:, :, i), i, 'A1');
    end

    path = strcat(MySaveDir,'\', num ,'\',num, '_', 'desc');
    xlswrite(path, epo.ment.oxy.event.desc);
end

disp('MA finish');
