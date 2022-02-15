%% Dataset A
% Extract data from .mat to .xls.
% The original paper does not provide code, we first converted into .xls files and then preprocessed with Python.

% Written by Zenghui Wang, February 15, 2022.

WorkingDir = pwd;
MyMatDir = fullfile(WorkingDir,'matdata'); % raw data 
MySaveDir  = fullfile(WorkingDir,'savedata'); % save data

for t = 1:8
    if t <= 3
        trial = 3;
    else 
        trial = 4;
    end
    name = strcat('S0', num2str(t), '.mat');
    mat_file = load(strcat(MyMatDir, '\', name));
    display(name);
    
    save_path = strcat(MySaveDir, '\' , 'S', num2str(t));
    mkdir(save_path);
    
    for tri = 1: trial
        path = strcat(save_path, '\s', num2str(t), num2str(tri), '_hb');
        xlswrite(path, mat_file.data{1, tri}.X);
        
        path = strcat(save_path, '\s', num2str(t), num2str(tri), '_trial');
        xlswrite(path, mat_file.data{1, tri}.trial);
        
        path = strcat(save_path, '\s', num2str(t),  num2str(tri), '_y');
        xlswrite(path, mat_file.data{1, tri}.y);
    end
end

display('Finish');
