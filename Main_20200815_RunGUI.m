% Main for running the GUI to extrat BCG features
addpath('C:\Users\sumbu\Documents\Research\GUI files')
addpath('C:\Users\sumbu\Documents\Research\data')
DirectoryPath = 'C:\Users\sumbu\Documents\Research\data';    % Enter your path here


load('pig01Data_Abs19')      % run pig06, relative hypo, intervention 5
%load('Feature_Pig_5Rel_5')      % run pig06, relative hypo, intervention 5
run('App_20200802_app2_Tab')    % run GUI