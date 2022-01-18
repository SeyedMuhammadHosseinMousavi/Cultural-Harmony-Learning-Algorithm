%% Cultural Harmony Learning Algorithm - Created in 18 Jan 2022 by Seyed Muhammad Hossein Mousavi
% Here is all about learning with evolutionary algorithms. Harmony search
% and cultural algorithms are two fast optimization algorithms which their
% result are combined here in order to train inputs for targets in a simple
% dataset. Basically, system starts with making initial fuzzy model and fit
% the outputs based on inputs by harmony search first and then tries to fit the
% harmony search outputs with inputs in the second stage. That means we are using both
% evolutionary algorithms to improve the accuracy. System easily could be
% used for regression, classification and other optimization tasks. You can
% use your data and play with parameters.
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)

%% Cleaning
clc;
clear;
warning('off');
%% Data Loading
data=JustLoad();
%% Generate Basic Fuzzy Model
% Number of Clusters in FCM
ClusNum=4; 
%
fis=GenerateFuzzy(data,ClusNum);
%
%% Tarining Cultural Harmony Algorithm
% Harmony Search Learning
HarFis=hars(fis,data);        
% Harmony Cultural Algorithm Learning
CAHSfis=CulturalFCN(HarFis,data);        

%% Plot Cultural Harmony Results (Train - Test)
% Train Output Extraction
TrTar=data.TrainTargets;
TrainOutputs=evalfis(data.TrainInputs,CAHSfis);
% Test Output Extraction
TsTar=data.TestTargets;
TestOutputs=evalfis(data.TestInputs,CAHSfis);
% Train calc
Errors=data.TrainTargets-TrainOutputs;
MSE=mean(Errors.^2);RMSE=sqrt(MSE);  
error_mean=mean(Errors);error_std=std(Errors);
% Test calc
Errors1=data.TestTargets-TestOutputs;
MSE1=mean(Errors1.^2);RMSE1=sqrt(MSE1);  
error_mean1=mean(Errors1);error_std1=std(Errors1);
% Train
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1);
plot(data.TrainTargets,'c');
hold on;
plot(TrainOutputs,'k');legend('Target','Output');
title('Cultural Harmony Training Part');xlabel('Sample Index');grid on;
% Test
subplot(3,2,2);
plot(data.TestTargets,'c');
hold on;
plot(TestOutputs,'k');legend('Cultural Harmony Target','Cultural Harmony Output');
title('Cultural Harmony Testing Part');xlabel('Sample Index');grid on;
% Train
subplot(3,2,3);
plot(Errors,'k');legend('Cultural Harmony Training Error');
title(['Train MSE =     ' num2str(MSE) '  ,     Train RMSE =     ' num2str(RMSE)]);grid on;
% Test
subplot(3,2,4);
plot(Errors1,'k');legend('Cultural Harmony Testing Error');
title(['Test MSE =     ' num2str(MSE1) '  ,    Test RMSE =     ' num2str(RMSE1)]);grid on;
% Train
subplot(3,2,5);
h=histfit(Errors, 50);h(1).FaceColor = [.1 .2 0.9];
title(['Train Error Mean =   ' num2str(error_mean) '  ,   Train Error STD =   ' num2str(error_std)]);
% Test
subplot(3,2,6);
h=histfit(Errors1, 50);h(1).FaceColor = [.1 .2 0.9];
title(['Test Error Mean =   ' num2str(error_mean1) '  ,   Test Error STD =    ' num2str(error_std1)]);

%% Plot Just Fuzzy Results (Train - Test)
% Train Output Extraction
fTrainOutputs=evalfis(data.TrainInputs,fis);
% Test Output Extraction
fTestOutputs=evalfis(data.TestInputs,fis);
% Train calc
fErrors=data.TrainTargets-fTrainOutputs;
fMSE=mean(fErrors.^2);fRMSE=sqrt(fMSE);  
ferror_mean=mean(fErrors);ferror_std=std(fErrors);
% Test calc
fErrors1=data.TestTargets-fTestOutputs;
fMSE1=mean(fErrors1.^2);fRMSE1=sqrt(fMSE1);  
ferror_mean1=mean(fErrors1);ferror_std1=std(fErrors1);
% Train
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1);
plot(data.TrainTargets,'m');hold on;
plot(fTrainOutputs,'k');legend('Target','Output');
title('Fuzzy Training Part');xlabel('Sample Index');grid on;
% Test
subplot(3,2,2);
plot(data.TestTargets,'m');hold on;
plot(fTestOutputs,'k');legend('Target','Output');
title('Fuzzy Testing Part');xlabel('Sample Index');grid on;
% Train
subplot(3,2,3);
plot(fErrors,'g');legend('Fuzzy Training Error');
title(['Train MSE =     ' num2str(fMSE) '   ,    Test RMSE =     ' num2str(fRMSE)]);grid on;
% Test
subplot(3,2,4);
plot(fErrors1,'g');legend('Fuzzy Testing Error');
title(['Train MSE =     ' num2str(fMSE1) '   ,    Test RMSE =     ' num2str(fRMSE1)]);grid on;
% Train
subplot(3,2,5);
h=histfit(fErrors, 50);h(1).FaceColor = [.3 .8 0.3];
title(['Train Error Mean =    ' num2str(ferror_mean) '   ,   Train Error STD =    ' num2str(ferror_std)]);
% Test
subplot(3,2,6);
h=histfit(fErrors1, 50);h(1).FaceColor = [.3 .8 0.3];
title(['Test Error Mean =    ' num2str(ferror_mean1) '   ,   Test Error STD =    ' num2str(ferror_std1)]);

