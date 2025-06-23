close all 
clear
clc
format short % for better visualization
hold all;

load('dataScenario2v3.mat')

Veicolo = VeicoloOK;%(1:100); % congugation
VeicoloNot = VeicoloOKNoT;%(1:300); % normal traffic

numVehicles1 = size(Veicolo,2);
numVehicles2 = size(VeicoloNot,2);
numVehiclesVar = size(VeicoloNot{1},2);



rng(1);
cv = cvpartition(numVehicles1,'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
VeicoloTrain = Veicolo(:,~idx);
VeicoloTest  = Veicolo(:,idx);

cv1 = cvpartition(numVehicles2 ,'HoldOut',0.3);
idx1 = cv1.test;
% Separate to training and test data
VeicoloNotTrain = VeicoloNot(:,~idx1);
VeicoloNotTest  = VeicoloNot(:,idx1);


size_train_1=size(VeicoloTrain,2); % congugation 0
size_test_1=size(VeicoloTest,2);
size_train_2=size(VeicoloNotTrain,2); % normal tarffic 1
size_test_2=size(VeicoloNotTest,2);


lengthSamples = 10;
numSamples = 60;
%offset=1;



X_1 = [];
X_2=[];

for i=1:size_train_1
    lenght_i=size(VeicoloTrain{i},1);
    X=cell2mat(VeicoloTrain{i});

    temp_x = lagmatrix(X(:,5), -(lenght_i-numSamples):lengthSamples:0);
    x=transpose(temp_x(1:numSamples,:));

    %temp_x = lagmatrix(X(:,5), offset1:-lengthSamples:-(lenght_i-numSamples));
    %x=transpose(temp_x(1:numSamples,:));

    X_1 = [X_1;x];
    
end
Y_1 = zeros(size(X_1,1),1);

for i=1:size_train_2
    lenght_i=size(VeicoloNotTrain{i},1);
    X=cell2mat(VeicoloNotTrain{i});

    temp_x = lagmatrix(X(:,5), -(lenght_i-numSamples):lengthSamples:0);
    x=transpose(temp_x(1:numSamples,:));

    %temp_x = lagmatrix(X(:,5), offset1:-lengthSamples:-(lenght_i-numSamples));
    %x=transpose(temp_x(1:numSamples,:));

    X_2 = [X_2;x];
    
end
Y_2 = ones(size(X_2,1),1);

X_train=[X_1;X_2];
Y_train=[Y_1;Y_2];

%Permutation
numSamples_train = size(X_train, 1);
perm = randperm(numSamples_train);
X_train = X_train(perm, :);
Y_train = Y_train(perm, :);

X_11 = [];
X_21=[];


for i=1:size_test_1
    lenght_i=size(VeicoloTest{i},1);
    X=cell2mat(VeicoloTest{i});

    temp_x = lagmatrix(X(:,5), -(lenght_i-numSamples):lengthSamples:0);
    x=transpose(temp_x(1:numSamples,:));

    %temp_x = lagmatrix(X(:,5), offset1:-lengthSamples:-(lenght_i-numSamples));
    %x=transpose(temp_x(1:numSamples,:));

    X_11 = [X_11;x];
    
end
Y_11 = zeros(size(X_11,1),1);

for i=1:size_test_2
    lenght_i=size(VeicoloNotTest{i},1);
    X=cell2mat(VeicoloNotTest{i});

    temp_x = lagmatrix(X(:,5), -(lenght_i-numSamples):lengthSamples:0);
    x=transpose(temp_x(1:numSamples,:));

    %temp_x = lagmatrix(X(:,5), offset1:-lengthSamples:-(lenght_i-numSamples));
    %x=transpose(temp_x(1:numSamples,:));

    X_21 = [X_21;x];
    
end
Y_21 = ones(size(X_21,1),1);
X_test=[X_11;X_21];
Y_test=[Y_11;Y_21];

%Permutation
numSamples_test = size(X_test, 1);
perm = randperm(numSamples_test);
X_test = X_test(perm, :);
Y_test = Y_test(perm, :);


% Linear regression


X_data = [X_train; X_test];
Y_data = [Y_train; Y_test];
eta =1 ;

Input = [X_data ones(size(X_data,1),1)];

%% Encoding
Y = zeros(length(Y_data),2);
Y(Y_data==0,1) = 1; % normal traffic
Y(Y_data==1,2) = 1; % congugation

%% Least-Square for Linear Classifiers
W = zeros(61,2);
W(:,1) = Input\Y(:,1);
W(:,2) = Input\Y(:,2);

%% Validation
Y_hat = Input*W;
[~,C_hat] = max(Y_hat,[],2);

% missclassification 
MissClass0 = 1-sum(C_hat(Y_data==0)==1)/sum((Y_data==0))
MissClass1 = 1-sum(C_hat(Y_data==1)==2)/sum((Y_data==1))

% Learning curve
% Define the range of training set sizes
X_data_size= size(X_data,1);
trainSetSizes_regression = [int64(X_data_size*0.1) int64(X_data_size*0.2) int64(X_data_size*0.3) int64(X_data_size*0.4) int64(X_data_size*0.5) int64(X_data_size*0.6) int64(X_data_size*0.7) int64(X_data_size*0.8) int64(X_data_size*0.9) X_data_size] % Modify as needed

% Initialize arrays to store the performance metrics
accuracy_list_regression = zeros(1, length(trainSetSizes_regression));

% Loop through different training set sizes
for i = 1:length(trainSetSizes_regression)
    % Select a subset of the training data
    SetSize = trainSetSizes_regression(i);
    X_d = X_data(1:SetSize, :);
    Y_d = Y_data(1:SetSize);

    Input_curve = [X_d ones(size(X_d,1),1)];

    %% Encoding
    Y = zeros(length(Y_d),2);
    Y(Y_d==0,1) = 1; % normal traffic
    Y(Y_d==1,2) = 1; % congugation

    %% Least-Square for Linear Classifiers
    W_curve = zeros(61,2);
    W_curve(:,1) = Input_curve\Y(:,1);
    W_curve(:,2) = Input_curve\Y(:,2);

    %% Validation
    Y_hat_curve = Input_curve*W_curve;
    [~,C_hat_curve] = max(Y_hat_curve,[],2);
    
    % Evaluate the performance metrics
    accuracy_list_regression(i) = (sum(C_hat_curve(Y_d==0)==1) + sum(C_hat_curve(Y_d==1)==2)) / size(Y_d,1);
end
disp(accuracy_list_regression)

% Plot the learning curve
plot(trainSetSizes_regression, accuracy_list_regression, 'o-');
xlabel('Training Set Size');
ylabel('Accuracy');
title('Learning Curve - Linear Regression');
grid on;

% Classification tree
% Use the fitctree function

model = fitctree(X_train, Y_train);
view(model, 'mode', 'graph')
predictions = model.predict(X_test);

accuracy = sum(predictions == Y_test) / size(Y_test,1);
disp("Accuracy for Classification Tree");
disp(accuracy)

% Learning curve
% Define the range of training set sizes
X_train_size= size(X_train,1);
trainSetSizes = [int64(X_train_size*0.1) int64(X_train_size*0.2) int64(X_train_size*0.3) int64(X_train_size*0.4) int64(X_train_size*0.5) int64(X_train_size*0.6) int64(X_train_size*0.7) int64(X_train_size*0.8) int64(X_train_size*0.9) X_train_size] % Modify as needed

% Initialize arrays to store the performance metrics
accuracy_list = zeros(1, length(trainSetSizes));

% Loop through different training set sizes
for i = 1:length(trainSetSizes)
    % Select a subset of the training data
    trainSize = trainSetSizes(i);
    X_t = X_train(1:trainSize, :);
    Y_t = Y_train(1:trainSize);
    
    % Train the classification tree
    tree = fitctree(X_t, Y_t);
   
    % Predict the output for the test data
    Y_pred = predict(tree, X_test);
    
    % Evaluate the performance metrics
    accuracy_list(i) = sum(Y_pred == Y_test) / size(Y_test,1);
    disp(sum(Y_pred == Y_test) / size(Y_test,1));
end
disp(accuracy_list)

% Plot the learning curve
plot(trainSetSizes, accuracy_list, 'o-');
xlabel('Training Set Size');
ylabel('Accuracy');
title('Learning Curve - Classification Tree');
grid on;




%SVM
cl = fitcsvm(X_train,Y_train,'KernelFunction','rbf','Standardize',true,Verbose=1);%'rbf','ClassNames',[0,1]);
prediction=predict(cl,X_test);
acc= sum(prediction==Y_test)./size(Y_test);
misclass1 = 1-sum(prediction == Y_test)/length(Y_test);

LearningCurve('SVM', X_train,Y_train,X_test,Y_test,0)

%Random Forest
acc_f=zeros(1,5);
misclass_f= zeros(1,5);
NRSME= zeros(1,5);
i=1;
for numTrees = [1 10 50 100 200 ]
    forest=TreeBagger(numTrees,X_train,Y_train,'Method','classification');
    prediction_ff= forest.predict( X_test);
    prediction_f = str2double(prediction_ff);
    
    acc_f(i)= sum(prediction_f==Y_test)/size(Y_test,1);
    misclass_f(i)= 1-sum(prediction_f == Y_test)/length(Y_test);
    NRSME(i) = sqrt(sum((prediction_f-Y_test).^2)/length(Y_test));
    i=i+1;

    LearningCurve('Random Forest', X_train,Y_train,X_test,Y_test,numTrees)
end 


function LearningCurve(flag, X_train,Y_train,X_test,Y_test,numTrees)
    % Learning curve
    % Define the range of training set sizes
    X_train_size= size(X_train,1);
    trainSetSizes = [int64(X_train_size*0.1) int64(X_train_size*0.2) int64(X_train_size*0.3) int64(X_train_size*0.4) int64(X_train_size*0.5) int64(X_train_size*0.6) int64(X_train_size*0.7) int64(X_train_size*0.8) int64(X_train_size*0.9) X_train_size]; % Modify as needed

    % Initialize arrays to store the performance metrics
    accuracy_list = zeros(1, length(trainSetSizes));

    % Loop through different training set sizes
    for i = 1:length(trainSetSizes)
        % Select a subset of the training data
        trainSize = trainSetSizes(i);
        X_t = X_train(1:trainSize, :);
        Y_t = Y_train(1:trainSize);
    
        % Train the classification tree
        if strcmp(flag,'SVM')
            t = fitcsvm(X_t,Y_t,'KernelFunction','rbf','Standardize',true,Verbose=1);
            Y_pred = predict(t, X_test);
        end
        if strcmp(flag,'Random Forest')
            forest=TreeBagger(numTrees,X_t,Y_t,'Method','classification');
            prediction_ff= forest.predict( X_test);
            Y_pred = str2double(prediction_ff);

        end
   
    
         % Evaluate the performance metrics
        accuracy_list(i) = sum(Y_pred == Y_test) / size(Y_test,1);
        disp(sum(Y_pred == Y_test) / size(Y_test,1));
    end
    disp(accuracy_list)
    %disp(plus('Learning Curve - ' , flag))

    % Plot the learning curve
    figure
    plot(trainSetSizes, accuracy_list, 'o-');
    xlabel('Training Set Size');
    ylabel('Accuracy');
    %title('Learning Curve - ' + flag);
    grid on;
end

