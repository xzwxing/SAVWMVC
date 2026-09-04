clear,clc

addpath(genpath('./'));
namelist = dir('./data/*.mat');
Pr={0.005, 5000, 100, 0.001;% parametres for 3sources 
    0.005, 5000, 100, 0.001; % parametres for BBCSport
    0.0005, 5000, 200, 0.01;%for COIL20_3VIEWS
    0.0005, 5000, 200, 0.01;%for MSRC-v1
    0.0005, 5000, 200, 0.01;%for NUS
    0.0005, 5000, 200, 0.01;%for ORL
    0.0005, 5000, 200, 0.01;%for UCI_3view
    0.01, 5000, 200, 0.01;%for WebKB
    0.005, 5000,100, 0.001;%for YaleB_first10
    0.005, 5000, 100, 0.001;%for Youtube
    0.001, 5000, 2000, 0.001;%for scene-15
    0.0005, 5000, 200, 0.01};%for yale

m = length(namelist);
Results = cell(m,1); results = cell(m,11);

parpool(4); %Open parallel computing pool

for i = 1:m
    filename = namelist(i).name;
    
    [eta, max_iter, lambda1, lambda2] = deal(Pr{i,:});
    data=load(filename);
    X=data.X; Y=data.Y;
    clear data
        
        %%%%%%%%%%% ====Data Preprocessing==== %%%%%%%% 
    zeroColumns = [];
    for v=1:length(X)   % delete all-zero columns (samples), which are an invalid samples.
        X{v} =  X{v}';
        temp = find( all( X{v} == 0, 2)); X{v}(temp',:)=[];
        temp = find( all( X{v} == 0, 1));
        zeroColumns = [zeroColumns,temp ];
    end

    if ~isempty(zeroColumns)
        Y(zeroColumns)=[];% Exclude the labels corresponding to all-zero samples.
    end

    for v=1:length(X)
        X{v}(:,zeroColumns)=[];
        X{v}= ( X{v}-min(min(X{v})) )/( max(max(X{v}))-min(min(X{v})) ); % Normalization.
    end

      %%%%% ==== 10 folds cross validation ====%%%%%
    rng(0);
    N = length(Y);
    L = max(Y); 
    k_folds = 10; % Ten fold
    fold_indices = crossvalind('Kfold', Y, k_folds);
    
    acc_list = []; % Store the results of each fold.
    nmi_list = [];
    purity_list = [];
   
    fprintf('========== Start 10 fold cross validation  ==========\n');
     
    tic;
   
    parfor fold = 1:k_folds
        % fprintf('\n------------  %d fold: ------------\n', fold);
         % The k-th fold is used as the test set, and the rest is used as the training set        
          test_idx = (fold_indices == fold);
          train_idx = ~test_idx;
   
          % Divide training/testing
        X_train=cell(1,length(X)); X_test=cell(1,length(X));
        for v=1:length(X)
             X_train{v} = X{v}(:, train_idx);
             X_test{v}  = X{v}(:, test_idx);
        end
        Y_train = Y(train_idx);
        Y_test  = Y(test_idx);
    

        gnd1=repmat(Y_train,1,L);
        Y_train1=zeros(size(gnd1));
    
        for j=1:L
           Ind=(gnd1(:,j)==j);
           Y_train1(Ind,j)=1;
        end
    
          %% fixed parameters in the model
         mu_max = 5000; tol1 = 1e-6; mu1 = 0.99;
    
         %% Train the model
        % fprintf('Train the %d fold model ...\n', fold);
        [P,F,a,obj_values] = SAVWMVC(X_train, Y_train1,lambda1,lambda2, eta,mu1,mu_max, max_iter, tol1);

    
         %% test model
         % fprintf('Test the %d fold model ...\n', fold);
         Y_out = zeros(length(Y_test),L);
         for v=1:length(X_test)
             Y_out = Y_out + sparse( diag( X_test{v}'*a{v} ) ) * [ones(1,size(X_test{v},2));X_test{v}]'*P{v};
         end
         [~,Y_pred]=max(Y_out,[],2);
         [acc, nmi, purity] = evaluate_metrics(Y_test', Y_pred');
    
         acc_list = [acc_list, acc];
         nmi_list = [nmi_list, nmi];
        purity_list = [purity_list, purity];
        
        fprintf('The %d fold results: acc=%.4f,mi=%.4f,purity=%.4f\n', fold, acc, nmi, purity);
    end

       %% ===== Calculate the ten fold average value ===========
    avg_acc = mean(acc_list);
    avg_nmi = mean(nmi_list);
    avg_purity = mean(purity_list);

    fprintf('\n========== Ten fold cross validation result ==========\n');
    fprintf('average ACC = %.4f\n', avg_acc);
    fprintf('average NMI = %.4f\n', avg_nmi);
    fprintf('average Purity = %.4f\n', avg_purity);
    fprintf('=========================================\n');

      %% Save the average value
    results =  {
            filename, ...
            avg_acc, ...
            avg_nmi, ...
            avg_purity, ...
            eta, max_iter, lambda1, lambda2, ...
            acc_list, nmi_list};  % Save detailed results for each discount.

      toc           
      %%%%%%%%%%%%%%%%%%%    
    
    Results{i,1} = results;
    disp(Results{i,1})   
end

delete(gcp('nocreate'));% Delete the current parallel computing pool
