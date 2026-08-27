addpath(genpath('./'));
namelist = dir('./data/*.mat');
Pr={0.01, 5000, 200, 0.01;
    0.005, 5000, 100, 0.001;
    0.0005, 5000, 200, 0.01;
    0.001, 5000, 300, 0.01; 
    };
m = length(namelist);
n = size(Pr,1);
% parpool(n); 
Results = cell(m,1);
for i = 1:m
    results = cell(n,11);
    filename = namelist(i).name;
    
    parfor j = 1 : n
        [eta, max_iter, lambda1, lambda2] = deal(Pr{j,:});
        results(j,:)  = SAVWMVC(filename,eta,max_iter, lambda1, lambda2);   
    end
    Results{i,1} = results;
    disp(Results{i,1})
end


