function [acc, nmi, purity] = evaluate_metrics(true_labels, pred_labels)
% Clustering evaluation metrics calculation function
% Input:
%   true_labels: Ground-truth label vector (n x 1)
%   pred_labels: Predicted cluster label vector (n x 1)
% Output:
%   acc: Clustering Accuracy
%   nmi: Normalized Mutual Information
%   purity: Clustering Purity

    % Convert labels to consecutive integers starting from 1
    true_labels = label_convert(true_labels);
    pred_labels = label_convert(pred_labels);
    
    n = length(true_labels);
    
    % Compute clustering accuracy via optimal label matching
    acc = calculate_accuracy(true_labels, pred_labels);
    
    % Compute normalized mutual information
    nmi = calculate_nmi(true_labels, pred_labels);
    
    % Compute clustering purity
    purity = calculate_purity(true_labels, pred_labels);

end

function labels_new = label_convert(labels)
% Map arbitrary input labels to consecutive integers starting from 1
    unique_labels = unique(labels);
    [~, ~, idx] = unique(labels);
    labels_new = idx;
end

function acc = calculate_accuracy(true_labels, pred_labels)
% Calculate clustering accuracy using optimal label assignment
% Approximates Hungarian algorithm via greedy matching
    n = length(true_labels);
    C = build_confusion_matrix(true_labels, pred_labels);
    
    [m, n_classes] = size(C);
    % Pad confusion matrix to square matrix
    if m > n_classes
        pad_size = m - n_classes;
        C = [C, zeros(m, pad_size)];
    elseif n_classes > m
        pad_size = n_classes - m;
        C = [C; zeros(pad_size, n_classes)];
    end
    
    % Solve linear assignment problem for best label mapping
    [row_idx, col_idx] = find_optimal_assignment(C);
    
    % Total number of correctly matched samples
    matched_count = sum(C(sub2ind(size(C), row_idx, col_idx)));
    acc = matched_count / sum(C(:));
end

function [row_idx, col_idx] = find_optimal_assignment(C)
% Greedy approximate solver for linear assignment problem
% Select highest count pairs without repeated rows/columns
    [rows, cols] = size(C);
    row_idx = [];
    col_idx = [];
    used_rows = false(rows, 1);
    used_cols = false(cols, 1);
    
    C_temp = C;
    
    % Sort all matrix entries in descending order of count
    [values, linear_indices] = sort(C_temp(:), 'descend');
    [all_row, all_col] = ind2sub([rows, cols], linear_indices);
    
    for i = 1:length(values)
        r = all_row(i);
        c = all_col(i);
        
        % Assign pair if row & column are unused and count > 0
        if ~used_rows(r) && ~used_cols(c) && values(i) > 0
            row_idx(end+1) = r;
            col_idx(end+1) = c;
            used_rows(r) = true;
            used_cols(c) = true;
        end
    end
end

function nmi = calculate_nmi(true_labels, pred_labels)
% Compute Normalized Mutual Information (NMI) with log base 2
    unique_true = unique(true_labels);
    unique_pred = unique(pred_labels);
    
    n_true = length(unique_true);
    n_pred = length(unique_pred);
    n_total = length(true_labels);
    
    C = build_confusion_matrix(true_labels, pred_labels);
    
    % Marginal probability distributions
    p_true = sum(C, 2) / n_total;
    p_pred = sum(C, 1) / n_total;
    
    % Filter out zero-probability entries to avoid log2(0)
    p_true = p_true(p_true > 0);
    p_pred = p_pred(p_pred > 0);
    
    % Compute marginal entropies H(true), H(pred)
    H_true = -sum(p_true .* log2(p_true));
    H_pred = -sum(p_pred .* log2(p_pred));
    
    % Compute Mutual Information MI(true; pred)
    MI = 0;
    for i = 1:size(C, 1)
        for j = 1:size(C, 2)
            if C(i, j) > 0
                pij = C(i, j) / n_total;
                pi = sum(C(i, :)) / n_total;
                pj = sum(C(:, j)) / n_total;
                
                MI = MI + pij * log2(pij / (pi * pj));
            end
        end
    end
    
    % Edge case handling for constant labels
    if H_true == 0 && H_pred == 0
        nmi = 1;
    elseif H_true == 0 || H_pred == 0
        nmi = 0;
    else
        nmi = 2 * MI / (H_true + H_pred);
    end
end

function purity = calculate_purity(true_labels, pred_labels)
% Calculate clustering purity metric
    unique_true = unique(true_labels);
    unique_pred = unique(pred_labels);
    
    C = build_confusion_matrix(true_labels, pred_labels);
    
    % Max ground-truth count within each predicted cluster
    max_in_each_cluster = max(C, [], 1);
    % Total samples in each predicted cluster
    total_in_each_cluster = sum(C, 1);
    
    purity_sum = 0;
    for j = 1:size(C, 2)
        if total_in_each_cluster(j) > 0
            purity_sum = purity_sum + max_in_each_cluster(j);
        end
    end
    
    purity = purity_sum / sum(sum(C));
end

function C = build_confusion_matrix(true_labels, pred_labels)
% Construct contingency table (confusion matrix) for true vs predicted labels
% Row: ground truth classes, Column: predicted clusters
    unique_true = unique(true_labels);
    unique_pred = unique(pred_labels);
    
    n_true = length(unique_true);
    n_pred = length(unique_pred);
    
    C = zeros(n_true, n_pred);
    
    for i = 1:length(true_labels)
        true_idx = find(unique_true == true_labels(i));
        pred_idx = find(unique_pred == pred_labels(i));
        C(true_idx, pred_idx) = C(true_idx, pred_idx) + 1;
    end
end