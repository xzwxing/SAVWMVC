
function [P,F,a,obj_values]= SAVWMVC(X, Y, lambda1,lambda2, eta, mu1, mu_Max, max_iter, tol)

beta1 = 0.9;  beta2 = 0.999;
epsilon = 1e-10; 
rho = 1.1; 
mu2 = 1;
V=length(X);
[N, L] = size(Y); 
OneL =ones(L,1);

XXt=cell(1,V);
for v=1:V
    XXt{v}=X{v}*X{v}';
end

%% Initialize variables
P = cell(1, V); 
a = cell(1, V); 
X_hat = cell(1, V); 
rng(0)
for v = 1:V
    % P{v}   =   ones(size(X{v},1)+1,L) /sqrt( size(X{v},1)+1 ); 
    a{v}   =   ones(size(X{v},1),1) /size(X{v}+1,1);
    P{v}   =   rand(size(X{v},1)+1,L) /sqrt( size(X{v},1)+1 ); 
    X_hat{v} = [ones(1,N); X{v}];
end

Q = P;  % Initialize Q = P
F = ones(N, V)/V; %Initialize F, ensure that the sum of F(i,:) is 1

m_F = zeros(size(F));  %Initialize the first-order moment estimation of the gradient
v_F = zeros(size(F));  %Initialize the second-order moment estimation of the gradient
m_Q = cell(1, V); v_Q = cell(1, V); m_av=cell(1, V); v_av=cell(1, V);
for v = 1:V
    m_Q{v} = zeros(size(Q{v}));
    v_Q{v} = zeros(size(Q{v}));   
    m_av{v} = zeros(size(a{v}));
    v_av{v} = zeros(size(a{v}));
end

% Initialize Lagrange multipliers
Delta = cell(1, V);
for v = 1:V
    Delta{v} = zeros(size(P{v}));
end

obj_values = zeros(max_iter, 1);
for iter = 1:max_iter
    % Calculate loss function
    Y_out = zeros(size(Y));  fXav = zeros(size(F));
    for v = 1:V       
        temp = X_hat{v}'*P{v};
        Y_out = Y_out + sparse( diag(F(:,v)) ) * temp;
        fXav(:,v) = X{v}'*a{v};
    end
    loss = 0.5*sum( sum( (Y_out - Y).^2 ) );
        
    % Add regularization term
    obj_value = loss + 0.5*lambda1 * sum( sum(( F- fXav ).^2)); %+ lambda2 * sum( sum(abs(PP)) );
    obj_values(iter) = obj_value;
    
%     if iter==1 || mod(iter,100)==0 || iter==max_iter
%         fprintf('Iteration %d/%d, Objective function value: %f\n', iter, max_iter, obj_value);
%     end
    
    % Check convergence
    if iter > 1 && abs(obj_values(iter) - obj_values(iter-1)) < tol
       %  fprintf('The algorithm has converged at %d iteration\n', iter);
        break;
    end
    
    %% Update F 
    grad_F = zeros(size(F));  Temp = zeros(size(Y));
    for k=1:V
        Temp = Temp + sparse( diag(F(:,k)) ) * X_hat{k}'*Q{k};            
    end
    for v=1:V
        grad_F(:,v) = grad_F(:,v) + ( ( Temp -Y) .* ( X_hat{v}'* Q{v}) )* OneL;
    end
    grad_F =grad_F + lambda1*( F - fXav ) -mu1./(F.^2);

    m_F = beta1 * m_F + (1 - beta1) * grad_F;
    v_F = beta2 * v_F + (1 - beta2) * (grad_F.^2);
    m_F_hat = m_F / (1 - beta1^iter);
    v_F_hat = v_F / (1 - beta2^iter);
    F = F - eta * m_F_hat ./ (sqrt(v_F_hat) + epsilon);
    F = max(F, 0);  % ensure F > 0 
    mu1=0.99*mu1;
    
    F = simplex_projection(F); %Project F onto the probability space
       
    %% Update a_v
   for v=1:V
        grad_av = lambda1*( XXt{v}*a{v}-X{v}*F(:,v) ) ;   
        m_av{v} = beta1 * m_av{v} + (1 - beta1) * grad_av;
        v_av{v} = beta2 * v_av{v} + (1 - beta2) * (grad_av.^2);
        m_av_hat = m_av{v} / (1 - beta1^iter);
        v_av_hat = v_av{v} / (1 - beta2^iter);
        a{v} = a{v} - (eta * m_av_hat) ./ (sqrt(v_av_hat) + epsilon);
   end
    
    %% Update Q and P
    Temp = zeros(size(Y));
    for k=1:V
       Temp = Temp + sparse( diag(F(:,k)) ) * X_hat{k}'*Q{k};            
    end    
    for v = 1:V    
       grad_Qv = X_hat{v} * sparse( diag(F(:,v) ) ) * (Temp - Y) +  Delta{v} + mu2*( Q{v} - P{v} );        
        m_Q{v} = beta1 * m_Q{v} + (1 - beta1) * grad_Qv;
        v_Q{v} = beta2 * v_Q{v} + (1 - beta2) * (grad_Qv.^2);
        m_Q_hat = m_Q{v} / (1 - beta1^iter);
        v_Q_hat = v_Q{v} / (1 - beta2^iter);
        Q{v} = Q{v} - (eta * m_Q_hat) ./ (sqrt(v_Q_hat) + epsilon);
    end

      
       %% Update P
    for v=1:V 
        temp = Q{v} + Delta{v}/mu2;
        P{v} = wthresh(temp, 's', lambda2/mu2);
    end 

      %% Update Delta  
    for v=1:V 
        Delta{v} = Delta{v} + mu2 * (Q{v} - P{v});
    end
    
     mu2=min(rho*mu2,mu_Max);
end


% figure;
% plot(1:max_iter, obj_values(1:max_iter));
