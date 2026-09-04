function X = simplex_projection(Y)
% The code is obtained from literature: 
% Wang W, Carreira-Perpinan M A. Projection onto the probability simplex: An
% efficient algorithm with a simple proof, and an application. arXiv:1309.1541, Sept. 3 2013.

%The output of X satisfies sum(X(i,:))=1,i=1,2,...,N; 
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
end
