    function [U, S, V, res] = LanSVD_min(X, dd, msteps) 
%%  function [U, S, V, res] = LanSVD(X, dd, msteps) 
%%  simple LANCZOS procedure for the SVD
%%  *not* the Golub-Kahan diagonalization.
%%  does Lanczos on X'*X [but X'*X not formed]
%%  We assume here that m>n -- otherwise
%%  it is best to get the approximate svd of
%%  X' and then transpose everything
%%  input:
%%   X      = matrix
%%   dd     = number of desired singular vectors
%%   msteps = number of Lanczos steps to run
%%  On return
%%   U = left singular vectors [can be commented out below
%%   S = diagonal of singular values
%%   V = right singular vectors

[m,n] =size(X);
%%-------------------- pre-allocating
p    = msteps+1;
VV   = zeros(n,p);
Tmat = zeros(p,p);
%%-------------------- initializing
 v  = randn(n,1);
 v = v/norm(v,2); 
 beta = 0; 
 VV(:,1)= v; 
 vold = v;
 orthTol = 1.e-12;
 wn = 0.0 ;
for k=1:msteps
    w = X'*(X*v) - beta*vold ; 
    alpha = w'*v; 
    wn = wn + alpha*alpha;
    T(k,k) = alpha; 
    w = w - alpha*v;
%%-------------------- reorth. *** NOT NEEDED
    w = w - VV* (VV'*w);
%% above line
    beta = w'*w;
%%-------------------- test for exit
    if (beta*k < orthTol*wn) 
        break
    end
    wn = wn+2.0*beta;
    beta = sqrt(beta) ;
    vold = v; 
    v = w / beta;
    VV(:,k+1) = v;
    T(k,k+1) = beta; 
    T(k+1,k) = beta; 
end
%fprintf(1,'k == %d n = %d m = %d  beta %e \n',k, n,m,beta);
 dd = min(dd,k);
%%-------------------- rr procedure
 C = T(1:k,1:k); 
 [Q, S2] = eig(C);
 res =  beta*abs(Q(k,:));
 [lam, indx] = sort(diag(S2),'ascend');
%%-------------------- get right sing. vectors 
 V = VV(:,1:k)*Q(:,indx(1:dd));
 res = res(indx);
 res = res(:);
%%-------------------- get left sing vectors  
 U = X*V;
 %%-------------------- scale columns of U 
 d = sqrt(sum(U .* U,1));
 S = diag(d);
 U = U * diag( 1.0 ./ d);
%%-------------------- SVD at this point is U*S*V' 
%%------------------------------------------------------
