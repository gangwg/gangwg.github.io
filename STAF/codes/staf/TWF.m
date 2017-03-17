%% Implementation of the truncated Wirtinger Flow (TWF) algorithm proposed in the paper
%  ``Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems'' by Y. Chen and E. J. Candès.
%  The code below is adapted from implementation of the Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li, and M. Soltanolkotabi

function [Relerrs, z] = TWF(y, x, Params, A, At)
%% Initialization
npower_iter = Params.npower_iterWF;           % Number of power iterations
z0 = randn(Params.n1,Params.n2); z0 = z0/norm(z0,'fro');    % Initial guess
normest = sqrt(sum(y(:))/numel(y(:)));    % Estimate norm to scale eigenvector
Relerrs = NaN(Params.T + 1, 1);

for tt = 1:npower_iter,                     % Truncated power iterations
    ytr = y.* (abs(y) <= Params.alpha_y^2 * normest^2);
    z0 = At( ytr.* (A(z0))); z0 = z0/norm(z0,'fro');
end

z = normest * z0;                   % Apply scaling
Relerrs(1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error


%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Arnorm = sqrt(sum(abs(Amatrix).^2, 2)); % norm of rows of Amatrix
% ysqr = sqrt(y);
% ynorm = ysqr ./ (Arnorm .* normest);
%     z = x + 2 * (rand(Params.n1, 1) - .5);
% Relerrs = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x, 'fro'); % Initial rel. error
%
% [ys, ind] = sort(ysqr, 'descend');
% [norm(ysqr - abs(Amatrix * z), 'inf'), ys(Params.n1)]
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Loop
grad_type = Params.grad_type;
if strcmp(grad_type, 'TWF_Poiss') == 1
    mu = @(t) Params.mu; % Schedule for step size
elseif strcmp(grad_type, 'WF_Poiss') == 1
    tau0 = 330;                         % Time constant for step size
    mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size
end

for t = 1: Params.T,
    grad = compute_grad(z, y, Params, A, At);
    z = z - mu(t) * grad;             % Gradient update
    if Params.faster == 1 && t <= 0.8 * Params.T
        continue;
    end
    Relerrs(t+1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
   
    if Relerrs(t+1) <= Params.stol || abs(Relerrs(t+1)-Relerrs(t)) <= 1e-1 * Params.stol
        break;
    end
end

Relerrs = Relerrs(1:t+1, 1);


