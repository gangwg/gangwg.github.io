% Implementation of the Wirtinger Flow (WF) algorithm presented in the paper
% "Phase Retrieval via Wirtinger Flow: Theory and Algorithms"
% by E. J. Candes, X. Li, and M. Soltanolkotabi

% The input data are phaseless measurements about a random complex
% valued 1D signal.


function [Relerrs, z] = WF(y, x, Params, Amatrix)
%% Make signal and data
% n = 100;
% x = randn(n,1) + 1i*randn(n,1);
%
% m = round(5*n);
% A = 1/sqrt(2)*randn(m,n) + 1i/sqrt(2)*randn(m,n);
% y = abs(A*x).^2 ;

m = Params.m;
n = Params.n1;
Relerrs = NaN(Params.T + 1, 1);

%% Initialization
npower_iter = Params.npower_iterWF;           % Number of power iterations

z0 = randn(n, 1); z0 = z0 / norm(z0, 'fro');    % Initial guess
for tt = 1:npower_iter,                     % Power iterations
    z0 = Amatrix' * (y .* (Amatrix * z0)); 
    z0 = z0 / norm(z0, 'fro');
end

normest = sqrt(sum(y(:)) / numel(y(:)));    % Estimate norm to scale eigenvector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

z = normest * z0;                   % Apply scaling
Relerrs(1) = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error

%% Loop
%z = 2*randn(n,1);
tau0 = 330;                         % Time constant for step size
mu = @(t) min(1-exp(-t/tau0), 0.2); % Schedule for step size

for t = 1:Params.T
    yz = Amatrix * z;
    grad  = 1 / m * Amatrix' * ((abs(yz).^2 - y) .* yz); % Wirtinger gradient
    z = z - mu(t)/normest^2 * grad;             % Gradient update
    if Params.faster == 1 && t <= 0.8 * Params.T
        continue;
    end
    Relerrs(t+1) = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro');
    
    if Relerrs(t+1) <= Params.stol || abs(Relerrs(t+1)-Relerrs(t)) <= 1e-1 * Params.stol
        break;
    end
end

Relerrs = Relerrs(1:t+1, 1);



