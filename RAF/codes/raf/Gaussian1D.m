%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Solving Systems of Random Quadratic Equations via Truncated Amplitude
%  Flow'' by G. Wang, G. B. Giannakis, and Y. C. Eldar.
%  The code below is adapted from implementation of the (Truncated) Wirtinger Flow
% algorithm implemented by E. Candes, X. Li, M. Soltanolkotabi, and Y. Chen.
clear;
clc;
close all;

if exist('Params', 'var')           == 0,  Params.n2            = 1;    end
if isfield(Params, 'n1')            == 0,  Params.n1            = 10000; end             % signal dimension
if isfield(Params, 'm')             == 0,  Params.m             = floor(2 * Params.n1) - 1;  end     % number of measurements
if isfield(Params, 'cplx_flag')     == 0,  Params.cplx_flag     = 0;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'T')             == 0,  Params.T             = 1000;  end    	% number of iterations
if isfield(Params, 'mu')            == 0,  Params.mu            = 0.6 * (1 - Params.cplx_flag) + 1 * Params.cplx_flag;  end		% step size / learning parameter
if isfield(Params, 'gamma_lb')      == 0,  Params.gamma_lb      = 1;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'npower_iter')   == 0,  Params.npower_iter   = 200;   end		% number of power iterations
if isfield(Params, 'eta')           == 0,  Params.eta           = 3;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'alpha')         == 0,  Params.alpha         = 0.5;   end	% thresholding of throwing small entries of Az 0.3 is a good one

if isfield(Params, 'grad_type')     == 0,  Params.grad_type     = 'TWF_Poiss';  end     % 'TWF_Poiss': Poisson likelihood
if isfield(Params, 'alpha_lb')      == 0,  Params.alpha_lb      = 0.3;  end
if isfield(Params, 'alpha_ub')      == 0,  Params.alpha_ub      = 5;    end
if isfield(Params, 'alpha_h')       == 0,  Params.alpha_h       = 5;    end
if isfield(Params, 'alpha_y')       == 0,  Params.alpha_y       = 3;    end
if isfield(Params, 'muTWF')         == 0,  Params.muTWF         = 0.2;  end		% step size / learning parameter % originally 0.2
if isfield(Params, 'muRAF')         == 0,  Params.muRAF         = 2 * (1 - Params.cplx_flag) + 4 * Params.cplx_flag;  end

display(Params);

%% Make signal and data (noiseless)
Amatrix = (1 * randn(Params.m, Params.n1) + Params.cplx_flag * 1i * randn(Params.m, Params.n1)) / (sqrt(2)^Params.cplx_flag);
x       = 1 * randn(Params.n1, 1) + Params.cplx_flag * 1i * randn(Params.n1, 1);

A       = @(I) Amatrix  * I;
At      = @(Y) Amatrix' * Y;

var     = 0e-1;
noise   = var * randn(Params.m, 1);
y       = (abs(A(x)) + noise).^2;

%% run TAF algorithm
timetaf = tic;
[Relerrs_TAF] = TAF1D(y, x, Params, Amatrix); 
times(1) = toc(timetaf);
disp('----------TAF done!----------');

%% plot the relative error of TAF
figure,
semilogy(Relerrs_TAF)
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('TAF: Relerr vs. itercount')
grid

% timetwf = tic;
% [Relerrs_TWF] = TWF(y, x, Params, Amatrix); 
% times(2) = toc(timetwf);
% disp('----------TWF done!----------');

%% run RAF algorithm
timeraf = tic;
[Relerrs_RAF, z] = RAF1D(y, x, Params, Amatrix); 
times(3) = toc(timeraf);
disp('----------RAF done!----------');

%% plot the relative error of TAF
figure,
semilogy(Relerrs_RAF)
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('RAF: Relerr vs. itercount')
grid


