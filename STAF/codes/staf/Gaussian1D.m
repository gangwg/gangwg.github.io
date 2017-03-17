%% Implementation of the Stochatsic Truncated Amplitude Flow algorithm proposed in the paper
%  ``Solving Large-scale Systems of Random Quadratic Equations via Stochastic
% Truncated Amplitude Flow'' by G. Wang, G. B. Giannakis, and J. Chen
% The code below is adapted from implementation of the Truncated Wirtinger Flow
% algorithm designed and implemented by E. Candes, X. Li, and M. Soltanolkotabi

clc;
clear;
close all;

if exist('Params', 'var')           == 0,  Params.n2            = 1;    end
if isfield(Params, 'n1')            == 0,  Params.n1            = 1000; end             % signal dimension
if isfield(Params, 'm')             == 0,  Params.m             = floor(2. * Params.n1) ;  end     % number of measurements
if isfield(Params, 'cplx_flag')     == 0,  Params.cplx_flag     = 0;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'gamma_ub')      == 0,  Params.gamma_sgd     = .7;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'eta')           == 0,  Params.eta           = .5;   end
if isfield(Params, 'npower_iterVR') == 0,  Params.npower_iterVR = 60;   end		% number of power iterations
if isfield(Params, 'imuSGD')        == 0,  Params.imuSGD        = .6;  end		% step size / learning parameter % originally 0.2

display(Params);
n     = Params.n1;
m     = floor(Params.m);

%% Make signal and data (noiseless)
Amatrix = (1 * randn(m, n) + Params.cplx_flag * 1i * randn(m, n)) / (sqrt(2)^Params.cplx_flag);
x       = 1 * randn(n, 1)  + Params.cplx_flag * 1i * randn(n, 1);
A       = @(I) Amatrix  * I;
At      = @(Y) Amatrix' * Y;
var     = 1e-5; % noiseless would correspond to var = 0
noise   = var * randn(m, 1);
y       = abs(A(x) + noise).^2;

[Relerrs_STAF, z] = STAF(y, x, Params, Amatrix); %iNULL_tr_test
disp('----------STAF done!---------');

figure
semilogy(Relerrs_STAF)
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('STAF: Relerr vs. itercount')

