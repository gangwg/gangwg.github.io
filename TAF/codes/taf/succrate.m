%% Example of the truncated Wirtinger Flow (TWF) algorithm under 1D Gaussian designs
% The TWF algorithm is presented in the paper
% ``Solving Random Quadratic Systems of Equations Is Nearly as Easy as Solving Linear Systems'' by Y. Chen and E. J. Candès.
% The code below is adapted from implementation of the Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li, and M. Soltanolkotabi
%clc;
clear
close all;
%% Set Parameters
if exist('Params', 'var')         == 0,  Params.n2          = 1;    end
if isfield(Params, 'n1')          == 0,  Params.n1          = 1000; end             % signal dimension
if isfield(Params, 'm')           == 0,  Params.m           = 0;  end     % number of measurements
if isfield(Params, 'cplx_flag')   == 0,  Params.cplx_flag   = 0;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'grad_type')   == 0,  Params.grad_type   = 'TWF_Poiss';  end     % 'TWF_Poiss': Poisson likelihood

if isfield(Params, 'alpha_lb')    == 0,  Params.alpha_lb    = 0.3;  end
if isfield(Params, 'alpha_ub')    == 0,  Params.alpha_ub    = 5;    end
if isfield(Params, 'alpha_h')     == 0,  Params.alpha_h     = 5;    end
if isfield(Params, 'alpha_y')     == 0,  Params.alpha_y     = 3;    end
if isfield(Params, 'T')           == 0,  Params.T           = 201;  end    	% number of iterations
if isfield(Params, 'mu')          == 0,  Params.mu          = 0.2;  end		% step size / learning parameter % originally 0.2
if isfield(Params, 'npower_iter') == 0,  Params.npower_iter = 50;   end		% number of power iterations

% if isfield(Params, 'tau_y')       == 0,  Params.tau_y       = 0.38 * (1 - Params.cplx_flag) + 0.5 * Params.cplx_flag;   end	% 0.43 for real .48 for complex larger threshold for small dimension (0.6 for <=10) 0.5 for larger dimension
if isfield(Params, 'imax')        == 0,  Params.imax        = 201;   end	% number of iterations for iNull
if isfield(Params, 'jmax')        == 0,  Params.jmax        = 1;  end	% number of monte carlo simulations
if isfield(Params, 'gamma_lb')    == 0,  Params.gamma_lb    = 1.5;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'gamma_ub')    == 0,  Params.gamma_ub    = 1;   end	% thresholding of throwing small entries of Az 0.3 is a good one

if isfield(Params, 'imu')         == 0,  Params.imu          = .6 * (1 - Params.cplx_flag) + 1 * Params.cplx_flag;  end		% step size / learning parameter % originally 0.2


cplx_flag	= Params.cplx_flag;  % real-valued: cplx_flag = 0;  complex-valued: cplx_flag = 1;
display(Params);
tol = 1e-5;
Tmax = 51;
T = Params.T;

TWF_success = zeros(1, Tmax);
TAF_success = zeros(1, Tmax);
WF_success  = zeros(1, Tmax);
AF_success  = zeros(1, Tmax);

% initial_error_null = zeros(Params.jmax, 1);
% initial_error_twf  = zeros(Params.jmax, 1);
n = Params.n1;

delta = 0.1;

for t = 31
    
    display(t);
    %     if t <= 11
    %         delta = 0.1;
    %     end
    Params.m = (.9 + t * delta) * Params.n1;
    m        = floor(Params.m);
    success1 = 0;
    success2 = 0;
    success3 = 0;
    success4 = 0;

    for j = 1:Params.jmax
        
        %% Make signal and data (noiseless)
        x = 1 * randn(n, 1)  + cplx_flag * 1i * randn(n, 1);
%         x = x / norm(x);
        var = 0.1 * norm(x);
        Amatrix = (1 * randn(m, n) + cplx_flag * 1i * randn(m, n)) / (sqrt(2)^cplx_flag);
        A  = @(I) Amatrix  * I;
        At = @(Y) Amatrix' * Y;
        noise = var * randn(m, 1);
        y  = abs(A(x) + noise ).^2; 
                
%             y  = poissrnd(y);
        %     sinr = sum(abs(noise)) / sum(abs(A(x)).^2);
        %     display('-----------------SNR----------------');
        %     display(sinr);
        
        if t >= 8
            
            % Check results and Report Success/Failure
            [Relerrs_TWF] = TWF(y, x, Params, A, At);
            T = Params.T;
            
            %   initial_error_twf(j) = Relerrs_twf(1);
            if Relerrs_TWF(T+1) <= tol
                success1 = success1 + 1;
            end
            
        end
        
        if t >= 3
            
            [Relerrs_TAF, z] = TAF(y, x, Params, Amatrix); %iNULL_tr_test
            T = Params.imax;
            
            %   initial_error_null(j) = Relerrs_iNull_tr(1);
            if Relerrs_TAF(T+1) <= tol
                success2 = success2 + 1;
            end
        end

        if t >= 3
            
            [Relerrs_AF, z] = AF(y, x, Params, Amatrix); %iNULL_tr_test
            T = Params.imax;
            
            if Relerrs_AF(T+1) <= tol
                success4 = success4 + 1;
            end
        end
        
        if t >= 10
            
            [Relerrs_WF] = WF(y, x, Params, Amatrix);
            T = Params.T;
            
            %   initial_error_twf(j) = Relerrs_twf(1);
            if Relerrs_WF(T+1) <= tol
                success3 = success3 + 1;
            end
            
        end
        
    end
    
    TWF_success(t) = success1 / Params.jmax;
    TAF_success(t) = success2 / Params.jmax;
    AF_success(t)  = success4 / Params.jmax;
    WF_success(t)  = success3 / Params.jmax;
    
end

width = 1.6;

semilogy(0:T-1, Relerrs_WF(1:T), '-kx', 'LineWidth', width, 'MarkerSize', eps);
hold on
semilogy(0:T-1, Relerrs_TWF(1:T), '-bs', 'LineWidth', width, 'MarkerSize', eps);
% semilogy(Relerrs_AF(1:1000), '-m+', 'LineWidth', width, 'MarkerSize', eps);
semilogy(0:T-1, Relerrs_TAF(1:T), '-rv', 'LineWidth', width, 'MarkerSize', eps);

xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('Relative error vs. iteration count')
grid

% save irate140_65

% display('Success rate: TWF, TGGF, WF, AF');
% display(TWF_success);
% display(TAF_success);
% display(WF_success);
% display('Computation times: TWF, iNull, iNull_tr');
% display([sum(TWFtimes), sum(iNulltimes), sum(iNulltimes_tr)]);
%
% display('average initialization error: tsp vs null');
% display([mean(initial_error_twf), mean(initial_error_null)]);

