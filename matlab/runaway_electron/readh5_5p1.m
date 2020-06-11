close all
clear all

hinfo = hdf5info('asgard_realspace.h5');
dset = hdf5read(hinfo.GroupHierarchy.Datasets(1));
dset(:,2) = [];
dset = dset(:,[1 2 6 11 16 21 51]);

degree = 4;
level = 4;
domain = [0 10];

grid = create_grid(degree, level, domain);

figure(1)
plot(grid,dset)
axis([domain(1) domain(2) 0 2.5])
ax = gca;
ax.ColorOrderIndex = 2;

t = [0,1,5,10,15,20,50];
si = grid;

% Analytical Solution
E = 2; C = 1; R = 2;
A = E/C;
B = R/C;



e0 = soln(si);
figure(1)
hold on
plot(si,e0,'--','lineWidth',2)

xlabel('$\xi$','fontsize',14,'interpreter','latex')
ylabel('$f(\xi)$','fontsize',14,'interpreter','latex')
labels = {'Initial Condition','Steady State'};
labels = {'t=0','t=1','t=5','t=10','t=15','t=20','t=50','Steady State Soln.'};
legend(labels)
%integral
[x,w] = lgwt(degree,-1,1);
w = 0.5*w/2^level;
h = (domain(end) - domain(1))/2^level;
displacement = 0:1:(2^level - 1);
displacement = repmat(displacement,degree,1);
displacement = h*reshape(displacement,1,[])';
xx = repmat((x+1)*h/2,2^level,1) + displacement;
ww = repmat(w,2^level,1);
integrand = xx.^2.*dset;
integral1 = h/degree*sum(integrand);
% eq = interp1(si,si.^2.*e0,x,'pchip',0);
% int1 = sum(eq.*w);
% soln2 = @(x) 4/sqrt(pi)/8 * x.^2.*exp(-x.^2/2^2);
% dset2 = ic(si);
% int2 = integral(soln2,0,10);
% % hold on
% plot(si,dset2)
figure(2)
plot(t,integral1)
axis([t(1) t(end) 0.9 1.1])


function ret = soln(x)

ret = 4/sqrt(pi) * exp(-x.^2);
end
function ret = ic(x)
a = 2;
ret = 4.0/(sqrt(pi)*a^3) * exp(-x.^2/a^2);
end

function grid = create_grid(degree, level, domain)
h = (domain(2) - domain(1))/2^level;
[x,w] = lgwt(degree,0,h);
grid = [];
for i=0:(2^level-1)
    grid = [grid;i*h + x + domain(1)];
end
end

function [x,w]=lgwt(N,a,b)

% lgwt.m
%
% This script is for computing definite integrals using Legendre-Gauss
% Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
% [a,b] with truncation order N
%
% Suppose you have a continuous function f(x) which is defined on [a,b]
% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
% the values contained in the x vector to obtain a vector f. Then compute
% the definite integral using sum(f.*w);
%
% Written by Greg von Winckel - 02/25/2004
N=N-1;
N1=N+1; N2=N+2;

xu=linspace(-1,1,N1)';

% Initial guess
y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);

% Legendre-Gauss Vandermonde Matrix
L=zeros(N1,N2);

% Derivative of LGVM
Lp=zeros(N1,N2);

% Compute the zeros of the N+1 Legendre Polynomial
% using the recursion relation and the Newton-Raphson method

y0=2;

% Iterate until new points are uniformly within epsilon of old points
while max(abs(y-y0))>eps
    
    
    L(:,1)=1;
    Lp(:,1)=0;
    
    L(:,2)=y;
    Lp(:,2)=1;
    
    for k=2:N1
        L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
    end
    
    Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);
    
    y0=y;
    y=y0-L(:,N2)./Lp;
    
end

% Linear map from[-1,1] to [a,b]
x=(a*(1-y)+b*(1+y))/2;

% Compute the weights
w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;

x=x(end:-1:1);
w=w(end:-1:1);
end