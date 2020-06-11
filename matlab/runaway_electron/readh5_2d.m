% close all
% clear all

hinfo = hdf5info('asgard_realspace.h5');
[dset info] = hdf5read(hinfo.GroupHierarchy.Datasets(1));
% dset(:,2) = [];

degree = 3;
level = 4;
domain_p = [0 10];
domain_xi = [-1 1];
p = create_grid(degree, level, domain_p);
xi = create_grid(degree, level, domain_xi);
size_dset = size(dset);
n = 2^level*degree;

for i=1:20:size_dset(2)
    [pp,xx] = meshgrid(p,xi);
    p_par = pp.*xx;
    p_perp = pp.*sqrt(1-xx.^2);
figure(i)
this_data = reshape(dset(:,i),n,[]);
h = pcolor(xi,p,this_data)
% contour(p_par,p_perp,this_data,linspace(0,0.5,400))
h.EdgeColor = 'none';
colorbar


end

colorbar
% set(gca, 'YDir', 'normal')
 set(gca, 'XScale', 'log')
title({'title'})
xlabel('E [eV]') % x-axis label
ylabel('Angle [degrees]') % y-axis label
set(gca,'fontsize',16)


t = [0,0.5,1,1.5,2,2.5,3];
si = linspace(-0.999,0.999,length(dset));
for i=1:length(t)
    phi = tanh(atanh(si) - t(i));
    analytic = (1-phi.^2)./(1-si.^2).*exp(-phi.^2/.01);
    analytic(isnan(analytic)) = 0;
    figure(1)
    hold on
    plot(si,analytic,'--','lineWidth',2)
end
xlabel('$\xi$','fontsize',14,'interpreter','latex')
ylabel('$f(\xi)$','fontsize',14,'interpreter','latex')
labels = {'t=0','t=0.5','t=1.0','t=1.5','t=2.0','t=2.5','t=3.0'};
legend(labels)

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