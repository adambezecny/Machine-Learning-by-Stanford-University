clear ; close all; clc;
%C = [0 2 4 6; 8 10 12 14; 16 18 20 22];
%x = [5 8];
%y = [3 6];
%imagesc(C);
%imagesc(x,y,C);
%colorbar;

initial_theta = zeros(401, 1);
X=[1:5000,1:401];
y=[1:5000];
lambda=0.1;
[J grad] = lrCostFunction(initial_theta, X, y, lambda);

