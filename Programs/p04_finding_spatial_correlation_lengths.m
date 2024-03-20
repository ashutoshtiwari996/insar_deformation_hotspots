
clc;
clear all;
close all;

%load dataset

ppath='/media/projects/insar_deformation_hotspots/Data';
load([ppath '/output_dbscan.mat']);
load([ppath '/R_TS_den.mat']);
% load([ppath '/elpx_ll.mat']);

elpx_ll=rs_db_full(:,1:2);

R_TS=rs_db_full(:,3:size(rs_db_full,2)-1);


NaN_rows = find(any(isnan(R_TS),2));     

R_TS(NaN_rows,:)=[];

elpx_ll(NaN_rows,:)=[];


[X,Y] = latlon2local(elpx_ll(:,2), elpx_ll(:,1), 0*elpx_ll(:,1), [min(elpx_ll(:,2))-0.5 min(elpx_ll(:,1))-0.5 0]); %in m
X = X/1e3; % m >> km;
Y = Y/1e3; % m >> km;
P = fix(length(R_TS) * rand(1e5 , 2))+1;
d = P(: , 2) - P(: , 1);
fd = find(d == 0);
P(fd , :) = [];
L = ((X(P(:,1)) - X(P(:,2))).^2 + (Y(P(:,1)) - Y(P(:,2))).^2).^0.5;
clear fd


Lbin = 0 : 3 : 130;


for i = 1 : length(Lbin)-1
    fd = find(L >= Lbin(i) & L < Lbin(i+1));
    A = R_TS(P(fd,1) , :);
    B = R_TS(P(fd,2) , :);
    D = A .* B;
%     C(i) = sum(D(:));
    C(i) = sum(D(:)) / (2 * size(B,1) * size(B,2));
end

% clear A B C D fd;

figure; plot(Lbin(1:length(Lbin)-1), C, LineWidth=2); 
xlabel("Distance");
ylabel("Covariance")