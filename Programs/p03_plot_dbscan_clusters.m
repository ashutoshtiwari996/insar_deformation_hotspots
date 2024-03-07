clc
clear all
close all


ppath = '/media/ashutosh/Data/Leonard/JavaIsland/Data';     % path to root directory

load([ppath '/output_dbscan_frame3.mat']);

% load([ppath '/radheshyam_predicted_ts_clusters_frame2.mat']);

elpx_ll=lonlat_full;
MVel=mvel_full;

% idx=find(isnan(MVel));
% 
% elpx_ll(idx,:)=[];
% MVel(idx)=[];
% load([ppath '/MVel']);
% load([ppath '/MVelq.mat']);
% 
% load([ppath '/coh.mat']);
% inc=elpx_ll(:,3);

% xlim = [-80.604 -80.31];
% ylim = [37.1 37.3];

% xlim = [-80.468 -80.351];
% ylim = [37.173 37.276];
% 
% fd = find (elpx_ll(:,1) >= xlim(1) & elpx_ll(:,1) <= xlim(2) & elpx_ll(:,2) >= ylim(1) & elpx_ll(:,2) <=ylim(2));
% elpx_ll = elpx_ll(fd,:);
% MVel = MVel(fd,:);

% MVel = MVel ./  cosd(elpx_ll(:,3))  ; % projecting on dz and accounting for reference point offset

%% plotting every 100s pixel to reduce load

% fid=isnan(MVel);
% % 
% MVel(fid)=[];
% % 
% elpx_ll(fid,:)=[];
% 
% MVel=yhat_full(:,1);
% y_predn_full=y_pred_dba_X;

nc=length(unique(y_predn_full));


y=y_predn_full';

ll=lonlat_full;

ds = 1;
figure
ax = geoaxes('Basemap','satellite');
hold on
gs = geoscatter(ll(1:ds:end,2), ll(1:ds:end,1),5, y(1:ds:end , 1) , 'o', 'filled'); % cm/yr >> mm/yr
geolimits([min(ll(:,2)) max(ll(:,2))],[min(ll(:,1)) max(ll(:,1))])
% geoplot(37.2083 , -80.4553, '+w')
% colormap(c)

caxis([0 nc])
cmp = colormap(jet(nc));
cmp = flipud(cmp);
colormap(cmp)
colorbar
title('Clustering')


MVel=MVel';

ds = 1;
figure
ax = geoaxes('Basemap','satellite');
hold on
gs = geoscatter(elpx_ll(1:ds:end,2), elpx_ll(1:ds:end,1),5, MVel(1:ds:end , 1) , 'o', 'filled'); % cm/yr >> mm/yr
geolimits([min(elpx_ll(:,2)) max(elpx_ll(:,2))],[min(elpx_ll(:,1)) max(elpx_ll(:,1))])
% geoplot(37.2083 , -80.4553, '+w')
% colormap(c)

% caxis([0 nc])
cmp = colormap(jet);
cmp = flipud(cmp);
colormap(cmp)
colorbar
title('Mean vertical land motion (mm/year)')
