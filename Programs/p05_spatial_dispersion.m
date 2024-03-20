% Program to compute spatial dispersion of clusters from any clustering
% algorithm output

clc;
clear all;
close all;

%load dataset

ppath='/media/projects/insar_deformation_hotspots/Data';
load([ppath '/output_dbscan.mat']);
load([ppath '/R_TS_den.mat']);
load([ppath '/elpx_ll.mat']);

%% Working with clusters

% data=load([ppath '/cluster_arrays_frame1.mat']);

% Assuming the last column contains cluster IDs
cluster_ids = rs_db_full(:, end);

% Getting unique cluster IDs
unique_cluster_ids = unique(cluster_ids);

% Create a cell array to store separate arrays for each cluster
cluster_arrays = cell(1, numel(unique_cluster_ids));

% Iterate over each unique cluster ID and extract the corresponding rows
for i = 1:numel(unique_cluster_ids)
    current_cluster_id = unique_cluster_ids(i);
    cluster_arrays{i} = rs_db_full(cluster_ids == current_cluster_id, :);
    elpx_ll{i}=rs_db_full(cluster_ids==current_cluster_id, 1:2);
    RTS{i}=rs_db_full(cluster_ids==current_cluster_id, 4:size(rs_db_full,2)-1);
end

elpx_ll=elpx_ll';
RTS=RTS';
cluster_arrays=cluster_arrays';

elpx_ll_full=rs_db_full(:,1:2);

R_TS_full=rs_db_full(:,3:size(rs_db_full,2)-1);


NaN_rows = find(any(isnan(R_TS_full),2));     

R_TS_full(NaN_rows,:)=[];

elpx_ll_full(NaN_rows,:)=[];
% 

mvel=mvel_full';

histogram(mvel); 
xlabel('Mean LOS velocity (cm/year)');
ylabel('Frequency of pixels');


%Find correlation lengths

[X,Y] = latlon2local(elpx_ll_full(:,2), elpx_ll_full(:,1), 0*elpx_ll_full(:,1), [min(elpx_ll_full(:,2))-0.5 min(elpx_ll_full(:,1))-0.5 0]); %in m
X = X/1e3; % m >> km;
Y = Y/1e3; % m >> km;
P = fix(length(R_TS_full) * rand(1e6 , 2))+1;
d = P(: , 2) - P(: , 1);
fd = find(d == 0);
P(fd , :) = [];
L = ((X(P(:,1)) - X(P(:,2))).^2 + (Y(P(:,1)) - Y(P(:,2))).^2).^0.5;
clear fd

Lbin = 0 : 3 : round(max(L));

for i = 1 : length(Lbin)-1
    fd = find(L >= Lbin(i) & L < Lbin(i+1));
    A = R_TS_full(P(fd,1) , :);
    B = R_TS_full(P(fd,2) , :);
    D = A .* B;
%     C(i) = sum(D(:));
    C(i) = sum(D(:)) / (2 * size(B,1) * size(B,2));
end


%% Find standard deviation for clusters

s_avg=meanm(elpx_ll_full(:,2),elpx_ll_full(:,1));

%stad dev of full data

[std_lat,sts_lon] = stdm(elpx_ll_full(:,2),elpx_ll_full(:,1));

%Standard distance - a measure of the dispersion of the data
% in terms of its distance from the geographic mean

%standard distance for full dataset

dist_full = stdist(elpx_ll_full(:,2),elpx_ll_full(:,1));

%computing area for the whole map

wgs84 = wgs84Ellipsoid("km");
a = areaint(elpx_ll_full(:,2),elpx_ll_full(:,1),wgs84);


rd_full=dist_full/a;

%Area calculation for relative standard distance 
% https://www.mathworks.com/help/map/ref/areaint.html

%Standard distance

%Ref: https://www.mathworks.com/help/map/geographic-statistics-for-point-locations-on-a-sphere.html

nc=length(RTS);
c=jet(length(RTS));

% Create a figure handle outside the loop
figure_handle = figure;

for j=1:nc
    disp(['j = ', num2str(j)]);
    disp(['Size of c: ', num2str(size(c))]);

    elpx=elpx_ll{j,1};
% elpx=eval(['elpx_ll_c' num2str(j)]);

    sd(j) = stdist(elpx(:,2),elpx(:,1)); %standard spatial distance 
    a(j)=areaint(elpx(:,2),elpx(:,1),wgs84); %area covered by cluster
    
    rd(j)=sd(j)/a(j); %relative standard distance
    
    c_no=j;
    
    %computing ellipse
    
    Xc=sum(elpx(:,1))/length(elpx);
    
    Yc=sum(elpx(:,2))/length(elpx);
    
    tx=sum((elpx(:,1)-Xc).^2);
    
    ty=sum((elpx(:,2)-Yc).^2);
    
    sdx(j)=sqrt(tx/length(elpx));
    
    sdy(j)=sqrt(ty/length(elpx));
    
    cnt=[Xc Yc]; % ellipse centre
    
    std_elpx=[sdx(j) sdy(j)];
    
    rads=std_elpx;
    % figure;
    
    llc = cnt(:)-rads(:);
    % Compute the width and height
    wh = rads(:)*2; 
    % Draw rectangle 
    h = rectangle('Position',[llc(:).',wh(:).'],'Curvature',[1,1], 'FaceColor','b'); 
    pos=[llc(:).',wh(:).'];
    text(pos(1)+pos(3)/2,pos(2)+pos(4)/2, num2str(j), 'HorizontalAlignment','center', 'FontSize', 12, 'FontWeight','bold', 'Color', c(j,:));
    % set(h,'FaceColor','none','EdgeColor','b','LineWidth',2);
    % h = plotEllipses(center, std_elpx);
    axis equal
    
    % set(h,'FaceColor','none','EdgeColor','b','LineWidth',2);
    h.FaceColor = [0 0 0 0]; %4th value is undocumented: transparency
    h.EdgeColor = c(j,:); 
    h.LineWidth = 2; 
    % title('Cluster', num2str(j));
    
    set(gca,'color',[0.6 0.6 0.6])
    xlabel('Standard distance in x-direction', 'FontSize', 24);
    ylabel('Standard distance in y-direction', 'FontSize', 24)
    
    hold on;

end

hold off;

% Save the figure after the loop
dpi = 300;
filename = [opath '/standard_distance.png'];
print(figure_handle, filename, sprintf('-r%d', dpi), '-dpng');

% Close the figure handle after saving (optional)
close(figure_handle);

%Finding clusters with high standard deviation

ind=find(sd>0.1);
ind2=find(sd<=0.1)';

ind_rd=find(rd<abs(mean(rd)-std(rd)));

% plotting standard distance


figure_handle = figure;
set(figure_handle, 'Units', 'inches', 'Position', [0, 0, 10, 8]); % Adjust size as needed

bar(sd);

xlabel('Cluster label', FontWeight='bold');
ylabel('Standard distance', FontWeight='bold');

% Save the figure after the loop
dpi = 600;
filename = [opath '/standard_distance.png'];
print(figure_handle, filename, sprintf('-r%d', dpi), '-dpng');

close(figure_handle);


% plotting relative distance

figure_handle = figure;
set(figure_handle, 'Units', 'inches', 'Position', [0, 0, 10, 8]); % Adjust size as needed


% figure;
bar(rd);

xlabel('Cluster label', FontWeight='bold');
ylabel('Relative Standard distance', FontWeight='bold');

dpi = 600;
filename = [opath '/relative_distance.png'];
print(figure_handle, filename, sprintf('-r%d', dpi), '-dpng');

close(figure_handle);


% figure;
% boxplot(sd);
% xlabel('Cluster label', FontWeight='bold');
% ylabel('Standard distance', FontWeight='bold');

%% retain clusters with less standard deviation

RTS_rem=RTS(ind2,:);
elpx_ll_rem=elpx_ll(ind2,:);
cluster_arrays_rem=cluster_arrays(ind2,:);


%% combine remaining clusters

% Assuming your matrix name is 'RTS_rem'
totalRows = sum(cellfun(@(x) size(x, 1), RTS_rem)); % Calculate total rows

% Initialize the combined matrix
combined_RTS = zeros(totalRows, size(R_TS_full,2)-1);

rowIndex = 1; % Initialize row index

% Iterate through each cell in the cell array
for i = 1:numel(RTS_rem)
    % Get the current matrix within the cell
    currentMatrix = RTS_rem{i};
    
    % Get the number of rows in the current matrix
    numRows = size(currentMatrix, 1);
    
    % Assign the current matrix to the combined matrix
    combined_RTS(rowIndex:rowIndex+numRows-1, :) = currentMatrix;
    
    % Update the row index for the next iteration
    rowIndex = rowIndex + numRows;
end

% do the same for elpx_ll and cluster arrays

totalRows = sum(cellfun(@(x) size(x, 1), elpx_ll_rem)); % Calculate total rows

% Initialize the combined matrix
combined_elpx_ll = zeros(totalRows, size(elpx_ll_full,2));

rowIndex = 1; % Initialize row index

% Iterate through each cell in the cell array
for i = 1:numel(elpx_ll_rem)
    % Get the current matrix within the cell
    currentMatrix = elpx_ll_rem{i};
    
    % Get the number of rows in the current matrix
    numRows = size(currentMatrix, 1);
    
    % Assign the current matrix to the combined matrix
    combined_elpx_ll(rowIndex:rowIndex+numRows-1, :) = currentMatrix;
    
    % Update the row index for the next iteration
    rowIndex = rowIndex + numRows;
end


totalRows = sum(cellfun(@(x) size(x, 1), cluster_arrays_rem)); % Calculate total rows

% Initialize the combined matrix
combined_cluster_arrays = zeros(totalRows, size(cluster_arrays{1,1},2));

rowIndex = 1; % Initialize row index

% Iterate through each cell in the cell array
for i = 1:numel(cluster_arrays_rem)
    % Get the current matrix within the cell
    currentMatrix = cluster_arrays_rem{i};
    
    % Get the number of rows in the current matrix
    numRows = size(currentMatrix, 1);
    
    % Assign the current matrix to the combined matrix
    combined_cluster_arrays(rowIndex:rowIndex+numRows-1, :) = currentMatrix;
    
    % Update the row index for the next iteration
    rowIndex = rowIndex + numRows;
end

combined_cid=combined_cluster_arrays(:,end);

ncr=length(ind2);

c=c(1:ncr,:);
%% Plotting remaining clusters

figure_handle = figure;
set(figure_handle, 'Units', 'inches', 'Position', [0, 0, 10, 8]); % Adjust size as needed

ds = 1;

ll=combined_elpx_ll;
gauranga=combined_cid;

ax = geoaxes('Basemap','satellite');
hold on
gs = geoscatter(ll(1:ds:end,2), ll(1:ds:end,1),3, gauranga(1:ds:end , 1) , 'o', 'filled'); % cm/yr >> mm/yr
geolimits([min(ll(:,2)) max(ll(:,2))],[min(ll(:,1)) max(ll(:,1))])
% caxis([-0.5 17.5])
% cmp = colormap(clstm);
% cmp = flipud(cmp);
cbh = colorbar ;
colormap(c)

% Save the figure after the loop
dpi = 600;
filename = [opath '/hotspots.png'];
print(figure_handle, filename, sprintf('-r%d', dpi), '-dpng');

close(figure_handle);

% %
% cbh.Ticks = linspace(0, ncr-1, ncr) ; %Create 18 ticks from zero to 17
% cbh.TickLabels = num2cell() ;    %Replace
% % 
% colormap(cmp)
% colorbar
% % 
% cmap=colormap(c);
% cbh=colorbar;

% 
% histogram(coh);
% xlabel('Coherence');
% ylabel('Frequency of pixels');
% 
% inc=elpx_ll(:,3);
% 
% histogram(inc);
% xlabel('Incidence angle (Degrees)');
% ylabel('Frequency of pixels');



rts_cluster=zeros(length(cluster_arrays_rem), size(R_TS_full,2)-1);

for j=1:length(cluster_arrays_rem)
    ts=cluster_arrays_rem{j}(:,4:end-1);
    rts_cluster(j,:)=mean(ts);
end

opath=ppath;

%opath='/media/projects/insar_deformation_hotspots/Output' %alternatively, create a new path

save([opath '/rts_cluster.mat'], 'rts_cluster');

save([opath '/remaining_clusters.mat'], 'cluster_arrays_rem', 'elpx_ll_rem', 'combined_RTS', 'combined_cid', 'combined_elpx_ll','combined_cluster_arrays');

%Plotting average time series clusters
% Create a figure handle outside the loop

% figure;

figure_handle = figure;
set(figure_handle, 'Units', 'inches', 'Position', [0, 0, 10, 8]); % Adjust size as needed

hold on
% j=1;
for k = 1:size(rts_cluster,1)
    plot(rts_cluster(k,:),'-','LineWidth',2, 'color', c(k,:));
end

colormap(c)
legend(num2str(ind2), 'Location', 'best');

% Save the figure after the loop
dpi = 600;
filename = [opath '/cluster_time_series.png'];
print(figure_handle, filename, sprintf('-r%d', dpi), '-dpng');
close(figure_handle);
