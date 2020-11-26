%%
 
%   Written by "Kais Suleiman" (ksuleiman.weebly.com)
%
%   Notes:
%
%   - The contents of this script have been explained in details 
%   in Chapter 4 of the thesis:
%       Kais Suleiman, "Popular Content Distribution in Public 
%       Transportation Using Artificial Intelligence Techniques.", 
%       Ph.D. thesis, University of Waterloo, Ontario, Canada, 2019.
%   - Simpler but still similar variable names have been used throughout 
%   this script instead of the mathematical notations used in the thesis.
%   - The assumptions used in the script are the same as those used in 
%   the thesis including those related to the case study considered 
%   representing the Grand River Transit bus service offered throughout 
%   the Region of Waterloo, Ontario, Canada.
%   - The following external MATLAB functions have been used throughout 
%   the script:
%       - M Sohrabinia (2020). LatLon distance 
%       (https://www.mathworks.com/matlabcentral/fileexchange/
%       38812-latlon-distance), 
%       MATLAB Central File Exchange. Retrieved November 16, 2020.
%       - Arturo Serrano (2020). Normalized histogram 
%       (https://www.mathworks.com/matlabcentral/fileexchange/
%       22802-normalized-histogram), 
%       MATLAB Central File Exchange. Retrieved November 16, 2020.
%       - Patrick Mineault (2020). Unique elements in cell array 
%       (https://www.mathworks.com/matlabcentral/fileexchange/
%       31718-unique-elements-in-cell-array), 
%       MATLAB Central File Exchange. Retrieved November 16, 2020.
%   - Figures and animations are created throughout this script to aid 
%   in thesis visulizations and other forms of results sharing.

%%

clear all; %#ok<*CLALL>
close all;
clc;

%   Generating data:

modified_trips = xlsread('modified_trips');
save('modified_trips.mat','modified_trips');

stop_times = xlsread('stop_times');
save('stop_times.mat','stop_times');

stops = xlsread('stops');
save('stops.mat','stops');

shapes = xlsread('shapes');
save('shapes.mat','shapes');

%%

clear all; %#ok<*CLALL>
close all;
clc;

load('modified_trips');
load('stop_times');
load('stops');

%   Collecting data:

data = zeros(size(stop_times,1),12);

handle = waitbar(0,'Please wait ... ');

for i = 1:size(stop_times,1)
    
    data(i,:) = horzcat( ...
        modified_trips(modified_trips(:,3) == stop_times(i,1),[6,1,5]), ...
        stop_times(i,1:5), ...
        stops(stops(:,1) == stop_times(i,4),5:6), ...
        modified_trips(modified_trips(:,3) == stop_times(i,1),[7,2]));
    
    waitbar(i / size(stop_times,1),handle)
    
    if i == size(stop_times,1)
        
        close(handle)
        
    end
    
end

%   Collecting weekday data only:

data = data(data(:,12) == 0,:);
data(:,12) = []; 

save('data.mat','data');

%%

clear all;
close all;
clc;

load('data');

%   Sorting data:

data = sortrows(data,[1,5]);

%   Re-sorting same-time row data with switching directions:

%   Re-sorting while assuming end-of-trip:

for i = 2:size(data,1) - 1
    
   if (data(i,3) ~= data(i - 1,3)) && (data(i,3) ~= data(i + 1,3))
      
       block_data = data(data(:,1) == data(i,1),:);
           
       same_time_block_data = block_data(block_data(:,5) == data(i,5),:);
           
       if data(i,3) == 0
           
           %    Sorting in a descending order:
           
           same_time_block_data = sortrows(same_time_block_data,-3);
           
       else
           
           %    Sorting in an ascending order:
           
           same_time_block_data = sortrows(same_time_block_data,3);
           
       end
       
       block_data(block_data(:,5) == data(i,5),:) = same_time_block_data;
       
       data(data(:,1) == data(i,1),:) = block_data;
       
   end
    
end

%   Re-sorting while assuming beginning-of-trip:

for i = 2:size(data,1) - 1
    
   if (data(i,3) ~= data(i - 1,3)) && (data(i,3) ~= data(i + 1,3))
      
       block_data = data(data(:,1) == data(i,1),:);
           
       same_time_block_data = block_data(block_data(:,5) == data(i,5),:);
           
       if data(i,3) == 0
           
           %    Sorting in an ascending order:
           
           same_time_block_data = sortrows(same_time_block_data,3);
           
       else
           
           %    Sorting in a descending order:
           
           same_time_block_data = sortrows(same_time_block_data,-3);
           
       end
       
       block_data(block_data(:,5) == data(i,5),:) = same_time_block_data;
       
       data(data(:,1) == data(i,1),:) = block_data;
       
   end
    
end

data = unique(data,'rows','stable');

save('data.mat','data');

%%

clear all;
close all;
clc;

load('data');

%   Computing speeds with noisy trips included:

data = horzcat(data,zeros(size(data,1),1));

handle = waitbar(0,'Please wait ... ');

for i = 2:size(data,1)
    
    if data(i,1) == data(i - 1,1)
        
        distance_difference = ...
            lldistkm([data(i,9) data(i,10)], ...
            [data(i - 1,9) data(i - 1,10)]);
        
        if distance_difference == 0
            
            data(i,12) = 0;
            
        else
            
            time_difference = ...
                (data(i,5) - data(i - 1,6)) * 24;
            
            if time_difference == 0
                
                data(i,12) = NaN;
                
            else
                
                data(i,12) = ...
                    distance_difference / time_difference;
                
            end
            
        end
        
    end
    
    waitbar(i / size(data,1),handle)
    
    if i == size(data,1)
        
        close(handle)
        
    end
    
end

save('data_with_noise.mat','data');

%%

clear all;
close all;
clc;

load('data');

%   Manually removing noisy trips:

data(data(:,4) == 1428788,:) = [];
data(data(:,4) == 1428789,:) = [];
data(data(:,4) == 1428790,:) = [];
data(data(:,4) == 1428791,:) = [];
data(data(:,4) == 1428806,:) = [];

%   Computing speeds with noisy trips excluded:

data = horzcat(data,zeros(size(data,1),1));

handle = waitbar(0,'Please wait ... ');

for i = 2:size(data,1)
    
    if data(i,1) == data(i - 1,1)
        
        distance_difference = ...
            lldistkm([data(i,9) data(i,10)], ...
            [data(i - 1,9) data(i - 1,10)]);
        
        if distance_difference == 0
            
            data(i,12) = 0;
            
        else
            
            time_difference = ...
                (data(i,5) - data(i - 1,6)) * 24;
            
            if time_difference == 0
                
                data(i,12) = NaN;
                
            else
                
                data(i,12) = ...
                    distance_difference / time_difference;
                
            end
            
        end
        
    end
    
    waitbar(i / size(data,1),handle)
    
    if i == size(data,1)
        
        close(handle)
        
    end
    
end

save('data_without_noise.mat','data');

%%

clear all;
close all;
clc;

%   Comparing speeds before and after removing noisy trips:

%   Before removing noisy trips:

load('data_with_noise');

trip_ids = unique(data(:,4),'rows','stable');
trip_average_speeds = zeros(size(trip_ids,1),1);

for i = 1:size(trip_ids,1)
    
    trip_speed_data = data(data(:,4) == trip_ids(i),12);
    trip_speed_data = ...
        trip_speed_data(isnan(trip_speed_data) == 0);
    
    trip_average_speeds(i) = mean(trip_speed_data);
    
end

figure(1);

subplot(2,1,1);
title('Speeds before removing noisy trips','FontSize',18);
hold on
xlabel('Trip ID','FontSize',18);
hold on
label = sprintf('Average trip\nspeed (km/h)');
ylabel(label,'FontSize',18);
hold on
scatter(trip_ids,trip_average_speeds,'r');
ylim([0 200]);
hold on
grid on

%   After removing noisy trips:

load('data_without_noise');

trip_ids = unique(data(:,4),'rows','stable');
trip_average_speeds = zeros(size(trip_ids,1),1);

for i = 1:size(trip_ids,1)
    
    trip_speed_data = data(data(:,4) == trip_ids(i),12);
    trip_speed_data = ...
        trip_speed_data(isnan(trip_speed_data) == 0);
    
    trip_average_speeds(i) = mean(trip_speed_data);
    
end

subplot(2,1,2);
title('Speeds after removing noisy trips','FontSize',18);
hold on
xlabel('Trip ID','FontSize',18);
hold on
label = sprintf('Average trip\nspeed (km/h)');
ylabel(label,'FontSize',18);
hold on
scatter(trip_ids,trip_average_speeds,'g');
ylim([0 200]);
hold on
grid on

saveas(figure(1),'before and after removing noisy trips.fig');
saveas(figure(1),'before and after removing noisy trips.bmp');

%%

clear all;
close all;
clc;

load('data_without_noise');

%   Cleaning data:

%   Replacing speed errors:

block_ids = unique(data(:,1),'stable');
percentage_of_block_speed_NaNs = zeros(size(block_ids,1),1);

for i = 1:size(block_ids,1)
    
    percentage_of_block_speed_NaNs(i) = ...
        sum(isnan(data(data(:,1) == block_ids(i),12))) / ...
        size(data(data(:,1) == block_ids(i),:),1) * 100;
    
end
    
figure(1);

subplot(2,1,1);
subplot_title = ...
    sprintf('Percentage of block speed NaN-values\n(Before replacing errors)');
title(subplot_title,'FontSize',18);
hold on
xlabel('Block ID','FontSize',18);
hold on
label = sprintf('Percentage of speed\nNaN-values');
ylabel(label,'FontSize',18);
hold on
scatter(block_ids,percentage_of_block_speed_NaNs,'r');
ylim([0 100]);
hold on
grid on

block_ids = unique(data(:,1),'rows','stable');

handle = waitbar(0,'Please wait ... ');

for i = 1:size(block_ids,1)
    
    block_data = data(data(:,1) == block_ids(i),:);
    
    while any(isnan(block_data(:,12)))
        
        for j = 2:size(block_data,1) - 1
            
            if (isnan(block_data(j,12)) == 1) && ...
                    (isnan(block_data(j + 1,12)) == 0)
                
                waiting_time = block_data(j,6) - block_data(j,5);
                
                block_data(j,5) = ...
                    block_data(j - 1,6) + ...
                    (block_data(j + 1,5) - block_data(j - 1,6)) * 1 / 2;
                
                block_data(j,6) = ...
                    min(block_data(j,5) + waiting_time, ...
                    block_data(j + 1,5));
                
                distance_difference = ...
                    lldistkm([block_data(j,9) block_data(j,10)], ...
                    [block_data(j - 1,9) block_data(j - 1,10)]);
                
                time_difference = ...
                    (block_data(j,5) - block_data(j - 1,6)) * 24;
                
                block_data(j,12) = ...
                    distance_difference / time_difference;
                
                distance_difference = ...
                    lldistkm([block_data(j,9) block_data(j,10)], ...
                    [block_data(j + 1,9) block_data(j + 1,10)]);
                
                time_difference = ...
                    (block_data(j + 1,5) - block_data(j,6)) * 24;
                
                if time_difference == 0
                    
                    block_data(j + 1,12) = NaN;
                    
                else
                    
                    block_data(j + 1,12) = ...
                        distance_difference / time_difference;
                    
                end
                
            end
            
        end
        
        if isnan(block_data(end,12)) == 1
            
            block_data(end,12) = block_data(end - 1,12);
            
            distance_difference = ...
                lldistkm([block_data(end,9) block_data(end,10)], ...
                [block_data(end - 1,9) block_data(end - 1,10)]);
            
            time_difference = ...
                distance_difference / block_data(end,12) / 24;
            
            waiting_time = block_data(end,6) - block_data(end,5);
            
            block_data(end,5) = ...
                block_data(end - 1,6) + time_difference;
            
            block_data(end,6) = block_data(end,5) + waiting_time;
            
        end
        
    end
    
    data(data(:,1) == block_ids(i),:) = block_data;

    waitbar(i / size(block_ids,1),handle)
    
    if i == size(block_ids,1)
        
        close(handle)
        
    end
    
end

high_speed_measurement_indices = ...
    find(data(:,12) > 115);

for i = 1:length(high_speed_measurement_indices)
    
    j = high_speed_measurement_indices(i);
    
    data(j,12) = 115;
    
    distance_difference = ...
        lldistkm([data(j,9) data(j,10)], ...
        [data(j - 1,9) data(j - 1,10)]);
    
    time_difference = ...
        distance_difference / data(j,12) / 24;
    
    waiting_time = data(j,6) - data(j,5);
    
    data(j,5) = ...
        data(j - 1,6) + time_difference;
    
    data(j,6) = ...
        min(data(j,5) + waiting_time,data(j + 1,5));
    
    distance_difference = ...
        lldistkm([data(j,9) data(j,10)], ...
        [data(j + 1,9) data(j + 1,10)]);
    
    time_difference = ...
        (data(j + 1,5) - data(j,6)) * 24;
    
    data(j + 1,12) = ...
        distance_difference / time_difference;
        
end

percentage_of_block_speed_NaNs = zeros(size(block_ids,1),1);

for i = 1:size(block_ids,1)
    
    percentage_of_block_speed_NaNs(i) = ...
        sum(isnan(data(data(:,1) == block_ids(i),12))) / ...
        size(data(data(:,1) == block_ids(i),:),1) * 100;
    
end

subplot(2,1,2);
subplot_title = ...
    sprintf('Percentage of block speed NaN-values\n(After replacing errors)');
title(subplot_title,'FontSize',18);
hold on
xlabel('Block ID','FontSize',18);
hold on
label = sprintf('Percentage of speed\nNaN-values');
ylabel(label,'FontSize',18);
hold on
scatter(block_ids,percentage_of_block_speed_NaNs,'g');
ylim([0 100]);
hold on
grid on  

saveas(figure(1),'before and after replacing speed errors.fig');
saveas(figure(1),'before and after replacing speed errors.bmp');

%   Visualizing speed distribution:

figure(2);
ahistnorm(data(data(:,12) > 5,12),100);
%Only faster-than-walking-speed values are included
hold on
title('Bus speed distribution','FontSize',18);
hold on
xlabel('Speed (km/h)','FontSize',18);
hold on
ylabel('Probability','FontSize',18);
xlim([min(data(:,12)) max(data(:,12))])
hold on
grid on

saveas(figure(2),'bus speed distribution.fig');
saveas(figure(2),'bus speed distribution.bmp');

cleaned_data = data;

save('cleaned_data.mat','cleaned_data');

%%

clear all;
close all;
clc;

load('cleaned_data');

%   Modifying data:

modified_data = ...
    horzcat(cleaned_data(:,1),cleaned_data(:,11), ...
    cleaned_data(:,4:6),cleaned_data(:,9:10));

handle = waitbar(0,'Please wait ... ');

for i = 1:125949
    
    if modified_data(i,4) ~= modified_data(i,5)
        
        extra_row = modified_data(i,:);
        extra_row(1,4) = modified_data(i,5);
        
        modified_data(i + 1:end + 1,:) = ...
            vertcat(extra_row,modified_data(i + 1:end,:)); 
        
    end
    
    waitbar(i / size(modified_data,1),handle)
    
    if i == size(modified_data,1)
        
        close(handle)
        
    end
    
end

modified_data(:,5) = [];

save('modified_data.mat','modified_data');

%%

clear all;
close all;
clc;

load('modified_data');
load('shapes');

%   Synthesizing data:

trip_ids = unique(modified_data(:,3),'rows','stable');
synthetic_trip_lats = zeros(size(trip_ids,1),27 * 60 * 6); 
synthetic_trip_lons = zeros(size(trip_ids,1),27 * 60 * 6);

handle = waitbar(0,'Please wait ... ');

for i = 1:size(trip_ids,1)
        
    trip_trajectory = ...
        modified_data(modified_data(:,3) == trip_ids(i),:);
    
    lat_model = ...
        fit(trip_trajectory(:,4),trip_trajectory(:,5), ...
        'linearinterp');
    
    lon_model = ...
        fit(trip_trajectory(:,4),trip_trajectory(:,6), ...
        'linearinterp');
    
    for t = 1:27 * 60 * 6
        
        if (t / (24 * 60 * 6) >= min(trip_trajectory(:,4))) && ...
                (t / (24 * 60 * 6) <= max(trip_trajectory(:,4)))
          
            model_result = ...
                horzcat(lat_model(t / (24 * 60 * 6)),...
                lon_model(t / (24 * 60 * 6)));
            
            %   Map-matching:
            
            shape_trajectory = ...
                shapes(shapes(:,1) == trip_trajectory(1,2),1:3);
            
            distances = zeros(size(shape_trajectory,1),1);
            
            for j = 1:size(shape_trajectory,1)
                
                distances(j) = ...
                    lldistkm([model_result(1) model_result(2)], ...
                    [shape_trajectory(j,2) shape_trajectory(j,3)]);
                
            end
            
            closest_point = find(distances == min(distances));
            
            synthetic_trip_lats(i,t) = ...
                shape_trajectory(closest_point(1),2);
            
            synthetic_trip_lons(i,t) = ...
                shape_trajectory(closest_point(1),3);
            
        end
        
    end
    
    waitbar(i / size(trip_ids,1),handle)
    
    if i == size(trip_ids,1)
        
        close(handle)
        
    end
    
end

block_ids = unique(modified_data(:,1),'rows','stable');
synthetic_block_lats = zeros(size(block_ids,1),27 * 60 * 6); 
synthetic_block_lons = zeros(size(block_ids,1),27 * 60 * 6);

handle = waitbar(0,'Please wait ... ');

for i = 1:size(block_ids,1)
    
    block_trips = ...
        unique(modified_data(modified_data(:,1) == block_ids(i),3),'rows','stable');
    
    for t = 1:27 * 60 * 6
        
        synthetic_block_lats(i,t) = ...
            sum(synthetic_trip_lats( ...
            find(trip_ids == block_trips(1)): ...
            find(trip_ids == block_trips(end)),t));
        
        synthetic_block_lons(i,t) = ...
            sum(synthetic_trip_lons( ...
            find(trip_ids == block_trips(1)): ...
            find(trip_ids == block_trips(end)),t));
        
    end
    
    waitbar(i / size(block_ids,1),handle)
    
    if i == size(block_ids,1)
        
        close(handle)
        
    end
    
end

%   Filling synthetic block data gaps between same-block trips:

%   Notice that most of these gaps are for buses waiting between the different trips

handle = waitbar(0,'Please wait ... ');

for i = 1:size(block_ids,1)
   
    block_modified_data = ...
        modified_data(modified_data(:,1) == block_ids(i),:);
    
    block_start_time = block_modified_data(1,4);
    block_end_time = block_modified_data(end,4);
    
    for t = 1:27 * 60 * 6
       
        if (t / (24 * 60 * 6) >= block_start_time) && ...
                (t / (24 * 60 * 6) <= block_end_time)
        
            if (synthetic_block_lats(i,t) == 0)
            
                gap_size = 0;
                
                while synthetic_block_lats(i,t + gap_size) == 0
                    
                    gap_size = gap_size + 1;
                    
                end
                
                gap_lat_trajectory = ...
                    vertcat([(t - 1) / (24 * 60 * 6), ...
                    synthetic_block_lats(i,t - 1)], ...
                    [(t + gap_size) / (24 * 60 * 6),...
                    synthetic_block_lats(i,t + gap_size)]); 
                
                gap_lat_model = ...
                    fit(gap_lat_trajectory(:,1),gap_lat_trajectory(:,2), ...
                    'linearinterp');
                
                synthetic_block_lats(i,t:t + gap_size) = ...
                    gap_lat_model((t:t + gap_size) ./ (24 * 60 * 6));
            
            end
            
            if (synthetic_block_lons(i,t) == 0)
                
                gap_size = 0;
                
                while synthetic_block_lons(i,t + gap_size) == 0
                    
                    gap_size = gap_size + 1;
                    
                end
                
                gap_lon_trajectory = ...
                    vertcat([(t - 1) / (24 * 60 * 6), ...
                    synthetic_block_lons(i,t - 1)], ...
                    [(t + gap_size) / (24 * 60 * 6),...
                    synthetic_block_lons(i,t + gap_size)]); 
                
                gap_lon_model = ...
                    fit(gap_lon_trajectory(:,1),gap_lon_trajectory(:,2), ...
                    'linearinterp');
                
                synthetic_block_lons(i,t:t + gap_size) = ...
                    gap_lon_model((t:t + gap_size) ./ (24 * 60 * 6));
                
            end
                
        end
            
    end
    
    waitbar(i / size(block_ids,1),handle)
    
    if i == size(block_ids,1)
        
        close(handle)
        
    end
    
end

%   Correcting same-block trip single-step overlaps:

%   Notice that these overlaps exist because of the same single-step shared 
%   between previous-trip and next-trip where both arrival times are 
%   equal as well as their stop coordinates.

max_synthetic_trip_lat = max(max(synthetic_trip_lats));
min_synthetic_trip_lon = min(min(synthetic_trip_lons));

handle = waitbar(0,'Please wait ... ');

for i = 1:size(block_ids,1)
    
    block_modified_data = ...
        modified_data(modified_data(:,1) == block_ids(i),:);
    
    block_start_time = block_modified_data(1,4);
    block_end_time = block_modified_data(end,4);
    
    for t = 1:27 * 60 * 6
        
        if (t / (24 * 60 * 6) >= block_start_time) && ...
                (t / (24 * 60 * 6) <= block_end_time)
            
            if (synthetic_block_lats(i,t) > max_synthetic_trip_lat) || ...
                    (synthetic_block_lons(i,t) < min_synthetic_trip_lon)
                
                if ((synthetic_block_lats(i,t - 1) < ...
                        max_synthetic_trip_lat) && ...
                        (synthetic_block_lats(i,t + 1) < ...
                        max_synthetic_trip_lat)) || ...
                        ((synthetic_block_lons(i,t - 1) > ...
                        min_synthetic_trip_lon) && ...
                        (synthetic_block_lons(i,t + 1) > ...
                        min_synthetic_trip_lon))
                    
                    synthetic_block_lats(i,t) = synthetic_block_lats(i,t) / 2;
                    synthetic_block_lons(i,t) = synthetic_block_lons(i,t) / 2;
                    
                end
                
            end
            
        end
        
    end
    
    waitbar(i / size(block_ids,1),handle)
    
    if i == size(block_ids,1)
        
        close(handle)
        
    end
    
end

save('synthetic_trip_lats.mat','synthetic_trip_lats');
save('synthetic_trip_lons.mat','synthetic_trip_lons');
save('synthetic_block_lats.mat','synthetic_block_lats');
save('synthetic_block_lons.mat','synthetic_block_lons');

%%

clear all;
close all;
clc;

load('synthetic_block_lats');
load('synthetic_block_lons');
load('modified_data');
load('shapes');
load('synthetic_trip_lats');
load('synthetic_trip_lons');

%   Visualizing synthetic data:

synthetic_data_animation = ...
    VideoWriter('synthetic_data_animation.avi');
synthetic_data_animation.FrameRate = 10;
synthetic_data_animation.Quality = 75;
open(synthetic_data_animation);

for t = 5 * 60 * 6:27 * 60 * 6

    figure(1)
    
    clf;

    title('Buses mobility','FontSize',18);
    hold on
    
    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
    
    plot(synthetic_block_lons(:,t),synthetic_block_lats(:,t), ...
        'ko','MarkerFaceColor','y');
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
    
    grid on
    
    drawnow;
    
    writeVideo(synthetic_data_animation, ...
        getframe(figure(1)));

end

close(synthetic_data_animation);

number_of_blocks_vs_time = [];

for t = 1:27 * 60 * 6
    
    number_of_blocks_vs_time = ...
        horzcat(number_of_blocks_vs_time, ...
        length(find(synthetic_block_lats(:,t) ~= 0))); %#ok<AGROW>

end

number_of_blocks_per_hour = zeros(1,27);

for h = 1:27

    number_of_blocks_per_hour(h) = ...
        mean(number_of_blocks_vs_time((h - 1) * 60 * 6 + 1:h * 60 * 6));  
    
end

figure(2);

title('Number of buses vs. time','FontSize',18);
hold on
bar(1:27,number_of_blocks_per_hour);
xlim([4 27.5]);
hold on
ylabel('Number of buses','FontSize',18);
xlabel('Time (hours)','FontSize',18);
hold on
grid on

saveas(figure(2),'number of buses vs. time.fig');
saveas(figure(2),'number of buses vs. time.bmp');

figure(3);

title('Buses @ 6:00 AM','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_block_lons(:,6 * 60 * 6), ...
    synthetic_block_lats(:,6 * 60 * 6), ...
    'ko','MarkerFaceColor','y', ...
    'MarkerSize',12,'LineWidth',3);
axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
grid on

saveas(figure(3),'buses at 6 am.fig');
saveas(figure(3),'buses at 6 am.bmp');

figure(4);

title('Buses @ 5:00 PM','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_block_lons(:,17 * 60 * 6), ...
    synthetic_block_lats(:,17 * 60 * 6), ...
    'ko','MarkerFaceColor','y', ...
    'MarkerSize',12,'LineWidth',3);
axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
grid on

saveas(figure(4),'buses at 5 pm.fig');
saveas(figure(4),'buses at 5 pm.bmp');

trip_ids = unique(modified_data(:,3),'stable');

trip_index = 1;%    Chosen at random

trip_modified_data = ...
    modified_data(modified_data(:,3) == trip_ids(trip_index),:);

realistic_trajectory = ...
    shapes(shapes(:,1) == trip_modified_data(1,2),2:3);

synthetic_trajectory = ...
    horzcat(synthetic_trip_lats(trip_index,:)', ...
    synthetic_trip_lons(trip_index,:)');

synthetic_trajectory = ...
    synthetic_trajectory(synthetic_trajectory(:,1) ~= 0,:);

figure(5);

title(sprintf('Synthetic vs. realistic\ntrip trajectory (Trip # %0.0f)', ...
    trip_ids(trip_index)),'FontSize',18);
hold on

map = imread('trajectories_map.jpg');
image('CData',map, ...
    'XData', ...
    [min(realistic_trajectory(:,2)) max(realistic_trajectory(:,2))], ...
    'YData', ...
    [min(realistic_trajectory(:,1)) max(realistic_trajectory(:,1))])
hold on

plot(realistic_trajectory(:,2),realistic_trajectory(:,1),'b','LineWidth',4);
hold on;
plot(synthetic_trajectory(:,2),synthetic_trajectory(:,1), ...
    'ko','MarkerFaceColor','y','MarkerSize',8);
hold on;

axis([min(realistic_trajectory(:,2)) max(realistic_trajectory(:,2)) ...
    min(realistic_trajectory(:,1)) max(realistic_trajectory(:,1))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on;
legend({'Realistic trajectory','Synthetic trajectory'}, ...
    'FontSize',18,'Location','best');
hold on
grid on

saveas(figure(5),'synthetic vs. realistic trip trajectories.fig');
saveas(figure(5),'synthetic vs. realistic trip trajectories.bmp');

%%

clear all;
close all;
clc;

load('cleaned_data');

stops_data = ...
    unique(cleaned_data(:,[7,9,10]),'rows','stable');

%   Converting stop-coordinates data:
    
stops_data(:,4) = ...
    6371 .* cos(stops_data(:,2) .* pi / 180) .* ...
    cos(stops_data(:,3) .* pi / 180);

stops_data(:,5) = ...
    6371 .* cos(stops_data(:,2) .* pi / 180) .* ...
    sin(stops_data(:,3) .* pi / 180);

stops_data(:,6) = ...
    6371 .* sin(stops_data(:,2) .* pi / 180);

%   Refining stop ids:

neighborhood_distance = 300 / 1000;

stop_clusters = clusterdata(stops_data(:,4:6), ...
    'linkage','complete','criterion','distance', ...
    'cutoff',neighborhood_distance);

number_of_clusters = size(unique(stop_clusters),1);

refined_stop_ids = zeros(number_of_clusters,1);

for i = 1:number_of_clusters
    
    cluster_members = ...
        stops_data(stop_clusters == i,1);
    
    cluster_size = ...
        size(cluster_members,1);
    
    cluster_member_coordinates = ...
        stops_data(stop_clusters == i,2:3);
    
    cluster_member_coordinates_mean = ...
        mean(cluster_member_coordinates,1);
    
    cluster_member_mean_inter_distances = ...
        zeros(cluster_size,1);
    
    for j = 1:cluster_size
        
        cluster_member_mean_inter_distances(j) = ...
            lldistkm(cluster_member_coordinates(j,:), ...
            cluster_member_coordinates_mean);
        
    end
    
    cluster_medoid = ...
        cluster_members( ...
        cluster_member_mean_inter_distances == ...
        min(cluster_member_mean_inter_distances));
    
    refined_stop_ids(i) = cluster_medoid;
        
end

save('refined_stop_ids.mat','refined_stop_ids');

%%

clear all;
close all;
clc;

load('refined_stop_ids');
load('synthetic_block_lats');
load('synthetic_block_lons');
load('stops');

%   Choosing optimal stop ids:

stop_popularities = zeros(size(refined_stop_ids,1),1);
broadcasting_range = 300;
maximum_number_of_stops = 500 - size(synthetic_block_lats,1);

handle = waitbar(0,'Please wait ... ');

for i = 1:size(refined_stop_ids,1)
    
    for t = 1:size(synthetic_block_lats,2)
    
        if any(synthetic_block_lats(:,t) ~= 0)
            
            block_coordinates = ...
                [nonzeros(synthetic_block_lats(:,t)), ...
                nonzeros(synthetic_block_lons(:,t))];
        
            stop_coordinates = ...
                stops(stops(:,1) == refined_stop_ids(i),5:6);
            
            for j = 1:size(block_coordinates,1)
                
                if lldistkm(stop_coordinates, ...
                        block_coordinates(j,:)) ...
                        <= broadcasting_range / 1000
                    
                    stop_popularities(i) = ...
                        stop_popularities(i) + 1;
                    
                end
                
            end
            
        end
        
    end
    
    waitbar(i / size(refined_stop_ids,1),handle)
    
    if i == size(refined_stop_ids,1)
        
        close(handle)
        
    end

end

figure(1);
title('Refined stops popularities','FontSize',18);
hold on
scatter(1:size(refined_stop_ids,1),stop_popularities,'r');
xlim([0 size(refined_stop_ids,1)]);
hold on
xlabel('Refined stop index','FontSize',18);
hold on
ylabel('Popularity','FontSize',18);
hold on
grid on

saveas(figure(1),'refined stops popularities.fig');
saveas(figure(1),'refined stops popularities.bmp');

[~,optimal_stop_ids] = sortrows(stop_popularities,-1);
optimal_stop_ids = refined_stop_ids(optimal_stop_ids);
optimal_stop_ids = ...
    optimal_stop_ids(1:maximum_number_of_stops);

save('stop_popularities.mat','stop_popularities');
save('optimal_stop_ids.mat','optimal_stop_ids');

%%

clear all;
close all;
clc;

load('synthetic_block_lats');
load('synthetic_block_lons');
load('optimal_stop_ids');
load('stops');

%   Adding optimal stops data to the synthetic block data:

synthetic_lats = ...
    zeros(size(synthetic_block_lats,1) + ...
    size(optimal_stop_ids,1), ...
    size(synthetic_block_lats,2));

synthetic_lats(1:size(synthetic_block_lats,1),:) = ...
    synthetic_block_lats;

synthetic_lons = ...
    zeros(size(synthetic_block_lons,1) + ...
    size(optimal_stop_ids,1), ...
    size(synthetic_block_lons,2));

synthetic_lons(1:size(synthetic_block_lons,1),:) = ...
    synthetic_block_lons;

handle = waitbar(0,'Please wait ... ');

for i = 1:size(optimal_stop_ids,1)
    
    synthetic_lats(size(synthetic_block_lats,1) + i,:) = ...
        repmat( ...
        stops(stops(:,1) == optimal_stop_ids(i),5), ...
        1,size(synthetic_block_lats,2));
    
    synthetic_lons(size(synthetic_block_lons,1) + i,:) = ...
        repmat( ...
        stops(stops(:,1) == optimal_stop_ids(i),6), ...
        1,size(synthetic_block_lons,2));
    
    waitbar(i / size(optimal_stop_ids,1),handle)
    
    if i == size(optimal_stop_ids,1)
        
        close(handle)
        
    end

end

save('synthetic_lats.mat','synthetic_lats');
save('synthetic_lons.mat','synthetic_lons');

%%

clear all;
close all;
clc;

load('modified_data');
load('stops');
load('refined_stop_ids');
load('optimal_stop_ids');

%   Visualizing all stops, refined stops and optimal stops:

figure(1);

title('All stops','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(stops(:,6),stops(:,5), ...
    'ko','MarkerFaceColor','y', ...
    'MarkerSize',12,'LineWidth',3);
hold on
axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
number_of_stops = ...
    sprintf('%04d stops',length(stops));
text(-80.41,43.575, ...
    number_of_stops,'Color','black','FontSize',20, ...
    'FontWeight','bold')
hold on
grid on

saveas(figure(1),'all stops.fig');
saveas(figure(1),'all stops.bmp');

figure(2);

title('Refined stops','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

for i = 1:length(refined_stop_ids)
    
    plot(stops(stops(:,1) == refined_stop_ids(i),6), ...
        stops(stops(:,1) == refined_stop_ids(i),5), ...
        'ko','MarkerFaceColor','y', ...
        'MarkerSize',12,'LineWidth',3);
    hold on
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
number_of_refined_stops = ...
    sprintf('%03d refined stops',length(refined_stop_ids));
text(-80.47,43.575, ...
    number_of_refined_stops,'Color','black','FontSize',20, ...
    'FontWeight','bold')
hold on
grid on

saveas(figure(2),'refined stops.fig');
saveas(figure(2),'refined stops.bmp');

figure(3);

title('Optimal stops','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

for i = 1:length(optimal_stop_ids)
    
    plot(stops(stops(:,1) == optimal_stop_ids(i),6), ...
        stops(stops(:,1) == optimal_stop_ids(i),5), ...
        'ko','MarkerFaceColor','y', ...
        'MarkerSize',12,'LineWidth',3);
    hold on
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
number_of_optimal_stops = ...
    sprintf('%03d optimal stops',length(optimal_stop_ids));
text(-80.47,43.575, ...
    number_of_optimal_stops,'Color','black','FontSize',20, ...
    'FontWeight','bold')
hold on
grid on

saveas(figure(3),'optimal stops.fig');
saveas(figure(3),'optimal stops.bmp');

%%

clear all;
close all;
clc;

load('synthetic_lats');
load('synthetic_lons');

broadcasting_range = 300;

for t = 1:size(synthetic_lats,2)
    
    connectivities{t} = ...
        zeros(size(synthetic_lats,1), ...
        size(synthetic_lats,1));
    
end

%   Determining inter-bus/stop connectivities:

handle = waitbar(0,'Please wait ... ');

for t = 1:size(synthetic_lats,2)
    
    for i = 1:size(synthetic_lats,1) - 1
        
        if synthetic_lats(i,t) ~= 0
            
            for j = i + 1:size(synthetic_lats,1)
                
                if synthetic_lats(j,t) ~= 0
                    
                    distance = ...
                        lldistkm( ...
                        [synthetic_lats(i,t) ...
                        synthetic_lons(i,t)], ...
                        [synthetic_lats(j,t) ...
                        synthetic_lons(j,t)]) ...
                        * 1000;

                    if distance <= broadcasting_range
                        
                        connectivities{t}(i,j) = uint8(1);
                        %   "uint8" is used to minimize memory consumption
                        %   and increase execution speed
                        
                        connectivities{t}(j,i) = ...
                            connectivities{t}(i,j);
                        
                    end

                end
            
            end
            
        end
        
    end
    
    waitbar(t / size(synthetic_lats,2),handle)
    
    if t == size(synthetic_lats,2)
        
        close(handle)
        
    end
    
end

save('connectivities.mat','connectivities','-v7.3');

%%

clear all;
close all;
clc;

load('modified_data');
load('synthetic_lats');
load('synthetic_lons');
load('synthetic_block_lats');
load('connectivities');

%   Visualizing connectivities:

%   Drawing connectivities:

figure(1);

title('Connectivities @ 5:00 PM','FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_lons(size(synthetic_block_lats,1) + 1:end,17 * 60 * 6), ...
    synthetic_lats(size(synthetic_block_lats,1) + 1:end,17 * 60 * 6), ...
    'ko','MarkerFaceColor','b');
hold on

plot(synthetic_lons(1:size(synthetic_block_lats,1),17 * 60 * 6), ...
    synthetic_lats(1:size(synthetic_block_lats,1),17 * 60 * 6), ...
    'ko','MarkerFaceColor','g');
hold on

legend({'Bus-stops','Buses'}, ...
    'Location','northwest','FontSize',18);
hold on

for i = 1:size(synthetic_lats,1)
    
    for j = 1:size(synthetic_lats,1)
        
        if connectivities{17 * 60 * 6}(i,j) == 1
            
            line( ...
                [synthetic_lons(i,17 * 60 * 6) ...
                synthetic_lons(j,17 * 60 * 6)], ...
                [synthetic_lats(i,17 * 60 * 6) ...
                synthetic_lats(j,17 * 60 * 6)], ...
                'Color','red','LineWidth',2);
            hold on
            
        end
        
    end
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
time = sprintf('17:00');
text(-80.35,43.575, ...
    time,'Color','black','FontSize',20,'FontWeight','bold')
hold on
grid on

saveas(figure(1),'connectivities at 5 pm.fig');
saveas(figure(1),'connectivities at 5 pm.bmp');

%   Animating connectivities:

period_start_time = 16.5 * 60;
period_end_time = 17.5 * 60;

connectivities_animation = ...
    VideoWriter('connectivities_animation.avi');
connectivities_animation.FrameRate = 10;
connectivities_animation.Quality = 75;
open(connectivities_animation);

for t = period_start_time * 6:period_end_time * 6

    figure(2)
    
    clf;

    title('Connectivities','FontSize',18);
    hold on
    
    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
        
    plot(synthetic_lons(size(synthetic_block_lats,1) + 1:end,t), ...
        synthetic_lats(size(synthetic_block_lats,1) + 1:end,t), ...
        'ko','MarkerFaceColor','b');
    hold on
    
    plot(synthetic_lons(1:size(synthetic_block_lats,1),t), ...
        synthetic_lats(1:size(synthetic_block_lats,1),t), ...
        'ko','MarkerFaceColor','g');
    hold on
    
    legend({'Bus-stops','Buses'}, ...
        'Location','northwest','FontSize',18);
    hold on

    for i = 1:size(synthetic_lats,1)
        
        for j = 1:size(synthetic_lats,1)
            
            if connectivities{t}(i,j) == 1
                
                line( ...
                    [synthetic_lons(i,t) synthetic_lons(j,t)], ...
                    [synthetic_lats(i,t) synthetic_lats(j,t)], ...
                    'Color','red','LineWidth',2);
                hold on
                
            end
            
        end
        
    end
    
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
        
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
        
    grid on
    
    drawnow;

    writeVideo(connectivities_animation, ...
        getframe(figure(2)));
    
end

close(connectivities_animation);

%   Visualizing zoomed-in connectivities:

%   Drawing zoomed-in connectivities:

figure(3);

title('Zoomed-in connectivities @ 5:00 PM','FontSize',18);
hold on

map = imread('zoomed_in_map.jpg');
image('CData',map, ...
    'XData',[-80.52 -80.48], ...
    'YData',[43.43 43.47])
hold on

plot(synthetic_lons(size(synthetic_block_lats,1) + 1:end,17 * 60 * 6), ...
    synthetic_lats(size(synthetic_block_lats,1) + 1:end,17 * 60 * 6), ...
    'ko','MarkerFaceColor','b');
hold on

plot(synthetic_lons(1:size(synthetic_block_lats,1),17 * 60 * 6), ...
    synthetic_lats(1:size(synthetic_block_lats,1),17 * 60 * 6), ...
    'ko','MarkerFaceColor','g');
hold on

legend({'Bus-stops','Buses'}, ...
    'Location','northwest','FontSize',18);
hold on

for i = 1:size(synthetic_lats,1)
    
    for j = 1:size(synthetic_lats,1)
        
        if connectivities{17 * 60 * 6}(i,j) == 1
            
            line( ...
                [synthetic_lons(i,17 * 60 * 6) ...
                synthetic_lons(j,17 * 60 * 6)], ...
                [synthetic_lats(i,17 * 60 * 6) ...
                synthetic_lats(j,17 * 60 * 6)], ...
                'Color','red','LineWidth',2);
            hold on
            
        end
        
    end
    
end

axis([-80.52 -80.48 43.43 43.47]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
time = sprintf('17:00');
text(-80.4875,43.465, ...
    time,'Color','black','FontSize',20,'FontWeight','bold')
hold on
grid on

saveas(figure(3),'zoomed-in connectivities at 5 pm.fig');
saveas(figure(3),'zoomed-in connectivities at 5 pm.bmp');

%   Animating zoomed-in connectivities:

period_start_time = 16.5 * 60;
period_end_time = 17.5 * 60;

zoomed_in_connectivities_animation = ...
    VideoWriter('zoomed_in_connectivities_animation.avi');
zoomed_in_connectivities_animation.FrameRate = 10;
zoomed_in_connectivities_animation.Quality = 75;
open(zoomed_in_connectivities_animation);

for t = period_start_time * 6:period_end_time * 6

    figure(4)
    
    clf;

    title('Zoomed-in connectivities','FontSize',18);
    hold on
    
    map = imread('zoomed_in_map.jpg');
    image('CData',map, ...
        'XData',[-80.52 -80.48], ...
        'YData',[43.43 43.47])
    hold on
        
    plot(synthetic_lons(size(synthetic_block_lats,1) + 1:end,t), ...
        synthetic_lats(size(synthetic_block_lats,1) + 1:end,t), ...
        'ko','MarkerFaceColor','b');
    hold on
    
    plot(synthetic_lons(1:size(synthetic_block_lats,1),t), ...
        synthetic_lats(1:size(synthetic_block_lats,1),t), ...
        'ko','MarkerFaceColor','g');
    hold on
    
    legend({'Bus-stops','Buses'}, ...
        'Location','northwest','FontSize',18);
    hold on

    for i = 1:size(synthetic_lats,1)
        
        for j = 1:size(synthetic_lats,1)
            
            if connectivities{t}(i,j) == 1
                
                line( ...
                    [synthetic_lons(i,t) synthetic_lons(j,t)], ...
                    [synthetic_lats(i,t) synthetic_lats(j,t)], ...
                    'Color','red','LineWidth',2);
                hold on
                
            end
            
        end
        
    end
    
    axis([-80.52 -80.48 43.43 43.47]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.4875,43.465, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
        
    grid on
    
    drawnow;

    writeVideo(zoomed_in_connectivities_animation, ...
        getframe(figure(4)));
    
end

close(zoomed_in_connectivities_animation);

%%

clear all;
close all;
clc;

load('connectivities');

%   Computing continuous contact durations:

period_start_times = ...
    [5,7,9,11,13,15,17,19,21,23,25] .* 60;
period_end_times = ...
    [7,9,11,13,15,17,19,21,23,25,27] .* 60;

handle = waitbar(0,'Please wait ... ');

for p = 1:size(period_start_times,2)
    
    contact_durations{p} = [];

    for i = 1:size(connectivities{1},1)
        
        contact_duration_counters = ...
            zeros(1,size(connectivities{1},1));
        
        for t = period_start_times(p) * 6:period_end_times(p) * 6
            
            for j = 1:size(connectivities{t}(i,:),2)
                
                if connectivities{t}(i,j) == 1
                    
                    contact_duration_counters(j) = ...
                        contact_duration_counters(j) + 1;
                    
                else
                    
                    if (t ~= period_start_times(p) * 6) && ...
                            (connectivities{t - 1}(i,j) == 1)
                        
                        contact_durations{p} = ...
                            vertcat(contact_durations{p}, ...
                            contact_duration_counters(j) * 10 / 60);
                        contact_duration_counters(j) = 0;
                        
                    end
                    
                end
                
            end
            
        end
        
    end
    
    waitbar(p / size(period_start_times,2),handle)
    
    if p == size(period_start_times,2)
        
        close(handle)
        
    end
    
end

save('continuous_contact_durations.mat','contact_durations');

all_contact_durations = [];

for p = 1:size(period_start_times,2)
    
    all_contact_durations = ...
        vertcat(all_contact_durations, ...
        contact_durations{p}); %#ok<AGROW>
    
end

figure(1);

figure_title = ...
    sprintf('Daily continuous contact\nduration distribution');
title(figure_title,'FontSize',18);
hold on
xlim([0 20]);
%   You did include up to 20 minutes because after that
%   the probabilities are very small to see
xlabel('Contact duration (min)','FontSize',18);
hold on
ylabel('Probability','FontSize',18);
hold on
ahistnorm(all_contact_durations, ...
    max(all_contact_durations));
hold on
grid on

saveas(figure(1),'daily continuous contact duration distribution.fig');
saveas(figure(1),'daily continuous contact duration distribution.bmp');

contact_durations_in_3d = [];

for p = 1:size(period_start_times,2)
    
    counts = ...
        ahistnorm(contact_durations{p}, ...
        max(all_contact_durations));
    
    contact_durations_in_3d = ...
        vertcat(contact_durations_in_3d,counts); %#ok<AGROW>
    
end

figure(2);

figure_title = ...
    sprintf('Periodic continuous contact duration\ndistribution vs. day period');
title(figure_title,'FontSize',18);
hold on
xlabel('Day period','FontSize',18);
hold on
ylabel('Contact duration (min)','FontSize',18);
hold on
zlabel('Probability','FontSize',18);
hold on

handle = bar3(contact_durations_in_3d');

for i = 1:numel(handle) 
    
    ydata = get(handle(i),'ydata');
    set(handle(i),'ydata',ydata);    
    cdata = get(handle(i),'zdata');
    set(handle(i),'cdata',cdata,'facecolor','interp')  
    
end

xlim([1 size(period_start_times,2)]);
ylim([0 20]);
%   You did include up to 20 minutes because after that
%   the probabilities are very small to see
zlim([0 max(max(contact_durations_in_3d))]);
view(125,25)
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
grid on

saveas(figure(2),'periodic continuous contact duration distribution vs. day period.fig');
saveas(figure(2),'periodic continuous contact duration distribution vs. day period.bmp');

all_contact_duration_periods = [];

for p = 1:size(period_start_times,2)
    
    all_contact_duration_periods = ...
        vertcat(all_contact_duration_periods, ...
        p .* ones(size(contact_durations{p},1),1)); %#ok<AGROW>
    
end

figure(3);

figure_title = ...
    sprintf('Periodic continuous contact duration\nboxplot vs. day period');
title(figure_title,'FontSize',18);
hold on
xlabel('Day period','FontSize',18);
hold on
ylabel('Contact duration (min)','FontSize',18);
hold on
boxplot(all_contact_durations, ...
    all_contact_duration_periods);
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
grid on

saveas(figure(3),'periodic continuous contact duration boxplot vs. day period.fig');
saveas(figure(3),'periodic continuous contact duration boxplot vs. day period.bmp');

contact_duration_25th_percentiles = ...
    zeros(size(period_start_times,2),1);
contact_duration_50th_percentiles = ...
    zeros(size(period_start_times,2),1);
contact_duration_75th_percentiles = ...
    zeros(size(period_start_times,2),1);

for p = 1:size(period_start_times,2)
   
    contact_duration_25th_percentiles(p) = ...
        prctile(contact_durations{p},25);
    
    contact_duration_50th_percentiles(p) = ...
        prctile(contact_durations{p},50);
    
    contact_duration_75th_percentiles(p) = ...
        prctile(contact_durations{p},75);

end

figure(4);

figure_title = ...
    sprintf('Periodic continuous contact duration\npercentiles vs. day period');
title(figure_title,'FontSize',18);
hold on
xlim([0 size(period_start_times,2) + 1]);
xlabel('Day period','FontSize',18);
hold on
ylim([0 3]);
ylabel('Contact duration (min)','FontSize',18);
hold on

plot(1:size(period_start_times,2), ...
    contact_duration_25th_percentiles, ...
    '--b','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    contact_duration_50th_percentiles, ...
    '-b','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    contact_duration_75th_percentiles, ...
    '--b','LineWidth',2);
hold on

set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
grid on

saveas(figure(4),'periodic continuous contact duration percentiles vs. day period.fig');
saveas(figure(4),'periodic continuous contact duration percentiles vs. day period.bmp');

%%

%The extractClusters function:

function next_cluster_members = ...
    extractClusters(period_start_time, period_end_time, ...
    connectivities, time_delta, minimum_contact_duration, maximum_number_of_hops)
        
    for i = 1:size(connectivities{1},1)
           
        for t = period_start_time * 6:period_end_time * 6
                
            if t == period_start_time * 6
                    
                period_connectivities{i} = connectivities{t}(i,:); %#ok<*SAGROW>
                    
            else
                    
                period_connectivities{i} = ...
                    period_connectivities{i} + ...
                    connectivities{t}(i,:);
                    
            end
                
        end
            
        previous_cluster_members{i} = ...
            find(period_connectivities{i} .* time_delta >= ...
            minimum_contact_duration);
            
    end
        
    next_cluster_members = previous_cluster_members;
        
    for n = 1:maximum_number_of_hops / 2 - 1
            
        for i = 1:size(connectivities{1},1)
                
            for j = 1:size(previous_cluster_members{i},2);
                    
                next_cluster_members{i} = ...
                    union(next_cluster_members{i}, ...
                    previous_cluster_members{ ...
                    previous_cluster_members{i}(1,j)});
            end
            
        end
        
        previous_cluster_members = next_cluster_members;
    
    end
    
end
    
%%
clear all;
close all;
clc;

load('connectivities');

%   Clustering:

period_start_times = ...
    [5,7,9,11,13,15,17,19,21,23,25] .* 60;
period_end_times = ...
    [7,9,11,13,15,17,19,21,23,25,27] .* 60;
minimum_contact_durations = ...
    [5,15,30] .* 60;
time_delta = 10;
maximum_number_of_hops = 20;

number_of_clusters = ...
    zeros(size(period_start_times,2),size(minimum_contact_durations,2));
number_of_unclustered_nodes = ...
    zeros(size(period_start_times,2),size(minimum_contact_durations,2));

cluster_size_means = ...
    zeros(size(period_start_times,2),size(minimum_contact_durations,2));
cluster_size_minima = ...
    zeros(size(period_start_times,2),size(minimum_contact_durations,2));
cluster_size_maxima = ...
    zeros(size(period_start_times,2),size(minimum_contact_durations,2));

handle = waitbar(0,'Please wait ... ');

for p = 1:size(period_start_times,2)
    
    period_start_time = period_start_times(p);
    
    period_end_time = period_end_times(p);
    
    for d = 1:size(minimum_contact_durations,2)
        
        minimum_contact_duration = minimum_contact_durations(d);
        
        next_cluster_members = ...
            extractClusters(period_start_time, period_end_time, ...
            connectivities, time_delta, ...
            minimum_contact_duration, maximum_number_of_hops);
        
        [cluster_members,cluster_indices,~] = ...
            uniquecell(next_cluster_members);
        
        number_of_clusters(p,d) = size(cluster_indices,1) - 1;
        %   We substract 1 to exclude the empty cluster set
        
        cluster_sizes{p,d} = zeros(number_of_clusters(p,d),1);
        
        for i = 1:number_of_clusters(p,d)
            
            cluster_sizes{p,d}(i) = ...
                size(cluster_members{i + 1},2);
            
        end
        
        number_of_unclustered_nodes(p,d) = 0;
        
        for i = 1:size(next_cluster_members,2)
            
            if isempty(next_cluster_members{i})
                
                number_of_unclustered_nodes(p,d) = ...
                    number_of_unclustered_nodes(p,d) + 1;
                
            end
            
        end
        
    end
    
    waitbar(p / size(period_start_times,2),handle)
    
    if p == size(period_start_times,2)
        
        close(handle)
        
    end

end

for p = 1:size(period_start_times,2)
    
    for d = 1:size(minimum_contact_durations,2)
        
        cluster_size_means(p,d) = ...
            mean(cluster_sizes{p,d});
        cluster_size_minima(p,d) = ...
            min(cluster_sizes{p,d});
        cluster_size_maxima(p,d) = ...
            max(cluster_sizes{p,d});

    end
    
end 
    
figure(1);

title('Number of clusters vs. day period','FontSize',18);
hold on
xlim([0,size(period_start_times,2) + 1]);
xlabel('Day period','FontSize',18);
hold on
ylabel('Number of clusters','FontSize',18);
hold on
plot(1:size(period_start_times,2),number_of_clusters(:,1), ...
    '-k','LineWidth',2);
hold on
plot(1:size(period_start_times,2),number_of_clusters(:,2), ...
    '--b','LineWidth',2);
hold on
plot(1:size(period_start_times,2),number_of_clusters(:,3), ...
    '-*r','LineWidth',2);
hold on
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
legend({'5-min discontinuous duration','15-min discontinuous duration', ...
    '30-min discontinuous duration'}, ...
    'Location','Best','FontSize',14);
hold on
grid on

saveas(figure(1),'number of clusters vs. day period.fig');
saveas(figure(1),'number of clusters vs. day period.bmp');

figure(2);

title('Cluster size means vs. day period','FontSize',18);
hold on
xlim([0,size(period_start_times,2) + 1]);
xlabel('Day period','FontSize',18);
hold on
ylabel('Cluster size mean','FontSize',18);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_means(:,1),'-k','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_means(:,2),'--b','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_means(:,3),'-*r','LineWidth',2);
hold on
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
legend({'5-min discontinuous duration','15-min discontinuous duration', ...
    '30-min discontinuous duration'}, ...
    'Location','Best','FontSize',14);
hold on
grid on

saveas(figure(2),'cluster size means vs. day period.fig');
saveas(figure(2),'cluster size means vs. day period.bmp');

figure(3);

title('Cluster size maximum vs. day period','FontSize',18);
hold on
xlim([0,size(period_start_times,2) + 1]);
xlabel('Day period','FontSize',18);
hold on
ylabel('Cluster size maximum','FontSize',18);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_maxima(:,1),'-k','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_maxima(:,2),'--b','LineWidth',2);
hold on
plot(1:size(period_start_times,2), ...
    cluster_size_maxima(:,3),'-*r','LineWidth',2);
hold on
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
legend({'5-min discontinuous duration','15-min discontinuous duration', ...
    '30-min discontinuous duration'}, ...
    'Location','Best','FontSize',14);
hold on
grid on

saveas(figure(3),'cluster size maximum vs. day period.fig');
saveas(figure(3),'cluster size maximum vs. day period.bmp');

figure(4);

title('Number of unclustered nodes vs. day period','FontSize',18);
hold on
xlim([0,size(period_start_times,2) + 1]);
xlabel('Day period','FontSize',18);
hold on
ylabel('Number of unclustered nodes','FontSize',18);
hold on
plot(1:size(period_start_times,2),number_of_unclustered_nodes(:,1), ...
    '-k','LineWidth',2);
hold on
plot(1:size(period_start_times,2),number_of_unclustered_nodes(:,2), ...
    '--b','LineWidth',2);
hold on
plot(1:size(period_start_times,2),number_of_unclustered_nodes(:,3), ...
    '-*r','LineWidth',2);
hold on
set(gca,'xtick',1:size(period_start_times,2)); 
xticklabels = ({'5-7','7-9','9-11','11-13', ...
    '13-15','15-17','17-19','19-21','21-23', ...
    '23-25','25-27'});
set(gca,'xticklabel',xticklabels);
hold on
legend({'5-min discontinuous duration','15-min discontinuous duration', ...
    '30-min discontinuous duration'}, ...
    'Location','Best','FontSize',14);
hold on
grid on

saveas(figure(4),'number of unclustered nodes vs. day period.fig');
saveas(figure(4),'number of unclustered nodes vs. day period.bmp');

save('clustering_results.mat','number_of_clusters', ...
    'cluster_sizes','number_of_unclustered_nodes');

%%

clear all;
close all;
clc;

load('connectivities');
load('synthetic_lats');
load('synthetic_lons');
load('modified_data');

period_start_time = 16 * 60;
period_end_time = 18 * 60;
time_delta = 10;
maximum_number_of_hops = 20;

%   Visualizing clusters at 5-min discontinuous contact duration:

minimum_contact_duration = 5 * 60;

next_cluster_members = ...
    extractClusters(period_start_time, period_end_time, ...
    connectivities, time_delta, ...
    minimum_contact_duration, maximum_number_of_hops);

[cluster_members,cluster_indices,~] = ...
    uniquecell(next_cluster_members);

for i = 1:size(cluster_indices,1)
    
    cluster_colors(i,:) = randi(5,1,3) ./ 5;

end

figure(1)

figure_title = ...
    sprintf('Clusters @ 5-min discontinuous\ncontact duration');
title(figure_title,'FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
    'ko','MarkerFaceColor','y');
hold on

for i = 2:size(cluster_indices,1)
    
    plot(synthetic_lons( ...
        cluster_members{1,i},t), ...
        synthetic_lats( ...
        cluster_members{1,i},t), ...
        'ko','MarkerFaceColor',cluster_colors(i,:));
    hold on
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
grid on

saveas(figure(1),'clusters at 5.fig');
saveas(figure(1),'clusters at 5.bmp');

cluster_animation_at_5 = ...
    VideoWriter('cluster_animation_at_5.avi');
cluster_animation_at_5.FrameRate = 10;
cluster_animation_at_5.Quality = 75;
open(cluster_animation_at_5);

for t = period_start_time * 6:period_end_time * 6

    figure(2)
    
    clf;

    figure_title = ...
        sprintf('Clusters @ 5-min discontinuous\ncontact duration');
    title(figure_title,'FontSize',18);
    hold on
    
    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
    
    plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
        'ko','MarkerFaceColor','y');
    hold on
    
    for i = 2:size(cluster_indices,1)
        
        plot(synthetic_lons( ...
            cluster_members{1,i},t), ...
            synthetic_lats( ...
            cluster_members{1,i},t), ...
            'ko','MarkerFaceColor',cluster_colors(i,:));
        hold on
        
    end
    
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
    
    grid on
    
    drawnow;

    writeVideo(cluster_animation_at_5, ...
        getframe(figure(2)));
    
end

close(cluster_animation_at_5);

%   Visualizing clusters at 15-min discontinuous contact duration:

minimum_contact_duration = 15 * 60;

next_cluster_members = ...
    extractClusters(period_start_time, period_end_time, ...
    connectivities, time_delta, ...
    minimum_contact_duration, maximum_number_of_hops);

[cluster_members,cluster_indices,~] = ...
    uniquecell(next_cluster_members);

for i = 1:size(cluster_indices,1)
    
    cluster_colors(i,:) = randi(5,1,3) ./ 5;

end

figure(3)

figure_title = ...
    sprintf('Clusters @ 15-min discontinuous\ncontact duration');
title(figure_title,'FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
    'ko','MarkerFaceColor','y');
hold on

for i = 2:size(cluster_indices,1)
    
    plot(synthetic_lons( ...
        cluster_members{1,i},t), ...
        synthetic_lats( ...
        cluster_members{1,i},t), ...
        'ko','MarkerFaceColor',cluster_colors(i,:));
    hold on
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
grid on

saveas(figure(3),'clusters at 15.fig');
saveas(figure(3),'clusters at 15.bmp');

cluster_animation_at_15 = ...
    VideoWriter('cluster_animation_at_15.avi');
cluster_animation_at_15.FrameRate = 10;
cluster_animation_at_15.Quality = 75;
open(cluster_animation_at_15);

for t = period_start_time * 6:period_end_time * 6

    figure(4)
    
    clf;

    figure_title = ...
        sprintf('Clusters @ 15-min discontinuous\ncontact duration');
    title(figure_title,'FontSize',18);
    hold on
    
    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
    
    plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
        'ko','MarkerFaceColor','y');
    hold on
    
    for i = 2:size(cluster_indices,1)
        
        plot(synthetic_lons( ...
            cluster_members{1,i},t), ...
            synthetic_lats( ...
            cluster_members{1,i},t), ...
            'ko','MarkerFaceColor',cluster_colors(i,:));
        hold on
        
    end
    
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
    
    grid on
    
    drawnow;

    writeVideo(cluster_animation_at_15, ...
        getframe(figure(4)));
    
end

close(cluster_animation_at_15);

%   Visualizing clusters at 30-min discontinuous contact duration:

minimum_contact_duration = 30 * 60;

next_cluster_members = ...
    extractClusters(period_start_time, period_end_time, ...
    connectivities, time_delta, ...
    minimum_contact_duration, maximum_number_of_hops);

[cluster_members,cluster_indices,~] = ...
    uniquecell(next_cluster_members);

for i = 1:size(cluster_indices,1)
    
    cluster_colors(i,:) = randi(5,1,3) ./ 5;

end

figure(5)

figure_title = ...
    sprintf('Clusters @ 30-min discontinuous\ncontact duration');
title(figure_title,'FontSize',18);
hold on

map = imread('map.jpg');
image('CData',map, ...
    'XData', ...
    [min(modified_data(:,6)) max(modified_data(:,6))], ...
    'YData', ...
    [min(modified_data(:,5)) max(modified_data(:,5))])
hold on

plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
    'ko','MarkerFaceColor','y');
hold on

for i = 2:size(cluster_indices,1)
    
    plot(synthetic_lons( ...
        cluster_members{1,i},t), ...
        synthetic_lats( ...
        cluster_members{1,i},t), ...
        'ko','MarkerFaceColor',cluster_colors(i,:));
    hold on
    
end

axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
    min(modified_data(:,5)) max(modified_data(:,5))]);
ylabel('Latitude','FontSize',18);
xlabel('Longitude','FontSize',18);
hold on
grid on

saveas(figure(5),'clusters at 30.fig');
saveas(figure(5),'clusters at 30.bmp');

cluster_animation_at_30 = ...
    VideoWriter('cluster_animation_at_30.avi');
cluster_animation_at_30.FrameRate = 10;
cluster_animation_at_30.Quality = 75;
open(cluster_animation_at_30);

for t = period_start_time * 6:period_end_time * 6

    figure(6)
    
    clf;

    figure_title = ...
        sprintf('Clusters @ 30-min discontinuous\ncontact duration');
    title(figure_title,'FontSize',18);
    hold on
    
    map = imread('map.jpg');
    image('CData',map, ...
        'XData', ...
        [min(modified_data(:,6)) max(modified_data(:,6))], ...
        'YData', ...
        [min(modified_data(:,5)) max(modified_data(:,5))])
    hold on
    
    plot(synthetic_lons(:,t),synthetic_lats(:,t), ...
        'ko','MarkerFaceColor','y');
    hold on
    
    for i = 2:size(cluster_indices,1)
        
        plot(synthetic_lons( ...
            cluster_members{1,i},t), ...
            synthetic_lats( ...
            cluster_members{1,i},t), ...
            'ko','MarkerFaceColor',cluster_colors(i,:));
        hold on
        
    end
    
    axis([min(modified_data(:,6)) max(modified_data(:,6)) ...
        min(modified_data(:,5)) max(modified_data(:,5))]);
    ylabel('Latitude','FontSize',18);
    xlabel('Longitude','FontSize',18);
    hold on
    
    time = ...
        sprintf('%02d:%02d',floor(t / (60 * 6)), ...
        floor((t / (60 * 6) - floor(t / (60 * 6))) * 60));
    text(-80.35,43.575, ...
        time,'Color','black','FontSize',20,'FontWeight','bold')
    hold on
    
    grid on
    
    drawnow;

    writeVideo(cluster_animation_at_30, ...
        getframe(figure(6)));
    
end

close(cluster_animation_at_30);

%%
 
%   Written by "Kais Suleiman" (ksuleiman.weebly.com)
