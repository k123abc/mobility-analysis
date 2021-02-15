#%%
# Written by "Kais Suleiman"(ksuleiman.weebly.com)
#
# Notes:
#
# - The contents of this script have been explained in details in Chapter 4
#   of the thesis:
#
#   Kais Suleiman, "Popular Content Distribution in Public Transportation
#   Using Artificial Intelligence Techniques.", Ph.D.thesis, University of
#   Waterloo, Ontario, Canada, 2019.
#
# - Simpler but still similar variable names have been used throughout this
#   script instead of the mathematical notations used in the thesis.
# - The assumptions used in the script are the same as those used in the thesis
#   including those related to the case study considered representing the Grand
#   River Transit bus service offered throughout the Region of Waterloo, Ontario,
#   Canada.
# - Figures are created throughout this script to aid in thesis
#   visualizations and other forms of results sharing.
#
#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Generating data:

import pandas as pd
import numpy as np

modified_trips = pd.read_excel('modified_trips.xlsx')
modified_trips.to_numpy()
np.save('modified_trips.npy', modified_trips)

modified_stop_times = pd.read_excel('modified_stop_times.xlsx')
modified_stop_times.to_numpy(dtype='float')
np.save('modified_stop_times.npy', modified_stop_times)

stops = pd.read_excel('stops.xlsx')
stops.to_numpy()
np.save('stops.npy', stops)

shapes = pd.read_excel('shapes.xlsx')
shapes.to_numpy()
np.save('shapes.npy', shapes)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys 

modified_trips = \
    np.load('modified_trips.npy', allow_pickle = True)
modified_stop_times = \
    np.load('modified_stop_times.npy', allow_pickle = True)
stops = np.load('stops.npy', allow_pickle = True)

# Collecting data:

data = np.zeros((np.shape(modified_stop_times)[0], 12))

for i in range(np.shape(modified_stop_times)[0]):

    data[i,:] = \
        np.concatenate( \
                       (modified_trips[modified_trips[:, 2] == \
                                       modified_stop_times[i, 0], [5, 0, 4]], \
                        modified_stop_times[i, 0: 5], \
                            stops[stops[:, 0] == modified_stop_times[i, 3], [4, 5]], \
                                modified_trips[modified_trips[:, 2] == modified_stop_times[i, 0], [6, 1]]), \
                           axis = 0)
    
    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(modified_stop_times)[0] \
                                * 100)))
    sys.stdout.flush()

# Collecting weekday data only:

data = data[data[:, 11] == 0, :]
data = np.delete(data, 11, 1)

np.save('data.npy', data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np

data = np.load('data.npy', allow_pickle = True)

# Sorting data:

data = data[data[:, 7].argsort()]
data = data[data[:, 4].argsort(kind = 'mergesort')]
data = data[data[:, 0].argsort(kind = 'mergesort')]

# Re - sorting same - time row data
# with switching directions:

# Re - sorting while assuming end-of-trip:

for i in range(1, np.shape(data)[0] - 1):

    if (data[i, 2] != data[i - 1, 2]) \
        and (data[i, 2] != data[i + 1, 2]):

        block_data = data[data[:, 0] == data[i, 0], :]

        same_time_block_data = \
            block_data[block_data[:, 4] == data[i, 4], :]

        if data[i, 2] == 0:

            # Sorting in a descending order:
                
            same_time_block_data = \
                same_time_block_data[same_time_block_data[:, 2].argsort()[::-1]]

        else:

            # Sorting in an ascending order:

            same_time_block_data = \
                same_time_block_data[same_time_block_data[:, 2].argsort()]

        block_data[block_data[:, 4] == data[i, 4], :] = \
            same_time_block_data

        data[data[:, 0] == data[i, 0], :] = block_data


# Re - sorting while assuming beginning-of-trip:

for i in range(1, np.shape(data)[0] - 1):

    if (data[i, 2] != data[i - 1, 2]) \
        and (data[i, 2] != data[i + 1, 2]):

        block_data = data[data[:, 0] == data[i, 0], :]

        same_time_block_data = \
            block_data[block_data[:, 4] == data[i, 4], :]

        if data[i, 2] == 0:

            # Sorting in an ascending order:
            
            same_time_block_data = \
                same_time_block_data[same_time_block_data[:, 2].argsort()]

        else:

            # Sorting in a descending order:

            same_time_block_data = \
                same_time_block_data[same_time_block_data[:, 2].argsort()[::-1]]

        block_data[block_data[:, 4] == data[i, 4], :] = \
            same_time_block_data

        data[data[:, 0] == data[i, 0], :] = block_data

_, idx = np.unique(data, axis = 0, return_index = True)
data = data[np.sort(idx)]

np.save('data.npy', data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import geopy.distance

data = np.load('data.npy', allow_pickle = True)

# Computing speeds with noisy trips included:

data = np.concatenate((data, \
                      np.zeros((np.shape(data)[0], 1))), axis = 1)

for i in range(1,np.shape(data)[0]):

    if data[i, 0] == data[i - 1, 0]:
        
        distance_difference = \
            geopy.distance.geodesic((data[i, 8], data[i, 9]), \
                                    (data[i - 1, 8], data[i - 1, 9])).km

        if distance_difference == 0:

            data[i, 11] = 0

        else:

            time_difference = \
                (data[i, 4] - data[i - 1, 5]) * 24

            if time_difference == 0:

                data[i, 11] = np.NaN

            else:

                data[i, 11] = \
                    distance_difference / time_difference

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(data)[0] \
                                * 100)))
    sys.stdout.flush()
    
np.save('data_with_noise.npy', data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import geopy.distance

data = np.load('data.npy', allow_pickle = True)

# Manually removing noisy trips:

data = data[data[:, 3] != 1428788, :]
data = data[data[:, 3] != 1428789, :]
data = data[data[:, 3] != 1428790, :]
data = data[data[:, 3] != 1428791, :]
data = data[data[:, 3] != 1428806, :]

# Computing speeds with noisy trips excluded:

data = np.concatenate((data, \
                       np.zeros((np.shape(data)[0], 1))), axis = 1)

for i in range(1,np.shape(data)[0]):

    if data[i, 0] == data[i - 1, 0]:

        distance_difference = \
            geopy.distance.geodesic((data[i, 8], data[i, 9]), \
                                    (data[i - 1, 8], data[i - 1, 9])).km

        if distance_difference == 0:

            data[i, 11] = 0

        else:

            time_difference = \
                (data[i, 4] - data[i - 1, 5]) * 24

            if time_difference == 0:

                data[i, 11] = np.NaN

            else:

                data[i, 11] = \
                    distance_difference / time_difference

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(data)[0] \
                                * 100)))
    sys.stdout.flush()

np.save('data_without_noise.npy', data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Comparing speeds before and after removing noisy trips:

# Before removing noisy trips:
    
import numpy as np
import matplotlib.pyplot as plt

data = np.load('data_with_noise.npy', allow_pickle = True)

_, idx = np.unique(data[:, 3], axis = 0, return_index = True)
trip_ids = data[np.sort(idx), 3]
trip_average_speeds = np.zeros((np.shape(trip_ids)[0], 1))

for i in range(np.shape(trip_ids)[0]):

    trip_speed_data = data[data[:, 3] == trip_ids[i], 11]
    trip_speed_data = \
        trip_speed_data[~ np.isnan(trip_speed_data)]

    trip_average_speeds[i] = trip_speed_data.mean(axis = 0)

figure1, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

ax1.scatter(trip_ids, trip_average_speeds, color = 'r')
ax1.set_ylim([0, 200])
ax1.set_title('Speeds before removing noisy trips')
ax1.set_xlabel('Trip ID')
ax1.set_ylabel('Average trip\nspeed (km/h)')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

# After removing noisy trips:

data = np.load('data_without_noise.npy', allow_pickle = True)

_, idx = np.unique(data[:, 3], axis = 0, return_index = True)
trip_ids = data[np.sort(idx), 3]
trip_average_speeds = np.zeros((np.shape(trip_ids)[0], 1))

for i in range(np.shape(trip_ids)[0]):

    trip_speed_data = data[data[:, 3] == trip_ids[i], 11]
    trip_speed_data = \
        trip_speed_data[~ np.isnan(trip_speed_data)]

    trip_average_speeds[i] = trip_speed_data.mean(axis = 0)

ax2.scatter(trip_ids, trip_average_speeds, color = 'g')
ax2.set_ylim([0, 200])
ax2.set_title('Speeds after removing noisy trips')
ax2.set_xlabel('Trip ID')
ax2.set_ylabel('Average trip\nspeed (km/h)')
ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('before and after removing noisy trips.png', dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt
import sys
import geopy.distance

data = np.load('data_without_noise.npy', allow_pickle = True)

# Cleaning data:

# Replacing speed errors:

_, idx = np.unique(data[:, 0], axis = 0, return_index = True)
block_ids = data[np.sort(idx), 0]
percentage_of_block_speed_NaNs = \
    np.zeros((np.shape(block_ids)[0], 1))

for i in range(np.shape(block_ids)[0]):

    percentage_of_block_speed_NaNs[i] = \
        np.sum(np.isnan(data[data[:, 0] == block_ids[i], 11])) / \
            np.shape(data[data[:, 0] == block_ids[i], :])[0] * 100

figure1, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)

ax1.scatter(block_ids, percentage_of_block_speed_NaNs, color = 'r')
ax1.set_ylim([0, 100])
ax1.set_title('Percentage of block speed NaN-values (Before replacing errors)')
ax1.set_xlabel('Block ID')
ax1.set_ylabel('Percentage of speed\nNaN-values')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

_, idx = np.unique(data[:, 0], axis = 0, return_index = True)
block_ids = data[np.sort(idx), 0]

for i in range(np.shape(block_ids)[0]):

    block_data = data[data[:, 0] == block_ids[i], :]

    while np.any(np.isnan(block_data[:, 11])):

        for j in range(1, np.shape(block_data)[0] - 1):

            if (np.isnan(block_data[j, 11])) \
                and (~ np.isnan(block_data[j + 1, 11])):

                waiting_time = block_data[j, 5] - block_data[j, 4]

                block_data[j, 4] = \
                    block_data[j - 1, 5] + \
                        (block_data[j + 1, 4] - block_data[j - 1, 5]) * 1 / 2

                block_data[j, 5] = \
                    np.minimum(block_data[j, 4] + waiting_time, \
                           block_data[j + 1, 4])

                distance_difference = \
                    geopy.distance.geodesic((block_data[j, 8], block_data[j, 9]), \
                                            (block_data[j - 1, 8], block_data[j - 1, 9])).km

                time_difference = \
                    (block_data[j, 4] - block_data[j - 1, 5]) * 24

                block_data[j, 11] = \
                    distance_difference / time_difference

                distance_difference = \
                    geopy.distance.geodesic((block_data[j, 8], block_data[j, 9]), \
                                            (block_data[j + 1, 8], block_data[j + 1, 9])).km

                time_difference = \
                    (block_data[j + 1, 4] - block_data[j, 5]) * 24

                if time_difference == 0:

                    block_data[j + 1, 11] = np.NaN

                else:

                    block_data[j + 1, 11] = \
                        distance_difference / time_difference

        if np.isnan(block_data[-1, 11]):

            block_data[-1, 11] = block_data[-2, 11]

            distance_difference = \
                geopy.distance.geodesic((block_data[-1, 8], block_data[-1, 9]), \
                                        (block_data[-2, 8], block_data[-2, 9])).km

            time_difference = \
                distance_difference / block_data[-1, 11] / 24

            waiting_time = block_data[-1, 5] - block_data[-1, 4]

            block_data[-1, 4] = \
                block_data[-2, 5] + time_difference

            block_data[-1, 5] = block_data[-1, 4] + waiting_time

    data[data[:, 0] == block_ids[i],:] = block_data

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(block_ids)[0] \
                                * 100)))
    sys.stdout.flush()
    
high_speed_measurement_indices = \
    np.where(data[:, 11] > 115)

for i in range(np.shape(high_speed_measurement_indices)[0]):

    j = high_speed_measurement_indices[0][i]

    data[j, 11] = 115

    distance_difference = \
        geopy.distance.geodesic((data[j, 8], data[j, 9]), \
                                (data[j - 1, 8], data[j - 1, 9])).km

    time_difference = \
        distance_difference / data[j, 11] / 24

    waiting_time = data[j, 5] - data[j, 4]

    data[j, 4] = \
        data[j - 1, 5] + time_difference

    data[j, 5] = \
        np.minimum(data[j, 4] + waiting_time, data[j + 1, 4])

    distance_difference = \
        geopy.distance.geodesic((data[j, 8], data[j, 9]), \
                                (data[j + 1, 8], data[j + 1, 9])).km

    time_difference = \
        (data[j + 1, 4] - data[j, 5]) * 24

    data[j + 1, 11] = \
        distance_difference / time_difference

percentage_of_block_speed_NaNs = np.zeros((np.shape(block_ids)[0], 1))

for i in range(np.shape(block_ids)[0]):

    percentage_of_block_speed_NaNs[i] = \
        np.sum(np.isnan(data[data[:, 0] == block_ids[i], 11])) / \
            np.shape(data[data[:, 0] == block_ids[i],:])[0] * 100

ax2.scatter(block_ids, percentage_of_block_speed_NaNs, color = 'g')
ax2.set_ylim([0, 100])
ax2.set_title('Percentage of block speed NaN-values (After replacing errors)')
ax2.set_xlabel('Block ID')
ax2.set_ylabel('Percentage of speed\nNaN-values')
ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('before and after replacing speed errors.png', dpi = 500)

# Visualizing speed distribution:

figure2, ax = plt.subplots(nrows = 1, ncols = 1)
ax.hist(data[data[:, 11] > 5, 11], 100, density = True)
# Only faster - than - walking - speed values are included
ax.set_title('Bus speed distribution')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Probability')
ax.set_xlim([0, 120])
ax.set_ylim([0, 0.06])
ax.grid(color = 'k', linestyle = '--', linewidth = 1)

figure2.tight_layout()
figure2.savefig('bus speed distribution.png', dpi = 500)

cleaned_data = data

np.save('cleaned_data.npy', cleaned_data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys

cleaned_data = np.load('cleaned_data.npy', allow_pickle = True)

# Modifying data:

modified_data = cleaned_data[:, [0,10,3,4,5,8,9]]

for i in range(125949):

    if modified_data[i, 3] != modified_data[i, 4]:

        extra_row = modified_data[i,:].reshape(1, 7)
        extra_row[0, 3] = modified_data[i, 4]
        modified_data = \
            np.insert(modified_data, i + 1, extra_row, 0) 

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(modified_data)[0] \
                                * 100)))
    sys.stdout.flush()
    
modified_data = np.delete(modified_data, 4, 1)

np.save('modified_data.npy', modified_data)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import geopy.distance

modified_data = np.load('modified_data.npy', allow_pickle = True)
shapes = np.load('shapes.npy', allow_pickle = True)

# Synthesizing data:

_, idx = np.unique(modified_data[:, 2], axis = 0, return_index = True)
trip_ids = modified_data[np.sort(idx), 2]
synthetic_trip_lats = np.zeros((np.shape(trip_ids)[0], 27 * 60 * 6))
synthetic_trip_lons = np.zeros((np.shape(trip_ids)[0], 27 * 60 * 6))

for i in range(np.shape(trip_ids)[0]):

    trip_trajectory = \
        modified_data[modified_data[:, 2] == trip_ids[i], :]

    lat_model = \
        np.poly1d(np.polyfit(trip_trajectory[:, 3], trip_trajectory[:, 4], 1))

    lon_model = \
        np.poly1d(np.polyfit(trip_trajectory[:, 3], trip_trajectory[:, 5], 1))

    for t in range(27 * 60 * 6):

        if (t / (24 * 60 * 6) >= np.amin(trip_trajectory[:, 3])) \
            and (t / (24 * 60 * 6) <= np.amax(trip_trajectory[:, 3])):

            model_result = \
                [lat_model(t / (24 * 60 * 6)), lon_model(t / (24 * 60 * 6))]

            # Map - matching:

            shape_trajectory = \
                shapes[shapes[:, 0] == trip_trajectory[0, 1], :]
            
            shape_trajectory = shape_trajectory[:, [0, 1, 2]]

            distances = np.zeros((np.shape(shape_trajectory)[0], 1))

            for j in range(np.shape(shape_trajectory)[0]):

                distances[j] = \
                    geopy.distance.geodesic((model_result[0], model_result[1]), \
                                            (shape_trajectory[j, 1], shape_trajectory[j, 2])).km

            closest_point = np.where(distances == np.amin(distances))
            closest_point = closest_point[0]

            synthetic_trip_lats[i, t] = \
                shape_trajectory[closest_point[0], 1]

            synthetic_trip_lons[i, t] = \
                shape_trajectory[closest_point[0], 2]

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(trip_ids)[0] \
                                * 100)))
    sys.stdout.flush()

_, idx = np.unique(modified_data[:, 0], axis = 0, return_index = True)
block_ids = modified_data[np.sort(idx), 0]
synthetic_block_lats = \
    np.zeros((np.shape(block_ids)[0], 27 * 60 * 6))
synthetic_block_lons = \
    np.zeros((np.shape(block_ids)[0], 27 * 60 * 6))

for i in range(np.shape(block_ids)[0]):

    temp = modified_data[modified_data[:, 0] == block_ids[i], 2]
    _, idx = np.unique(temp, axis = 0, return_index = True)
    block_trips = temp[np.sort(idx)]

    for t in range(27 * 60 * 6):

        synthetic_block_lats[i, t] = \
            np.sum(synthetic_trip_lats[ \
                                       list(range(np.where(trip_ids == block_trips[0])[0][0], \
                                           np.where(trip_ids == block_trips[-1])[0][0])), t])

        synthetic_block_lons[i, t] = \
            np.sum(synthetic_trip_lons[ \
                                       list(range(np.where(trip_ids == block_trips[0])[0][0], \
                                           np.where(trip_ids == block_trips[-1])[0][0])), t])

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(block_ids)[0] \
                                * 100)))
    sys.stdout.flush()

# Filling synthetic block data gaps between same - block trips:

# Notice that most of these gaps are for buses waiting between
# the different trips

for i in range(np.shape(block_ids)[0]):

    block_modified_data = \
        modified_data[modified_data[:, 0] == block_ids[i],:]

    block_start_time = block_modified_data[0, 3]
    block_end_time = block_modified_data[-1, 3]

    for t in range(27 * 60 * 6):

        if (t / (24 * 60 * 6) >= block_start_time) \
            and (t / (24 * 60 * 6) <= block_end_time):

            if (synthetic_block_lats[i, t] == 0):

                gap_size = 0

                while synthetic_block_lats[i, t + gap_size] == 0:

                    gap_size = gap_size + 1

                gap_lat_trajectory = \
                    np.vstack(([(t - 1) / (24 * 60 * 6), synthetic_block_lats[i, t - 1]], \
                                    [(t + gap_size) / (24 * 60 * 6), synthetic_block_lats[i, t + gap_size]]))

                gap_lat_model = \
                    np.poly1d(np.polyfit(gap_lat_trajectory[:, 0], gap_lat_trajectory[:, 1], 1))

                synthetic_block_lats[i, t: t + gap_size] = \
                    gap_lat_model([j / (24 * 60 * 6) for j in list(range(t, t + gap_size))])

            if (synthetic_block_lons[i, t] == 0):

                gap_size = 0

                while synthetic_block_lons[i, t + gap_size] == 0:

                    gap_size = gap_size + 1

                gap_lon_trajectory = \
                    np.vstack(([(t - 1) / (24 * 60 * 6), synthetic_block_lons[i, t - 1]], \
                                    [(t + gap_size) / (24 * 60 * 6), synthetic_block_lons[i, t + gap_size]]))

                gap_lon_model = \
                    np.poly1d(np.polyfit(gap_lon_trajectory[:, 0], gap_lon_trajectory[:, 1], 1))

                synthetic_block_lons[i, t: t + gap_size] = \
                    gap_lon_model([j / (24 * 60 * 6) for j in list(range(t, t + gap_size))])

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(block_ids)[0] \
                                * 100)))
    sys.stdout.flush()

# Correcting same - block trip single - step overlaps:

# Notice that these overlaps exist because of the same single - step shared
# between previous - trip and next - trip where both arrival times are
# equal as well as their stop coordinates.

max_synthetic_trip_lat = np.amax(np.amax(synthetic_trip_lats))
min_synthetic_trip_lon = np.amin(np.amin(synthetic_trip_lons))

for i in range(np.shape(block_ids)[0]):

    block_modified_data = \
        modified_data[modified_data[:, 0] == block_ids[i],:]

    block_start_time = block_modified_data[0, 3]
    block_end_time = block_modified_data[-1, 3]

    for t in range(27 * 60 * 6):

        if (t / (24 * 60 * 6) >= block_start_time) \
            and (t / (24 * 60 * 6) <= block_end_time):

            if (synthetic_block_lats[i, t] > max_synthetic_trip_lat) \
                or (synthetic_block_lons[i, t] < min_synthetic_trip_lon):

                if ((synthetic_block_lats[i, t - 1] \
                     < max_synthetic_trip_lat) \
                    and (synthetic_block_lats[i, t + 1] \
                         < max_synthetic_trip_lat)) \
                    or ((synthetic_block_lons[i, t - 1] \
                         > min_synthetic_trip_lon) \
                        and (synthetic_block_lons[i, t + 1] \
                             > min_synthetic_trip_lon)):

                    synthetic_block_lats[i, t] = \
                        synthetic_block_lats[i, t] / 2
                    synthetic_block_lons[i, t] = \
                        synthetic_block_lons[i, t] / 2

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(block_ids)[0] \
                                * 100)))
    sys.stdout.flush()

np.save('synthetic_trip_lats.npy', synthetic_trip_lats)
np.save('synthetic_trip_lons.npy', synthetic_trip_lons)
np.save('synthetic_block_lats.npy', synthetic_block_lats)
np.save('synthetic_block_lons.npy', synthetic_block_lons)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

synthetic_block_lats = np.load('synthetic_block_lats.npy', allow_pickle = True)
synthetic_block_lons = np.load('synthetic_block_lons.npy', allow_pickle = True)
modified_data = np.load('modified_data.npy', allow_pickle = True)
shapes = np.load('shapes.npy', allow_pickle = True)
synthetic_trip_lats = np.load('synthetic_trip_lats.npy', allow_pickle = True)
synthetic_trip_lons = np.load('synthetic_trip_lons.npy', allow_pickle = True)

# Visualizing synthetic data:

number_of_blocks_vs_time = np.empty((0,), int)

for t in range(27 * 60 * 6):

    number_of_blocks_vs_time = \
        np.append(number_of_blocks_vs_time, \
                  [np.shape(np.where(synthetic_block_lats[:, t] != 0))[1]], axis = 0)

number_of_blocks_per_hour = np.empty((0, ), int)

for h in range(27):

    number_of_blocks_per_hour = \
        np.append(number_of_blocks_per_hour, \
                  [np.mean(number_of_blocks_vs_time[list(range((h - 1) * 60 * 6 + 1, h * 60 * 6))])])

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)
ax1.set_title('Number of buses vs. time')
ax1.bar(list(range(27)), number_of_blocks_per_hour)
ax1.set_xlim([4, 27.5])
ax1.set_ylabel('Number of buses')
ax1.set_xlabel('Time (hours)')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('number of buses vs. time.png', dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Buses @ 6:00 AM')

map = plt.imread('python_map.jpg')
ax2.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])
    
ax2.scatter(synthetic_block_lons[:, 6 * 60 * 6], \
         synthetic_block_lats[:, 6 * 60 * 6], \
             color = 'yellow', marker = 'o', edgecolors = 'black')
ax2.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax2.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

figure2.tight_layout()
figure2.savefig('buses at 6 am.png', dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Buses @ 5:00 PM')

map = plt.imread('python_map.jpg')
ax3.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])
    
ax3.scatter(synthetic_block_lons[:, 17 * 60 * 6], \
         synthetic_block_lats[:, 17 * 60 * 6], \
             color = 'yellow', marker = 'o', edgecolors = 'black')
ax3.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax3.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')

figure3.tight_layout()
figure3.savefig('buses at 5 pm.png', dpi = 500)

temp = modified_data[:, 2]
_, idx = np.unique(temp, axis = 0, return_index = True)
trip_ids = temp[np.sort(idx)]

trip_index = 1 # Chosen at random

trip_modified_data = \
    modified_data[modified_data[:, 2] == trip_ids[trip_index],:]

realistic_trajectory = \
    shapes[shapes[:, 0] == trip_modified_data[0, 1], :]
realistic_trajectory = realistic_trajectory[:, [1, 2]]

synthetic_trajectory = \
    np.vstack((synthetic_trip_lats[trip_index, :], \
                    synthetic_trip_lons[trip_index, :]))

synthetic_trajectory = \
    synthetic_trajectory[:, synthetic_trajectory[0, :] != 0]

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Synthetic vs. realistic\ntrip trajectories')

map = plt.imread('python_trajectories_map.jpg')
ax4.imshow(map, \
           extent = [np.amin(realistic_trajectory[:, 1]), \
                     np.amax(realistic_trajectory[:, 1]), \
                         np.amin(realistic_trajectory[:, 0]), \
                             np.amax(realistic_trajectory[:, 0])])

ax4.plot(realistic_trajectory[:, 1], realistic_trajectory[:, 0], \
         color = 'blue', linewidth = 2, label = 'Realistic')
ax4.scatter(synthetic_trajectory[1, :], synthetic_trajectory[0, :], \
            color = 'yellow', marker = 'o', edgecolors = 'black', \
                label = 'Synthetic')
ax4.set_xlim([np.amin(realistic_trajectory[:, 1]), np.amax(realistic_trajectory[:, 1])])
ax4.set_ylim([np.amin(realistic_trajectory[:, 0]), np.amax(realistic_trajectory[:, 0])])
ax4.set_ylabel('Latitude')
ax4.set_xlabel('Longitude')
ax4.legend(loc = 'best')

figure4.tight_layout()
figure4.savefig('synthetic vs. realistic trip trajectories.png', dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import geopy.distance

cleaned_data = np.load('cleaned_data.npy', allow_pickle = True)

temp = cleaned_data[:, [6, 8, 9]]
_, idx = np.unique(temp, axis = 0, return_index = True)
stops_data = temp[np.sort(idx)]
stops_data = \
    np.append(stops_data, np.zeros((np.shape(stops_data)[0], 3)), \
              axis = 1)

# Converting stop - coordinates data:

stops_data[:, 3] = \
    6371 * np.cos(stops_data[:, 1] * np.pi / 180) * \
        np.cos(stops_data[:, 2] * np.pi / 180)

stops_data[:, 4] = \
    6371 * np.cos(stops_data[:, 1] * np.pi / 180) * \
        np.sin(stops_data[:, 2] * np.pi / 180)

stops_data[:, 5] = \
    6371 * np.sin(stops_data[:, 1] * np.pi / 180)

# Refining stop ids:

neighborhood_distance = 300 / 1000

stop_clusters_model = \
    AgglomerativeClustering(n_clusters = None, linkage = 'complete', \
                            distance_threshold = neighborhood_distance). \
        fit(stops_data[:, [3, 4, 5]])

number_of_clusters = stop_clusters_model.n_clusters_

refined_stop_ids = np.zeros((number_of_clusters, 1))

stop_cluster_labels = stop_clusters_model.labels_

for i in range(number_of_clusters):

    cluster_members = \
        stops_data[stop_cluster_labels == i, 0]
    cluster_members = \
        cluster_members.reshape(np.shape(cluster_members)[0], 1)

    cluster_size = \
        np.shape(cluster_members)[0]

    cluster_member_coordinates = \
        stops_data[stop_cluster_labels == i, :]
    cluster_member_coordinates = cluster_member_coordinates[:, [1, 2]]

    cluster_member_coordinates_mean = \
        np.mean(cluster_member_coordinates, axis = 0)

    cluster_member_mean_inter_distances = \
        np.zeros((cluster_size, 1))

    for j in range(cluster_size):

        cluster_member_mean_inter_distances[j] = \
            geopy.distance.geodesic((cluster_member_coordinates[j, :]), \
                                    cluster_member_coordinates_mean).km

    cluster_medoid = \
        cluster_members[ \
                        cluster_member_mean_inter_distances == \
                            np.amin(cluster_member_mean_inter_distances)]

    refined_stop_ids[i] = cluster_medoid

np.save('refined_stop_ids.npy', refined_stop_ids)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import geopy.distance
import matplotlib.pyplot as plt

refined_stop_ids = np.load('refined_stop_ids.npy', allow_pickle = True)
synthetic_block_lats = np.load('synthetic_block_lats.npy', allow_pickle = True)
synthetic_block_lons = np.load('synthetic_block_lons.npy', allow_pickle = True)
stops = np.load('stops.npy', allow_pickle = True)

# Choosing optimal stop ids:

stop_popularities = np.zeros((np.shape(refined_stop_ids)[0], 1))
broadcasting_range = 300
maximum_number_of_stops = 500 - np.shape(synthetic_block_lats)[0]

for i in range(np.shape(refined_stop_ids)[0]):

    for t in range(np.shape(synthetic_block_lats)[1]):

        if np.any(synthetic_block_lats[: ,t] != 0):

            block_coordinates = \
                np.vstack((synthetic_block_lats[synthetic_block_lats[:, t] != 0, t], \
                          synthetic_block_lons[synthetic_block_lons[:, t] != 0, t]))
        
            stop_coordinates = \
                stops[stops[:, 0] == refined_stop_ids[i], :]
            stop_coordinates = stop_coordinates[:, [4, 5]]

            for j in range(np.shape(block_coordinates)[1]):

                if geopy.distance.geodesic(stop_coordinates[0], \
                                           block_coordinates[:, j]).km \
                    <= broadcasting_range / 1000:

                    stop_popularities[i] = \
                        stop_popularities[i] + 1

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(refined_stop_ids)[0] \
                                * 100)))
    sys.stdout.flush()

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)
ax1.set_title('Refined stops popularities')
ax1.scatter([i for i in range(np.shape(refined_stop_ids)[0])], \
            stop_popularities, color = 'r')
ax1.set_xlim([0, np.shape(refined_stop_ids)[0]])
ax1.set_xlabel('Refined stop index')
ax1.set_ylabel('Popularity')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('refined stops popularities.png', dpi = 500)

stop_popularities = stop_popularities.flatten()
optimal_stop_ids = stop_popularities.argsort()[::-1]
optimal_stop_ids = refined_stop_ids[optimal_stop_ids]
optimal_stop_ids = \
    optimal_stop_ids[list(range(maximum_number_of_stops))]

np.save('stop_popularities.npy', stop_popularities)
np.save('optimal_stop_ids.npy', optimal_stop_ids)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import numpy.matlib

synthetic_block_lats = np.load('synthetic_block_lats.npy', allow_pickle = True)
synthetic_block_lons = np.load('synthetic_block_lons.npy', allow_pickle = True)
optimal_stop_ids = np.load('optimal_stop_ids.npy', allow_pickle = True)
stops = np.load('stops.npy', allow_pickle = True)

# Adding optimal stops data to the synthetic block data:

synthetic_lats = \
    np.zeros((np.shape(synthetic_block_lats)[0] + \
              np.shape(optimal_stop_ids)[0], \
                  np.shape(synthetic_block_lats)[1]))

synthetic_lats[list(range(np.shape(synthetic_block_lats)[0])), :] = \
    synthetic_block_lats

synthetic_lons = \
    np.zeros((np.shape(synthetic_block_lons)[0] + \
              np.shape(optimal_stop_ids)[0], \
                  np.shape(synthetic_block_lons)[1]))

synthetic_lons[list(range(np.shape(synthetic_block_lons)[0])), :] = \
    synthetic_block_lons

for i in range(np.shape(optimal_stop_ids)[0]):

    synthetic_lats[np.shape(synthetic_block_lats)[0] + i, :] = \
        np.matlib.repmat( \
                         stops[stops[:, 0] == optimal_stop_ids[i], 4], \
                             1, np.shape(synthetic_block_lats)[1])

    synthetic_lons[np.shape(synthetic_block_lons)[0] + i, :] = \
        np.matlib.repmat( \
                         stops[stops[:, 0] == optimal_stop_ids[i], 5], \
                             1, np.shape(synthetic_block_lons)[1])

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(i / np.shape(optimal_stop_ids)[0] \
                                * 100)))
    sys.stdout.flush()

np.save('synthetic_lats.npy', synthetic_lats)
np.save('synthetic_lons.npy', synthetic_lons)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

modified_data = np.load('modified_data.npy', allow_pickle = True)
stops = np.load('stops.npy', allow_pickle = True)
refined_stop_ids = np.load('refined_stop_ids.npy', allow_pickle = True)
optimal_stop_ids = np.load('optimal_stop_ids.npy', allow_pickle = True)

# Visualizing all stops, refined stops and optimal stops:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('All stops')

map = plt.imread('python_map.jpg')
ax1.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])

ax1.scatter(stops[:, 5], stops[:, 4], \
            color = 'yellow', marker = 'o', edgecolors = 'black')
ax1.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax1.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')

figure1.tight_layout()
figure1.savefig('all stops.png', dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Refined stops')

map = plt.imread('python_map.jpg')
ax2.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])

for i in range(np.shape(refined_stop_ids)[0]):
        
    ax2.scatter(stops[stops[:, 0] == refined_stop_ids[i], 5], \
                stops[stops[:, 0] == refined_stop_ids[i], 4], \
                    color = 'yellow', marker = 'o', edgecolors = 'black')

ax2.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax2.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

figure2.tight_layout()
figure2.savefig('refined stops.png', dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Optimal stops')

map = plt.imread('python_map.jpg')
ax3.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])

for i in range(np.shape(optimal_stop_ids)[0]):

    ax3.scatter(stops[stops[:, 0] == optimal_stop_ids[i], 5], \
                stops[stops[:, 0] == optimal_stop_ids[i], 4], \
                    color = 'yellow', marker = 'o', edgecolors = 'black')

ax3.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax3.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')

figure3.tight_layout()
figure3.savefig('optimal stops.png', dpi = 500)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import geopy.distance

synthetic_lats = np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)

broadcasting_range = 300
connectivities = np.zeros((np.shape(synthetic_lats)[1], \
                           np.shape(synthetic_lats)[0], \
                               np.shape(synthetic_lats)[0]))

# Determining inter - bus / stop connectivities:

for t in range(np.shape(synthetic_lats)[1]):

    for i in range(np.shape(synthetic_lats)[0] - 1):

        if synthetic_lats[i, t] != 0:

            for j in range(i + 1, np.shape(synthetic_lats)[0]):

                if synthetic_lats[j, t] != 0:

                    distance = \
                        geopy.distance.geodesic( \
                                                (synthetic_lats[i, t], \
                                                 synthetic_lons[i, t]), \
                                                    (synthetic_lats[j, t], \
                                                     synthetic_lons[j, t])).km \
                            * 1000

                    if distance <= broadcasting_range:

                        connectivities[t, i, j] = np.uint8(1)
                        # "uint8" is used to minimize memory consumption
                        # and increase execution speed

                        connectivities[t, j, i] = \
                            connectivities[t, i, j]

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(t / np.shape(synthetic_lats)[1] \
                                * 100)))
    sys.stdout.flush()

np.save('connectivities.npy', connectivities)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

modified_data = np.load('modified_data.npy', allow_pickle = True)
synthetic_lats = np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)
synthetic_block_lats = np.load('synthetic_block_lats.npy', allow_pickle = True)
connectivities = np.load('connectivities.npy', allow_pickle = True)

# Visualizing connectivities:

# Drawing connectivities:

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Connectivities @ 5:00 PM')

map = plt.imread('python_map.jpg')
ax1.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])
       
ax1.scatter(synthetic_lons[list(range(np.shape(synthetic_block_lats)[0] + 1, -1)), 17 * 60 * 6], \
            synthetic_lats[list(range(np.shape(synthetic_block_lats)[0] + 1, -1)), 17 * 60 * 6], \
                                color = 'blue', marker = 'o', \
                                    edgecolors = 'black', \
                                        label = 'Bus-stops')

ax1.scatter(synthetic_lons[list(range(np.shape(synthetic_block_lats)[0] + 1)), 17 * 60 * 6], \
            synthetic_lats[list(range(np.shape(synthetic_block_lats)[0] + 1)), 17 * 60 * 6], \
                                color = 'green', marker = 'o', \
                                    edgecolors = 'black', \
                                        label = 'Buses')
ax1.legend(loc = 'best')

for i in range(np.shape(synthetic_lats)[0]):

    for j in range(np.shape(synthetic_lats)[0]):

        if connectivities[17 * 60 * 6, i, j] == 1:

            ax1.plot([synthetic_lons[i, 17 * 60 * 6], \
                      synthetic_lons[j, 17 * 60 * 6]], \
                     [synthetic_lats[i, 17 * 60 * 6], \
                      synthetic_lats[j, 17 * 60 * 6]], \
                         color = 'red')

ax1.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax1.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')

figure1.tight_layout()
figure1.savefig('connectivities at 5 pm.png', dpi = 500)

# Visualizing zoomed - in connectivities:

# Drawing zoomed - in connectivities:

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Zoomed-in connectivities @ 5:00 PM')

map = plt.imread('python_zoomed_in_map.jpg')
ax2.imshow(map, extent = [-80.52, -80.48, 43.43, 43.47])

ax2.scatter(synthetic_lons[list(range(np.shape(synthetic_block_lats)[0] + 1, -1)), 17 * 60 * 6], \
            synthetic_lats[list(range(np.shape(synthetic_block_lats)[0] + 1, -1)), 17 * 60 * 6], \
                                color = 'blue', marker = 'o', \
                                    edgecolors = 'black', \
                                        label = 'Bus-stops')

ax2.scatter(synthetic_lons[list(range(np.shape(synthetic_block_lats)[0] + 1)), 17 * 60 * 6], \
            synthetic_lats[list(range(np.shape(synthetic_block_lats)[0] + 1)), 17 * 60 * 6], \
                                color = 'green', marker = 'o', \
                                    edgecolors = 'black', \
                                        label = 'Buses')
ax2.legend(loc = 'best')

for i in range(np.shape(synthetic_lats)[0]):

    for j in range(np.shape(synthetic_lats)[0]):

        if connectivities[17 * 60 * 6, i, j] == 1:

            ax2.plot([synthetic_lons[i, 17 * 60 * 6], \
                      synthetic_lons[j, 17 * 60 * 6]], \
                     [synthetic_lats[i, 17 * 60 * 6], \
                      synthetic_lats[j, 17 * 60 * 6]], \
                         color = 'red')

ax2.set_xlim([-80.52, - 80.48])
ax2.set_ylim([43.43, 43.47])
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

figure2.tight_layout()
figure2.savefig('zoomed-in connectivities at 5 pm.png', dpi = 500)

#%%

# The extractClusters function:

def extractClusters(period_start_time, period_end_time, \
                    connectivities, time_delta, \
                        minimum_contact_duration, \
                            maximum_number_of_hops):

    period_connectivities = \
        np.zeros((np.shape(connectivities[1, :, :])[0], \
                  np.shape(connectivities[1, :, :])[1]))
    
    previous_cluster_members = \
        np.zeros((np.shape(connectivities[1, :, :])[0], \
                  np.shape(connectivities[1, :, :])[1]))
    
    for i in range(np.shape(connectivities[1, :, :])[0]):

        for t in range(period_start_time * 6, period_end_time * 6):

            if t == period_start_time * 6:

                period_connectivities[i, :] = connectivities[t, i, :]

            else:

                period_connectivities[i, :] = \
                    period_connectivities[i, :] + \
                        connectivities[t, i, :]

        previous_cluster_members[i, :] = \
            np.where(period_connectivities[i, :] * time_delta >= \
                     minimum_contact_duration)

    next_cluster_members = previous_cluster_members

    for n in range(maximum_number_of_hops / 2 - 1):

        for i in range(np.shape(connectivities[1, :, :])[0]):

            for j in range(np.shape(previous_cluster_members[i, :])[1]):

                next_cluster_members[i, :] = \
                    next_cluster_members[i, :]. \
                        union(previous_cluster_members[ \
                                                       previous_cluster_members[i, j], :])

        previous_cluster_members = next_cluster_members

    return next_cluster_members

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import sys
import matplotlib.pyplot as plt

connectivities = np.load('connectivities.npy', allow_pickle = True)

# Clustering:

period_start_times = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25] * 60
period_end_times = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27] * 60
minimum_contact_durations = [5, 15, 30] * 60
time_delta = 10
maximum_number_of_hops = 20

number_of_clusters = \
    np.zeros((np.shape(period_start_times)[1], np.shape(minimum_contact_durations)[1]))
number_of_unclustered_nodes = \
    np.zeros((np.shape(period_start_times)[1], np.shape(minimum_contact_durations)[1]))

cluster_size_means = \
    np.zeros((np.shape(period_start_times)[1], np.shape(minimum_contact_durations)[1]))
cluster_size_minima = \
    np.zeros((np.shape(period_start_times)[1], np.shape(minimum_contact_durations)[1]))
cluster_size_maxima = \
    np.zeros((np.shape(period_start_times)[1], np.shape(minimum_contact_durations)[1]))

for p in range(np.shape(period_start_times)[1]):

    period_start_time = period_start_times[p]

    period_end_time = period_end_times[p]

    for d in range(np.shape(minimum_contact_durations)[1]):

        minimum_contact_duration = minimum_contact_durations[d]

        next_cluster_members = \
            extractClusters(period_start_time, period_end_time, \
                                 connectivities, time_delta, \
                                     minimum_contact_duration, \
                                         maximum_number_of_hops)

        _, cluster_indices = \
            np.unique(next_cluster_members, axis = 0, \
                      return_index = True)
        cluster_members = \
            next_cluster_members[np.sort(cluster_indices)]

        number_of_clusters[p, d] = np.shape(cluster_indices)[0] - 1
        # We substract 1 to exclude the empty cluster set

        cluster_sizes[p, d, :] = \
            np.zeros((number_of_clusters[p, d], 1))

        for i in range(number_of_clusters[p, d]):

            cluster_sizes[p, d, i] = \
                np.shape(cluster_members[i + 1, :])[1]

        number_of_unclustered_nodes[p, d] = 0

        for i in range(np.shape(next_cluster_members)[1]):

            if np.shape(next_cluster_members[i, :])[1] == 0:

                number_of_unclustered_nodes[p, d] = \
                    number_of_unclustered_nodes[p, d] + 1

    sys.stdout.write('\r' + \
                     str("Please wait ... {:.2f}%".\
                         format(p / np.shape(period_start_times)[1] \
                                * 100)))
    sys.stdout.flush()

for p in range(np.shape(period_start_times)[1]):

    for d in range(np.shape(minimum_contact_durations)[1]):

        cluster_size_means[p, d] = \
            np.mean(cluster_sizes[p, d, :])
        cluster_size_minima[p, d] = \
            np.amin(cluster_sizes[p, d, :])
        cluster_size_maxima[p, d] = \
            np.amax(cluster_sizes[p, d, :])

figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Number of clusters vs. day period')
ax1.set_xlim([0, np.shape(period_start_times)[1] + 1])
ax1.set_xlabel('Day period')
ax1.set_ylabel('Number of clusters')
ax1.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_clusters[:, 0], color = 'black', \
             label = '5-min discontinuous duration')
ax1.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_clusters[:, 1], color = 'blue', \
             label = '15-min discontinuous duration')
ax1.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_clusters[:, 2], color = 'red', \
             label = '30-min discontinuous duration')
xticklabels = ['5-7', '7-9', '9-11', '11-13', \
               '13-15', '15-17', '17-19', '19-21', '21-23', \
                   '23-25', '25-27']
ax1.set_xticks(list(range(np.shape(period_start_times)[1])), \
               xticklabels)
ax1.legend(loc = 'best')
ax1.grid(color = 'k', linestyle = '--', linewidth = 1)

figure1.tight_layout()
figure1.savefig('number of clusters vs. day period.png', dpi = 500)

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1)

ax2.set_title('Cluster size means vs. day period')
ax2.set_xlim([0, np.shape(period_start_times)[1] + 1])
ax2.set_xlabel('Day period')
ax2.set_ylabel('Cluster size mean')
ax2.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_means[:, 0], color = 'black', \
             label = '5-min discontinuous duration')
ax2.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_means[:, 1], color = 'blue', \
             label = '15-min discontinuous duration')
ax2.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_means[:, 2], color = 'red', \
             label = '30-min discontinuous duration')
ax2.set_xticks(list(range(np.shape(period_start_times)[1])), \
               xticklabels)
ax2.legend(loc = 'best')
ax2.grid(color = 'k', linestyle = '--', linewidth = 1)

figure2.tight_layout()
figure2.savefig('cluster size means vs. day period.png', dpi = 500)

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Cluster size maximum vs. day period')
ax3.set_xlim([0, np.shape(period_start_times)[1] + 1])
ax3.set_xlabel('Day period')
ax3.set_ylabel('Cluster size maximum')
ax3.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_maxima[:, 0], color = 'black', \
             label = '5-min discontinuous duration')
ax3.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_maxima[:, 1], color = 'blue', \
             label = '15-min discontinuous duration')
ax3.plot(list(range(np.shape(period_start_times)[1])), \
         cluster_size_maxima[:, 2], color = 'red', \
             label = '30-min discontinuous duration')
ax3.set_xticks(list(range(np.shape(period_start_times)[1])), \
               xticklabels)
ax3.legend(loc = 'best')
ax3.grid(color = 'k', linestyle = '--', linewidth = 1)

figure3.tight_layout()
figure3.savefig('cluster size maximum vs. day period.png', dpi = 500)

figure4, ax4 = plt.subplots(nrows = 1, ncols = 1)

ax4.set_title('Number of unclustered nodes vs. day period')
ax4.set_xlim([0, np.shape(period_start_times)[1] + 1])
ax4.set_xlabel('Day period')
ax4.set_ylabel('Number of unclustered nodes')
ax4.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_unclustered_nodes[:, 0], color = 'black', \
             label = '5-min discontinuous duration')
ax4.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_unclustered_nodes[:, 1], color = 'blue', \
             label = '15-min discontinuous duration')
ax4.plot(list(range(np.shape(period_start_times)[1])), \
         number_of_unclustered_nodes[:, 2], color = 'red', \
             label = '30-min discontinuous duration')
ax4.set_xticks(list(range(np.shape(period_start_times)[1])), \
               xticklabels)
ax4.legend(loc = 'best')
ax4.grid(color = 'k', linestyle = '--', linewidth = 1)

figure4.tight_layout()
figure4.savefig('number of unclustered nodes vs. day period.png', dpi = 500)

np.savez('clustering_results.npy', number_of_clusters, \
         cluster_sizes, number_of_unclustered_nodes)

#%%

# To clear memory

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import matplotlib.pyplot as plt

connectivities = np.load('connectivities.npy', allow_pickle = True)
synthetic_lats = np.load('synthetic_lats.npy', allow_pickle = True)
synthetic_lons = np.load('synthetic_lons.npy', allow_pickle = True)
modified_data = np.load('modified_data.npy', allow_pickle = True)

period_start_time = 16 * 60
period_end_time = 18 * 60
time_delta = 10
maximum_number_of_hops = 20

# Visualizing clusters at 5 - min discontinuous contact duration:

minimum_contact_duration = 5 * 60

next_cluster_members = \
    extractClusters(period_start_time, period_end_time, \
                         connectivities, time_delta, \
                             minimum_contact_duration, \
                                 maximum_number_of_hops)

_, cluster_indices = \
    np.unique(next_cluster_members, axis = 0, \
              return_index = True)
cluster_members = \
    next_cluster_members[np.sort(cluster_indices)]

cluster_colors = \
    np.zeros((np.shape(cluster_indices)[0], 3))

for i in range(np.shape(cluster_indices)[0]):

    cluster_colors[i, :] = \
        np.random.randint(5, size = (1, 3)) / 5
    
figure1, ax1 = plt.subplots(nrows = 1, ncols = 1)

ax1.set_title('Clusters @ 5-min discontinuous\ncontact duration')

map = plt.imread('python_map.jpg')
ax1.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])
    
ax1.scatter(synthetic_lons[:, t], synthetic_lats[:, t], \
            color = 'yellow', marker = 'o', edgecolors = 'black')
    
for i in range(1, np.shape(cluster_indices)[0]):
    
    ax1.scatter(synthetic_lons[ \
                               cluster_members[0, i], t], \
                synthetic_lats[ \
                               cluster_members[0, i], t], \
                    color = cluster_colors[i, :], marker = 'o', \
                        edgecolors = 'black')

ax1.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax1.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')

figure1.tight_layout()
figure1.savefig('clusters at 5.png', dpi = 500)

# Visualizing clusters at 15 - min discontinuous contact duration:

minimum_contact_duration = 15 * 60

next_cluster_members = \
    extractClusters(period_start_time, period_end_time, \
                         connectivities, time_delta, \
                             minimum_contact_duration, \
                                 maximum_number_of_hops)

_, cluster_indices = \
    np.unique(next_cluster_members, axis = 0, \
              return_index = True)
cluster_members = \
    next_cluster_members[np.sort(cluster_indices)]

cluster_colors = \
    np.zeros((np.shape(cluster_indices)[0], 3))
    
for i in range(np.shape(cluster_indices)[0]):

    cluster_colors[i, :] = \
        np.random.randint(5, size = (1, 3)) / 5

figure2, ax2 = plt.subplots(nrows = 1, ncols = 1) 

ax2.set_title('Clusters @ 15-min discontinuous\ncontact duration')

map = plt.imread('python_map.jpg')
ax2.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])

ax2.scatter(synthetic_lons[:, t], synthetic_lats[:, t], \
            color = 'yellow', marker = 'o', edgecolors = 'black')

for i in range(1, np.shape(cluster_indices)[0]):

    ax2.scatter(synthetic_lons[ \
                               cluster_members[0, i], t], \
                synthetic_lats[ \
                               cluster_members[0, i], t], \
                    color = cluster_colors[i, :], marker = 'o', \
                        edgecolors = 'black')

ax2.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax2.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

figure2.tight_layout()
figure2.savefig('clusters at 15.png', dpi = 500)

# Visualizing clusters at 30 - min discontinuous contact duration:

minimum_contact_duration = 30 * 60

next_cluster_members = \
    extractClusters(period_start_time, period_end_time, \
                         connectivities, time_delta, \
                             minimum_contact_duration, \
                                 maximum_number_of_hops)

_, cluster_indices = \
    np.unique(next_cluster_members, axis = 0, \
              return_index = True)
cluster_members = \
    next_cluster_members[np.sort(cluster_indices)]

cluster_colors = \
    np.zeros((np.shape(cluster_indices)[0], 3))
    
for i in range(np.shape(cluster_indices)[0]):

    cluster_colors[i, :] = \
        np.random.randint(5, size = (1, 3)) / 5

figure3, ax3 = plt.subplots(nrows = 1, ncols = 1)

ax3.set_title('Clusters @ 30-min discontinuous\ncontact duration')

map = plt.imread('python_map.jpg')
ax3.imshow(map, \
           extent = [np.amin(modified_data[:, 5]), \
                     np.amax(modified_data[:, 5]), \
                         np.amin(modified_data[:, 4]), \
                             np.amax(modified_data[:, 4])])
    
ax3.scatter(synthetic_lons[:, t], synthetic_lats[:, t], \
            color = 'yellow', marker = 'o', edgecolors = 'black')

for i in range(1, np.shape(cluster_indices)[0]):

    ax3.scatter(synthetic_lons[ \
                               cluster_members[0, i], t], \
                synthetic_lats[ \
                               cluster_members[0, i], t], \
                    color = cluster_colors[i, :], marker = 'o', \
                        edgecolors = 'black')

ax3.set_xlim([np.amin(modified_data[:, 5]), np.amax(modified_data[:, 5])])
ax3.set_ylim([np.amin(modified_data[:, 4]), np.amax(modified_data[:, 4])])
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')

figure3.tight_layout()
figure3.savefig('clusters at 30.png', dpi = 500)

#%%

# Written by "Kais Suleiman"(ksuleiman.weebly.com)
