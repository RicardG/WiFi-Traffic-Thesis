"""
BSD 3-Clause License

Copyright (c) 2020, Cyber Security Research Centre Limited
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

#some things in here are useful (especially multihist)
#others are relics of before numpy was used effectively

#data is a list of two arrays of timestamps
#timestamps is probably a 1d pandas dataframe or np array or just a list
#will return a list of timestamps of cluster starts and a list of arrays containing times within clusters
#the cluster ends when a new point has not been seen for cluster_size time
#assumes timestamps are positive
def ClusteriseSingle(timestamps, cluster_size):
    cluster_timestamps = [] #start of each cluster
    inner_cluster_timestamps = [] #arrays of timestamps of events within each cluster
    prev_cluster_time = -1
    curr_cluster_timestamps = [] #timestamps in the current cluster
    for time in timestamps:
        if (time > prev_cluster_time + cluster_size):
            #a new cluster
            cluster_timestamps.append(time)
            inner_cluster_timestamps.append(np.array(curr_cluster_timestamps, dtype=np.dtype('d')))
            curr_cluster_timestamps = [time]
        else:
            curr_cluster_timestamps.append(time)
        prev_cluster_time = time
    #ensure the final clusters inner timestamps are added
    if (curr_cluster_timestamps):
        inner_cluster_timestamps.append(np.array(curr_cluster_timestamps, dtype=np.dtype('d')))

    #ensure the return value is a np array for efficiency reasons
    return np.array(cluster_timestamps, dtype=np.dtype('d')), inner_cluster_timestamps

#takes a list of arrays (data)
#returns a list of cluster start times and a list of timestamps for events within each cluster for each cluster start
def Clusterise(data, cluster_size):
    num_series = len(data)
    cluster_starts = [] #the start of a cluster of events
    cluster_timestamps = [] #[cluster no, series no]
    #create iterators for each series
    iterators = []
    next_event = []
    for series in data:
        it = iter(series)
        iterators.append(it)
        try:
            next_event.append(next(it))
        except StopIteration:
            next_event.append(None)
    
    cluster_prev = float('-inf')
    #dont use [[]]*x it just duplicates the pointer to the list
    curr_cluster_timestamps = [[] for i in range(num_series)] #one list for each series
    while (True):
        curr_min, pos = SafeMin(next_event)
        if (curr_min is None):
            #no more points
            #save the current cluster
            c = []
            for cl in curr_cluster_timestamps:
                c.append(np.array(cl, dtype=np.dtype('d'))-cluster_starts[-1])
            cluster_timestamps.append(c)
            break

        if (curr_min > cluster_prev + cluster_size):
            #save current cluster, convert to np array
            if (cluster_prev >= 0):
                c = []
                for cl in curr_cluster_timestamps:
                    c.append(np.array(cl, dtype=np.dtype('d'))-cluster_starts[-1])
                cluster_timestamps.append(c)
            #create new cluster
            curr_cluster_timestamps = [[] for i in range(num_series)]
            cluster_starts.append(curr_min)
            #record the new event in the cluster
            curr_cluster_timestamps[pos].append(curr_min)
        else:
            #record the next event
            curr_cluster_timestamps[pos].append(curr_min)

        #prepare for the next event
        cluster_prev = curr_min
        next_event[pos] = SafeNext(iterators[pos])

    return np.array(cluster_starts, dtype=np.dtype('d')), cluster_timestamps


#min but with none values
#will return None if all inputs are none
#will also return the position of the min value
def SafeMin(l):
    m = None
    pos = None
    p = 0
    for i in l:
        if (i is not None):
            if (m is None or i < m):
                m = i
                pos = p
        p += 1
    return m, pos

def SafeNext(it):
    try:
        return next(it)
    except StopIteration:
        return None

def MultiHist(data, title="<Insert Title>", subtitles=[], bins=250, data_range=None, y_max=10):
    l = len(data)
    fig, axs = plt.subplots(l, figsize=(20,10), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=22)
    fig.tight_layout(pad=5.0)
    m = float('-inf')
    for series in data:
        s_max = max(series)
        if (s_max > m):
            m = s_max
    if (data_range is None):
        data_range = (0,m)

    for i in range(l):
        axs[i].set_ylabel('Freq (Log x)', fontsize=16)
        axs[i].set_xlabel('Time (Sec)', fontsize=16)
        axs[i].set_title(subtitles[i], fontsize=18)
        axs[i].set_xlim([data_range[0],data_range[1]])
        axs[i].set_ylim([0.1,y_max])
        _ = axs[i].hist(data[i], bins=bins, log=True, range=data_range)
    #fig.show()

#compute and return an array of interarrival times
def InterTimes(timestamps):
    t = []
    for i in range(len(timestamps)-1):
        t.append(timestamps[i+1] - timestamps[i])
    t.sort()
    return np.array(t, dtype=np.dtype('d'))

def ComputeClusterLengths(inner_cluster_times):
    cluster_lengths = []
    for cluster in inner_cluster_times:
        min = float("inf")
        max = 0
        for series in cluster:
            if (series.size > 0): #some series could be empty, but not all
                if (series[0] < min):
                    min = series[0]
                if (series[-1] > max):
                    max = series[-1]
        cluster_lengths.append(max-min)
    a = np.array(cluster_lengths, dtype=np.dtype('d'))
    #a.sort()
    return a

#timestamps is an array of timestamp arrays (one for each process)
#distribution is an array of cluster lengths to draw from
#cluster_size is the wait period for the end of a cluster, will be used as spacing for extracting samples
def GenerateClusters(timestamps, distribution, cluster_size):
    num_lengths = len(distribution)
    data = []
    for series in timestamps:
        data.append(pd.Series(series))

    #find min and max 
    low = []
    high = []
    for series in timestamps:
        low.append(series.min())
        high.append(series.max())

    #trim the first and last cluster_size seconds of data
    low = min(low) + cluster_size
    high = max(high) - cluster_size

    #generate the clusters
    curr_time = low
    new_clusters = []

    #TODO need to normalise the points
    while (True):
        #generate a new cluster length
        l = distribution[int(random.random()*num_lengths)]
        if (curr_time + l > high):
            #cant sample if there isnt enough data to sample
            break

        cluster = []
        is_empty = True
        for series in data:
            s = series[(series >= curr_time) & (series < curr_time + l)]
            cluster.append(np.array(s, dtype=np.dtype('d')))
            if (not s.empty):
                is_empty = False
        if (not is_empty):
            new_clusters.append(cluster)
        curr_time += l + cluster_size
    
    #the result is a list of lists containing arrays
    return new_clusters

def ConcatClusters(clusters, spacing):
    num_clusters = len(clusters)
    num_series = len(clusters[0])
    c_lengths = ComputeClusterLengths(clusters)
    curr_time = 0

    t_data = [[] for i in range(num_series)]

    #shift the cluster times so they are continuous
    for i, cluster in enumerate(clusters):
        for j, series in enumerate(cluster):
            cluster[j] = cluster[j]+curr_time
            t_data[j].append(cluster[j])
        curr_time += c_lengths[i] + spacing
    
    data = []
    for cluster_series in t_data:
        data.append(np.concatenate(cluster_series))
    
    return data

def ExpPoints(amplitude=10, decay=0.03, stop = 100, dt = 0.001):
    if (dt <= 0):
        return np.array([])
    i = dt
    point_list = []
    while (i < stop):
        point_list.append(amplitude * np.log(i**decay+1))
        i += dt

    return np.array(point_list, dtype=np.dtype('d'))

def ExpPoints2(time_scale=10, flatness=0.03, stop=100, points=100):
    gen_p = []
    i = 0
    while (i < points):
        p = time_scale * np.random.exponential(scale=flatness, size=None)
        if (p > stop):
            continue
        gen_p.append(p)
        i += 1
    
    return np.array(gen_p, dtype=np.dtype('d'))