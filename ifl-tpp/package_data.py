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
#takes a dataset in the regular format (specified in the report) and converts it to something suitable for the intensity free model

#python package_data.py <client_id> <cluster_size> <chunk_length>
#client_id [int]: the dataset (client) to use
#cluster_Size [float]: the silence period before creating a new clusters in seconds
#chunk_length [int]: how long each chunk should be. This is related to how the intensity free model works. (should be 128)

#it produces 3 files on each run
#1. raw data
#2. cluster times
#3. innercluster times

import sys
import numpy as np
import pandas as pd
from HelperFunctions import Clusterise, InterTimes, ComputeClusterLengths, GenerateClusters, ConcatClusters

sec = 1000000000

#tweakable
dataFile = 'data.csv'
#other settings are used as parameters to this script






X = pd.read_csv(dataFile)
#X1 = X[(X.client_id == 1) & (X.from_ds == 0) & (X.time < 70000*sec)].time.apply(lambda x:x/sec)
#X2 = X[(X.client_id == 1) & (X.from_ds == 1) & (X.time < 70000*sec)].time.apply(lambda x:x/sec)
X.time = X.time.apply(lambda x:x/sec)

curr_id = 1
cluster_size = 45
#1 is the series number
#2 is the cluster size (seconds)
#3 is the chunk size (seconds silence)
if (len(sys.argv) == 4):
    try:
        curr_id = int(sys.argv[1])
    except ValueError:
        print("bad id")
        exit()
    try:
        cluster_size = float(sys.argv[2])
    except ValueError:
        print("bad cluster size")
        exit()
    try:
        chunkSize = int(sys.argv[3])
    except ValueError:
        print("bad chunk size")
        exit()
else:
    print("need args")
    exit()

#packaging time
currData = X[(X.client_id == curr_id)]
totlen = len(currData)
currData = currData.drop_duplicates('time')
newlen = len(currData)
print(f"Dropped {totlen - newlen} duplicates")
X1 = currData[(currData.from_ds == 0)].time
X2 = currData[(currData.from_ds == 1)].time
X1.sort_values
X2.sort_values

maxtime = X1.max()
m = X2.max()
if (m > maxtime):
    maxtime = m
print(f"Max Time: {maxtime} seconds")
print(f"Num Points: {len(X1) + len(X2)}")

timeOnlyData = [X1.astype(np.dtype('d')).to_numpy(), X2.astype(np.dtype('d')).to_numpy()]

#save the entire sequence (no clustering)
#but make sure to split it chunks
timeChunks = np.asarray([list(currData.time[x:x+chunkSize]) for x in range(0, int(len(currData.time)), chunkSize)])
markChunks = np.asarray([list(currData.from_ds[x:x+chunkSize]) for x in range(0, int(len(currData.from_ds)), chunkSize)])
np.savez(f"data/{curr_id}-raw", arrival_times=timeChunks, marks=markChunks)

l = list(currData.time)
for i in range(len(l)-1):
    if (l[i] > l[i+1]):
        print("unsorted")
    

#cluster the data now

#find interarrivals
#interarr = []
#for timestamps in timeOnlyData:
    #interarr.append(InterTimes(timestamps))

#perform the clustering
clusters = []
inner_cluster_times = []
clusters, inner_cluster_times = Clusterise(timeOnlyData, cluster_size)

#save the cluster start times as a sequence
#also split the timeline into segments for training
chunkSize = 16
timeChunks = np.asarray([list(clusters[x:x+chunkSize]) for x in range(0, int(len(clusters)), chunkSize)])
np.savez(f"data/{curr_id}-clusters", arrival_times=timeChunks)

#now to save the innercluster times as a sequence
#need to convert the inner clusters into individual sequences
#ie, merge the inner clusters and instead use marks to identify which list they came from
arrivalSequences = []
markSequences = []
num = 0
tot = 0
for cluster in inner_cluster_times:
    tot += 1
    l0 = zip(cluster[0], [0]*len(cluster[0]))
    l1 = zip(cluster[1], [1]*len(cluster[1]))

    lnew = []
    lnew.extend(l0)
    lnew.extend(l1)
    lnew.sort()
    times = []
    marks = []
    for i in lnew:
        times.append(i[0])
        marks.append(i[1])

    if (len(times) <= 1):
        num += 1
        continue

    #it seems to hate clusters with only 1 element in it.. so ignore them?


    #print(lnew[0:10])
    #print(times[0:10])
    #print(marks[0:10])
    arrivalSequences.append(times)
    markSequences.append(marks)
    #if (tot >= 1000): #checking why it only works with smaller amounts of data
        #break

np.savez(f"data/{curr_id}-innerclusters", arrival_times=np.asarray(arrivalSequences), marks=np.asarray(markSequences))
print(f"Ignored {num}/{tot} clusters due to being length=1")

#print(clusters[0:2])
#print(inner_cluster_times[0:2])

