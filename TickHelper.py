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
from tick.hawkes import SimuHawkesExpKernels, SimuHawkesMulti, HawkesExpKern, SimuHawkes, HawkesEM, HawkesKernelTimeFunc, HawkesBasisKernels
from tick.plot import plot_hawkes_kernels, plot_point_process
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import ConcatClusters, ComputeClusterLengths
from tick_goodness_of_fit import plot_resid, goodness_of_fit_par
from scipy import integrate
import math

#contains various functions used to train or simulate models related to hawkes processes
#some maybe broken, but the ones used in the notebooks work


#takes a list of inner cluster timestamps
#attemps to train an exp model to produce more clusters
def TrainInnerClusterExp(clusters, num_decays=2000, decay_low=-10, decay_high=10):
    data = ConcatClusters(clusters, 0)
    best_score = -1e100
    #decays for multiple dimention process
    #update this to have different decays for each process
    #num_decays = 2000
    #print(f"Total decay combinations = {num_decays*num_decays*num_decays*num_decays}")
    decay_candidates = np.logspace(decay_low, decay_high, num_decays, dtype=np.dtype('d'))
    print(f"Training on {len(clusters)} clusters")
    print(f"Decay Range: {decay_candidates[0]} -> {decay_candidates[-1]}")
    best_decay = decay_candidates[0]
    score_list = np.zeros(num_decays)

    #x*e^(-xt)
    l = 0
    floaterrors = 0
    baseline_errors = 0
    for i, decay in enumerate(decay_candidates):
        decay = decay * np.ones((2,2))
        try:
            #might need a hyperbolic kernel?
            #it seems to get too excited and decays too slowly
            #only small decay values seem to make sense
            learner = HawkesExpKern(decay, penalty='l2', C=1000, max_iter=1000, solver='agd', tol=1e-3)#, max_iter=1000, tol=1e-5) #gofit='likelihood'
            ###Error functions
            #l1 - has 0 step errors
            #l2 - runs, but the results do not look good, heavily favours higher decay values that produce nonsense graphs
            #elasticnet (elastic_net_ratio, def 0.95) - values closer to 0 work better (since it uses l2) otherwise it produces step errors. Still similar to l2.
            #nuclear - basically the same
            #none - how can you have no penalty function?
            ###solvers
            #agd - all penalties favour super high decays, basicaly wants random event generation
            #gd - basically the same
            #bfgs - does weird things, but is quick
            #svrg

            learner.fit(data, start=learner.coeffs)

            """cluster_num = 0
            for cluster in clusters:
                if (cluster_num % 100 == 0):
                    #print out training progress
                    s = f"It: {i}, Decay: {decay[0]}, Cluster: {cluster_num}"
                    print(f"\r{' '*l}\r", end='')
                    print(f"It: {i}, Decay: {decay[0]}, Cluster: {cluster_num}", end='', flush=True)
                    l = len(s)
                learner.fit(cluster, start=learner.coeffs)
                cluster_num += 1"""
            hawkes_score = learner.score()
            #print(hawkes_score)
            #print(f"Coeffs: {learner.coeffs}")

            #ensure there is a non-0 baseline
            numb = 0
            for b in learner.baseline:
                if (b > 0):
                    numb += 1
            if (numb == 0):
                baseline_errors += 1
                continue
            
            #record the score for plotting
            score_list[i] = hawkes_score

            #record the best
            if (hawkes_score > best_score):
                best_score = hawkes_score
                best_learner = learner
                best_decay = decay

            step = 0.01
            #residuals = goodness_of_fit_par(learner,data,step,integrate.simps)
            #plot_resid(residuals,2,1)



        except ZeroDivisionError:
            #print("float error");
            floaterrors += 1
            continue;
    
    #create a score plot
    plt.plot(decay_candidates, score_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('decay Scores')
    plt.grid(True)
    plt.show()

    print(f"\nTraining Done")
    print(f"Float Errors: {floaterrors} ({100/num_decays*floaterrors}%)")
    print(f"Baseline Errors: {baseline_errors} ({100/num_decays*baseline_errors}%)")
    print(f"==========\nSuccessful Results: {num_decays - floaterrors - baseline_errors} ({100/num_decays*(num_decays - floaterrors - baseline_errors)}%)\n==========\n")

    print(f"\nBest Score: {best_score}")
    print(f"Best Decay: {best_decay}")
    plot_hawkes_kernels(best_learner)

    print(f"Adjacency: {best_learner.adjacency}")
    print(f"Baseline: {best_learner.baseline}")
    print(f"Coeffs: {best_learner.coeffs}")

    #return best_learner.adjacency, best_learner.baseline, best_decay
    return best_learner, best_decay

def TrainSeriesExp(series, num_decays=2000, decay_low=-10, decay_high=10):
    best_score = -1e100
    #decays for multiple dimention process
    #update this to have different decays for each process
    #num_decays = 2000
    #print(f"Total decay combinations = {num_decays*num_decays*num_decays*num_decays}")
    decay_candidates = np.logspace(decay_low, decay_high, num_decays, dtype=np.dtype('d'))
    print(f"Decay Range: {decay_candidates[0]} -> {decay_candidates[-1]}")
    best_decay = decay_candidates[0]
    score_list = np.zeros(num_decays)

    #x*e^(-xt)
    l = 0
    floaterrors = 0
    baseline_errors = 0
    for i, decay in enumerate(decay_candidates):
        #decay = decay * np.ones((2,2))
        try:
            #might need a hyperbolic kernel?
            #it seems to get too excited and decays too slowly
            #only small decay values seem to make sense
            learner = HawkesExpKern(decay, penalty='l2', C=1e-3, max_iter=1000, solver='agd', tol=1e-5)#, max_iter=1000, tol=1e-5) #gofit='likelihood'
            ###Error functions
            #l1 - has 0 step errors
            #l2 - runs, but the results do not look good, heavily favours higher decay values that produce nonsence graphs
            #elasticnet (elastic_net_ratio, def 0.95) - values closer to 0 work better (since it uses l2) otherwise it produces step errors. Still similar to l2.
            #nuclear - basically the same
            #none - how can you have no penalty function?
            ###solvers
            #agd - all penalties favour super high decays, basicaly wants random event generation
            #gd - basically the same
            #bfgs - does weird things, but is quick
            #svrg

            learner.fit([series])

            """cluster_num = 0
            for cluster in clusters:
                if (cluster_num % 100 == 0):
                    #print out training progress
                    s = f"It: {i}, Decay: {decay[0]}, Cluster: {cluster_num}"
                    print(f"\r{' '*l}\r", end='')
                    print(f"It: {i}, Decay: {decay[0]}, Cluster: {cluster_num}", end='', flush=True)
                    l = len(s)
                learner.fit(cluster, start=learner.coeffs)
                cluster_num += 1"""
            hawkes_score = learner.score()
            #print(hawkes_score)
            #print(f"Coeffs: {learner.coeffs}")

            #ensure there is a non-0 baseline
            numb = 0
            for b in learner.baseline:
                if (b > 0):
                    numb += 1
            if (numb == 0):
                baseline_errors += 1
                continue
            
            #record the score for plotting
            score_list[i] = hawkes_score

            #record the best
            if (hawkes_score > best_score):
                best_score = hawkes_score
                best_learner = learner
                best_decay = decay
        except ZeroDivisionError:
            #print("float error");
            floaterrors += 1
            continue;
    
    #create a score plot
    plt.plot(decay_candidates, score_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('decay Scores')
    plt.grid(True)
    plt.show()

    print(f"\nTraining Done")
    print(f"Float Errors: {floaterrors} ({100/num_decays*floaterrors}%)")
    print(f"Baseline Errors: {baseline_errors} ({100/num_decays*baseline_errors}%)")
    print(f"==========\nSuccessful Results: {num_decays - floaterrors - baseline_errors} ({100/num_decays*(num_decays - floaterrors - baseline_errors)}%)\n==========\n")

    print(f"\nBest Score: {best_score}")
    print(f"Best Decay: {best_decay}")
    plot_hawkes_kernels(best_learner)

    print(f"Adjacency: {best_learner.adjacency}")
    print(f"Baseline: {best_learner.baseline}")
    print(f"Coeffs: {best_learner.coeffs}")

    #return best_learner.adjacency, best_learner.baseline, best_decay
    return best_learner, best_decay


#careful, setting the time to be too high will eat up all available memory
def SimulateExp(baseline, adjacency, decays, time):

    hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False)

    hawkes.end_time = time
    dt = 0.001 #millisecond granularity
    #hawkes.track_intensity(dt)
    print(f"Starting sim")
    hawkes.simulate()
    timestamps = hawkes.timestamps
    l = 0
    for series in timestamps:
        l += len(series)
    print(f"Simulated {l} points")
    return hawkes.timestamps

def TrainInnerClusterEM(clusters, k_time = 1, k_size = 100):
    #merge all the clusters, the learner seems to only be able to fit a single long datastream
    num_clusters = len(clusters)
    data = ConcatClusters(clusters, 0)

    #kernel size is the granularity
    #kernel support is something... (is it the size of each step?)
    em_learner = HawkesEM(kernel_support=k_time, kernel_size=k_size, n_threads=8, verbose=True, tol=1e-5, max_iter=1000)
    em_learner.fit(data)

    """#train the em learner on each cluster
    cluster_num = 0
    for cluster in clusters:
        if (cluster_num % 10 == 0):
            #print out training progress
            s = f"Cluster: {cluster_num}/{num_clusters}"
            print(f"\r{' '*l}\r", end='')
            print(f"Cluster: {cluster_num}/{num_clusters}", end='', flush=True)
            l = len(s)
        print(em_learner.baseline)
        print(em_learner.kernel)
        print("==========")
        if (cluster_num == 0):
            em_learner.fit(cluster)
        else:
            em_learner.fit(cluster, baseline_start=em_learner.baseline, kernel_start=em_learner.kernel)
        cluster_num += 1"""
    #maybe add variation in kernel sie later?
    #use em_learner.score() to evaluate goodness
    print(f"\nEM Score: {em_learner.score()}")
    fig = plot_hawkes_kernels(em_learner) #TODO, remove this?

    t = np.linspace(0, k_time, endpoint=False, num=k_size)

    m = []
    for i in range(2):
        for j in range(2):
            m.append(max(em_learner.kernel[i][j]))
    #normalise to make a proper hawkes process
    spectral_radius = max(m)
    if (spectral_radius < 1):
        spectral_radius = 1
            
    #create a 2x2 array of time func kernels
    k = [[],[]]
    for i in range(2):
        for j in range(2):
            k[i].append(HawkesKernelTimeFunc(t_values=t, y_values=em_learner.kernel[i][j]/np.linalg.norm(em_learner.kernel[i][j])))

    #return k, em_learner.baseline #the kernel, baseline
    return em_learner

def SimulateEM(kernel, baseline, time=600):
    sim_em = SimuHawkes(kernels=kernel, baseline=baseline, verbose=False, end_time=time)

    dt = 0.001 #millisecond granularity
    #sim_em.track_intensity(dt)
    sim_em.simulate()

    timestamps = sim_em.timestamps
    l = 0
    for series in timestamps:
        l += len(series)
    print(f"Simulated {l} points")
    return sim_em.timestamps

def TrainInnerClusterBasis(clusters, k_time=1, k_size=100, num_kernels=2):
    num_clusters = len(clusters)
    #data = ConcatClusters(clusters, 0)
    l = 0

    basis_learner = HawkesBasisKernels(kernel_support=k_time, kernel_size=k_size, n_basis=num_kernels, C=1e-3, n_threads=8, verbose=False, ode_tol=1e-5, max_iter=1000)
    #train the basis learner on each cluster
    cluster_num = 0
    for cluster in clusters:
        if (cluster_num % 10 == 0):
            #print out training progress
            s = f"Cluster: {cluster_num}/{num_clusters}"
            print(f"\r{' '*l}\r", end='')
            print(f"Cluster: {cluster_num}/{num_clusters}", end='', flush=True)
            l = len(s)
        if (cluster_num == 0):
            basis_learner.fit(cluster)
        else:
            basis_learner.fit(cluster, baseline_start=basis_learner.baseline, amplitudes_start=basis_learner.amplitudes, basis_kernels_start=basis_learner.basis_kernels)
        cluster_num += 1


    #kernel size is the granularity
    #kernel support is something... (is it the size of each step?)
    #basis_learner = HawkesBasisKernels(kernel_support=k_time, kernel_size=k_size, n_basis=num_kernels, C=1e-3, n_threads=8, verbose=True, ode_tol=1e-5, max_iter=1000)
    #basis_learner.fit(data)
    #maybe add variation in kernel sie later?
    #use em_learner.score() to evaluate goodness
    #print(f"\nEM Score: {basis_learner.score()}")

    #TODO, remove this?
    fig = plot_hawkes_kernels(basis_learner)

    print(basis_learner.basis_kernels)
    print(basis_learner.amplitudes)
    print(basis_learner.baseline)
    return None

def SimulateBasis(kernel, baseline, time=600):
    sim_em = SimuHawkes(kernels=kernel, baseline=baseline, verbose=False, end_time=time)

    dt = 0.001 #millisecond granularity
    sim_em.track_intensity(dt)
    sim_em.simulate()

    timestamps = sim_em.timestamps
    l = 0
    for series in timestamps:
        l += len(series)
    print(f"Simulated {l} points")
    return sim_em.timestamps


#takes clusters of timestamps
#trains using clusters instead of concatenating them
def TrainInnerTimestampsExp(clusters, num_decays=2000, decay_low=-10, decay_high=10, max_iterations=100, tolerance=1e-5, e=10):
    cat_clusters = ConcatClusters(clusters, 0)
    best_score = -1e100
    print(f"Training on {len(clusters)} clusters")
    unique_decays = int(num_decays**(1.0/4))
    num_decays = unique_decays**4
    decay_candidates = np.logspace(decay_low, decay_high, unique_decays, dtype=np.dtype('d'))
    print(f"Decay Range: {decay_candidates[0]} -> {decay_candidates[-1]}")
    print(f"{unique_decays} unique decays. {num_decays} total")
    best_decay = None
    score_list = np.zeros(num_decays)

    #x*e^(-xt)
    l = 0
    floaterrors = 0
    baseline_errors = 0
    for i in range(num_decays):
        decay = np.ones((2,2))
        decay[0][0] = decay_candidates[int(i/(unique_decays**3))%unique_decays]
        decay[0][1] = decay_candidates[int(i/(unique_decays**2))%unique_decays]
        decay[1][0] = decay_candidates[int(i/(unique_decays**1))%unique_decays]
        decay[1][1] = decay_candidates[int(i)%unique_decays]
        prev_score = float('-inf')
        #print(decay)
        try:
            learner = HawkesExpKern(decay, penalty='l2', C=e, max_iter=1, solver='agd', tol=1e-5)

            #do the learning loop
            #need a stopping point
            for i in range(max_iterations):
                for cluster in clusters:
                    learner.fit(cluster, start=learner.coeffs)
                it_score = learner.score()
                print(f"It: {i}, Score: {it_score}")
                if (it_score <= prev_score + tolerance):
                    #barely changed
                    break
                prev_score = it_score
            hawkes_score = learner.score(events=clusters)

            #ensure there is a non-0 baseline
            numb = 0
            for b in learner.baseline:
                if (b > 0):
                    numb += 1
            if (numb == 0):
                baseline_errors += 1
                continue
            
            #record the score for plotting
            score_list[i] = hawkes_score

            #record the best
            if (hawkes_score > best_score):
                best_score = hawkes_score
                best_learner = learner
                best_decay = decay

        except ZeroDivisionError:
            #print("float error");
            floaterrors += 1
            continue;
    
    #create a score plot
    plt.plot(score_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('decay Scores')
    plt.grid(True)
    plt.show()

    print(f"\nTraining Done")
    print(f"Float Errors: {floaterrors} ({100/num_decays*floaterrors}%)")
    print(f"Baseline Errors: {baseline_errors} ({100/num_decays*baseline_errors}%)")
    print(f"==========\nSuccessful Results: {num_decays - floaterrors - baseline_errors} ({100/num_decays*(num_decays - floaterrors - baseline_errors)}%)\n==========\n")

    print(f"\nBest Score: {best_score}")
    print(f"Best Decay: {best_decay}")
    plot_hawkes_kernels(best_learner)

    print(f"Adjacency: {best_learner.adjacency}")
    print(f"Baseline: {best_learner.baseline}")
    print(f"Coeffs: {best_learner.coeffs}")

    #activate this for residuals (Warning, it is REALLLLLLLLLLY SLOOOOOOOOOOOOW)
    step = 0.1
    residuals = goodness_of_fit_par(best_learner,cat_clusters,step,integrate.simps)
    plot_resid(residuals,2,1)

    return best_learner.adjacency, best_learner.baseline, best_decay


#"you just put the clusters into it and it does it (tm)"
#takes clusters of timestamps
#trains using clusters instead of concatenating them
def TrainInnerTimestampsExp2(clusters, num_decays=2000, decay_low=-10, decay_high=10, e=10):
    best_score = -1e100
    print(f"Training on {len(clusters)} clusters")
    unique_decays = int(num_decays**(1.0/4))
    num_decays = unique_decays**4
    decay_candidates = np.logspace(decay_low, decay_high, unique_decays, dtype=np.dtype('d'))
    print(f"Decay Range: {decay_candidates[0]} -> {decay_candidates[-1]}")
    print(f"{unique_decays} unique decays. {num_decays} total")
    best_decay = None
    score_list = np.zeros(num_decays)

    #x*e^(-xt)
    l = 0
    floaterrors = 0
    baseline_errors = 0
    for i in range(num_decays):
        s = f"Decay {i} ({format(100/num_decays*i, '.2f')}% done)"
        l = len(s)
        #print(f"{' '*l}\r", end="", flush=True)
        print(f"{' '*l}\r{s}\r", end='', flush=True)
        decay = np.ones((2,2))
        decay[0][0] = decay_candidates[int(i/(unique_decays**3))%unique_decays]
        decay[0][1] = decay_candidates[int(i/(unique_decays**2))%unique_decays]
        decay[1][0] = decay_candidates[int(i/(unique_decays**1))%unique_decays]
        decay[1][1] = decay_candidates[int(i)%unique_decays]
        prev_score = float('-inf')
        #print(decay)
        try:
            learner = HawkesExpKern(decay, penalty='l2', C=e, max_iter=1000, solver='agd', tol=1e-5)
            learner.fit(clusters)
            hawkes_score = learner.score()

            #ensure there is a non-0 baseline
            numb = 0
            for b in learner.baseline:
                if (b > 0):
                    numb += 1
            if (numb == 0):
                baseline_errors += 1
                continue
            
            #record the score for plotting
            score_list[i] = hawkes_score

            #record the best
            if (hawkes_score > best_score):
                best_score = hawkes_score
                best_learner = learner
                best_decay = decay

        except ZeroDivisionError:
            #print("float error");
            floaterrors += 1
            continue;
    
    #create a score plot
    plt.plot(score_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('decay Scores')
    plt.grid(True)
    plt.show()

    print(f"\nTraining Done")
    print(f"Float Errors: {floaterrors} ({100/num_decays*floaterrors}%)")
    print(f"Baseline Errors: {baseline_errors} ({100/num_decays*baseline_errors}%)")
    print(f"==========\nSuccessful Results: {num_decays - floaterrors - baseline_errors} ({100/num_decays*(num_decays - floaterrors - baseline_errors)}%)\n==========\n")

    print(f"\nBest Score: {best_score}")
    print(f"Best Decay: {best_decay}")
    plot_hawkes_kernels(best_learner)

    print(f"Adjacency: {best_learner.adjacency}")
    print(f"Baseline: {best_learner.baseline}")
    print(f"Coeffs: {best_learner.coeffs}")

    #activate this for residuals (Warning, it is REALLLLLLLLLLY SLOOOOOOOOOOOOW)
    cat_clusters = ConcatClusters(clusters, 0)
    step = 0.1
    residuals = goodness_of_fit_par(best_learner,cat_clusters,step,integrate.simps)
    plot_resid(residuals,2,1)

    return best_learner.adjacency, best_learner.baseline, best_decay


def SimExp(baseline, adjacency, decays, num_clusters, data):
    hawkes = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baseline, verbose=False)

    #dt = 0.001 #millisecond granularity
    #hawkes.track_intensity(dt) # turning this on will eat up memory

    #need to compute and draw from the cluster length distrbution from the original data
    cluster_lengths = ComputeClusterLengths(data)

    multi = SimuHawkesMulti(hawkes, n_simulations=num_clusters)

    multi.end_time = np.random.choice(cluster_lengths, size=num_clusters, replace=True)
    multi.simulate()
    sim_inner_timestamps = multi.timestamps

    l = 0
    for realisation in sim_inner_timestamps:
        for series in realisation:
            l += len(series)
    print(f"Simulated {l} points")
    return sim_inner_timestamps

#k_time is the timescale on which a model should be formed
#k_size is how many points it should consider on this timescale
def TrainEM(times, k_time = 1, k_size = 100):

    #kernel size is the granularity
    #kernel support is the length of time to support
    em_learner = HawkesEM(kernel_support=k_time, kernel_size=k_size, n_threads=4, verbose=True, tol=1e-5, max_iter=250)
    em_learner.fit(times)

    #fig = plot_hawkes_kernels(em_learner) #TODO, remove this?

    return em_learner

def SimEM(smodel, time=600):
    sim_em = SimuHawkes(kernels=smodel.time_kernel, baseline=smodel.baseline, verbose=False, end_time=time)
    sim_em.simulate()
    return sim_em.timestamps

class SavedModel:

    def __init__(self, model, k_time, k_size, is_em=True, decay=0):
        self.baseline = model.baseline
        self.k_time = k_time
        self.k_size = k_size
        self.time_kernel = None
        self.intensity = 0
        self.decay = decay
        self.is_em = is_em
        if (is_em):
            self.kernel = model.kernel
            self.MakeTimeKernel(self.kernel)
            self.n_realizations = model.n_realizations
            self.n_nodes = model.n_nodes
        else:
            self.intensity = model.adjacency
            self.coeffs = model.coeffs

    def MakeTimeKernel(self, kernel):
        #check the spectral radius
        m = []
        for process in kernel:
            for intensity in process:
                m.append(max(intensity))
        self.spectral_radius = max(m)
        if (self.spectral_radius > 1):
            #oh no
            self.time_kernel = None
            return

        #create a time kernel for simulation
        t = np.linspace(0, self.k_time, endpoint=False, num=self.k_size)
        k = []
        for process in kernel:
            for intensity in process:
                k.append(HawkesKernelTimeFunc(t_values=t, y_values=intensity))

        k = np.reshape(k, kernel.shape[:2])
        self.time_kernel = k

    def Save(self, fname):
        np.savez(fname, savedmodel=self)