# Wi-Fi Traffic
A collection of scripts and notebooks researching the applications of machine learning techniques to generate realistic low level network traffic. Created as part of a honours thesis project by Ricard Grace. The thesis paper can be found in: 'Thesis_WiFi_Honeypots-Ricard_Grace.pdf'.

The following codebase was created Python 3.7 (unless otherwise said) primarily as Jupyter notebooks (https://jupyter.org/install). Installation instructions for required packages can be found in the sections for each technology (Hawkes Processes, Intensity Free, Hidden Markov Model).

## Hawkes Processes
All work with Hawkes processes is done using the Tick library for python.
Installation instructions for Tick along with the associated research paper can be found at https://github.com/X-DataInitiative/tick. An api reference for Hawkes processes using tick can be found at https://x-datainitiative.github.io/tick/modules/hawkes.html. Some tick notebooks also make use of a 'goodness of fit script' available at https://github.com/majkee15/TICK-goodness-of-fit. The file 'tick_goodness_of_fit.py' should be added to the directory containing the tick notebooks.

### TickHelper.py
Contains various wrapper functions for perfoming Tick related tasks, such as training or simulation of a Hawkes process model.

### Hawkes EM Learner.ipynb
Uses an EM (parametric estimator) kernel to model the source distribution. This attempts to model the data directly, peforming no preprocessing.

### Hawkes Clustering - EM.ipynb
Allows for modelling a dataset as a Hawkes process with an EM (parametric estimator) kernel. The dataset is first broken down into clusters of events before the model is applied to it.

### Hawkes Clustering - Exp.ipynb
Allows for modelling a dataset as a Hawkes process with a exponential kernel. The dataset is first broken down into clusters of events before the model is applied to it.

### Hawkes Split Distribution.ipynb
Training and simulation of a Hawkes process model (EM kernel) on the RHS of the split distribution.

### Hawkes Split Distribution Multivariate.ipynb
Similar to 'Hawkes Split Distribution' except using multivariate kernels. The data is split into two categories (one for each variable in the process) based on the direction of travel (to or from the given client)

## Intensity Free
The intensity free modelling approach implementation and installation instructions can be found at https://github.com/shchur/ifl-tpp. Since some of the backend code has been modified from the originals found in the above repo, these have been provided in the folder 'ifl-tpp' and should replace their respective components post installation. Once installed, ensure the backend code location at the top of the 'Intensity Free.ipynb' notebook points to the code folder inside the installation (typically '/ifl-tpp/code'). Note: the code was originally written in python 3.6 but will still work in python 3.7.

### Intensity Free.ipynb
A modification of the original intensity free script to provide simulation of new data points and culling of input data points, in line with modelling the RHS of the split distribution

### ifl-tpp/package_data.py
A script used to format input data appropriate for use in the intensity free notebook. More info can be found (including how to run it) at the top of the script.

### /ifl-tpp/code/dpp/model.py
A small modification on the original model class to support prediction of new points.

### /ifl-tpp/code/dpp/data.py
A small modification on the original data loader provided with the intensity free package to only load points occuring after a specified time.

## Hidden Markov Model
Hidden Markov Modelling is based on HMMlearn. Installation instructions can be found at: https://github.com/hmmlearn/hmmlearn. An API reference for HMMlearn can be found at: https://hmmlearn.readthedocs.io/en/latest/

### HMM Gaussian Distributions.ipynb
This script is used to model and generate both the RHS distribution and the ordering distribution using a Hidden Markov Model. Switching between the two is a matter of tweaking parameters.

## Other Scripts/Notebooks/Files
Below is a collection of other notebooks/files that do not rely on any of the three technologies above. Two of these are distribution analysis notebooks, taking a distribution and generating statistics on it, a third being a notebook to create composite distributions, the fourth a library of useful functions and the last a sample of data to be used as input.

### HelperFunctions.py
This file contains a collection of useful functions used throughout the project, the most important of which is the graphing function.

### Distribution Analysis.ipynb
Used for analysing pretrained models. Given a client id (source dataset) and precomputed distribution (from a file in saved_dist), plot two comparison histograms on differing timescales. Allows for analysis of how well a trained distribution matches the source one. Can either use the given distribution as is or remove points to the LHS of the split and instead fit a lognormal for those points.

### Distribution Split Statisitcs.ipynb
Computes statistics on the nature of the distribution split and on how closely the LHS of the distribution resembles a lognormal or exponential distribution.

### Composite Distributions.ipynb
This script will combine two seperate distributions, using a third distribution to perform the ordering. Two of these distributions should already be defined (RHS and ordering) and saved in files to be loaded. The RHS distribution can come from any of the saved distributions and more can be generated using the three technologies above. The ordering currently can only be generated in 'HMM Gaussian Distributions.ipynb'.The LHS distribution is auto generated using the raw data as input.

### data.csv
A section of sample input data for all the scripts. The event sequence here does not contain packet sizes (all are 0) or the direction of packet travel (all are 0) and all the client ids are the same (since they originate from the same event sequence). The sample was generated from the composite distribution model.

## Folders
### ifl-tpp
Contains the modified implementation of the intensity free method along with some additional useful scripts.

### saved_dist
Contains sample simulated event sequences that all the notebooks save/load from (except intensity free). Any newly generated distributions will be placed here.

### saved_model
Contains sample models previously trained. These models can be loaded back up to simulate more event sequences. The code to do this is not directly provided but many of the notebooks can be modified easily to do this (most of the notebooks that save a model also come with methods to load a model).

# Copyright/Acknowledgments
Copyright © Cyber Security Research Centre Limited 2020. This work has been supported by the Cyber Security Research Centre (CSCRC) Limited whose activities are partially funded by the Australian Government’s Cooperative Research Centres Programme. We are currently tracking the impact CSCRC funded research. If you have used this code in your project, please contact us at contact@cybersecuritycrc.org.au to let us know.