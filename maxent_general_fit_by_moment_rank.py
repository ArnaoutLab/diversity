import numpy as np
import pandas as pd
import seaborn as sns
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter, Namespace

from datetime import datetime
from IPython.display import display, HTML
from itertools import combinations, combinations_with_replacement
from matplotlib import pyplot as plt
#from mpi4py import MPI
#from mpi4py.MPI import ANY_SOURCE
from numpy.random import default_rng
rng = default_rng()  # replaces calls to numpy.random
from os.path import expanduser
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu as mwu, ttest_ind
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from scipy.stats import mvn
import warnings

np.set_printoptions(formatter={'float': lambda x: f"{x:>7.3f}"})

#comm = MPI.COMM_WORLD # represents all processes available at start-up time
#rank = comm.Get_rank() # Process number
#no_processors = comm.Get_size() # Number of processes

parser = ArgumentParser(description = "Train maximum entropy models of...", formatter_class=RawDescriptionHelpFormatter)
pa = parser.add_argument

pa("-L", default=2, type=int, help="dimension of the data")
pa("-n", default=200, type=int, help="number of points of each of the two clusters of the target")
pa("-max_moment_rank", default=10, type=int, help="maximum order of the moments")
pa("-burn_in_iterations", default=1000, type=int, help="number of iterations in the burn-in")
pa("-sample_every", default=1000, type=int, help="number of iterations in the mcmc")
pa("-sample_size", default=50, type=int, help="sample size")
pa("-tranches", default=10, type=int, help="number of tranches")
pa("-temperature", default=5, type=float, help="initial temperature")
pa("-starting_temperature", default=5, type=float, help="initial temperature")
pa("-iterations_to_cool_over", default=200, type=int, help="number of iterations to cool down to temperature 1")
pa("-mu", default=0.5, type=float, help="momentum constant")
pa("-beta", default=0.5, type=float, help="proposal stepsize in pCN algorithm")
pa("-parallelize", default=False, type=bool, help="whether or not to parallelize the sampling function")
pa("-learning_rate", default=1, type=float, help="initial learning rate")

args = parser.parse_args()
globals().update(vars(args))

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------------------
#Functions

def propose(S, L, scale=1., verbose=False):

    T = np.copy(S)                            # so we don't change in place
    i = rng.choice(L)                         # choose an index of S (now called T) to change
    s_0 = T[0][i]
    ii = 0
    if verbose:
        print(f"S:\t{S}")    
        print(f"{s_0:.3f}")
    while True:                               # caution: potentially infinite loop!
        sign = rng.choice((-1,1))                 # choose whether to increase or decrease
        stepsize = rng.exponential(scale)         # get the stepsize according to an exponential
        ii += 1
        step = sign*stepsize
        s_1 = s_0 + step
        if verbose:
            print(f"{ii:>10}\t{s_1:.3f}")
        if 0. <= s_1 <= 1.:
            break
    #
    # calculate q_ratio (see photo q_ratio_041022.png)
    numerator   = 1. - np.exp( -scale*(1.-s_0) )/2. - np.exp( -scale*s_0 )/2.
    denominator = 1. - np.exp( -scale*(1.-s_1) )/2. - np.exp( -scale*s_1 )/2.
    q_ratio = numerator / denominator
    #
    T[0][i] = s_1                             # update S (now called T)
    if verbose:
        print(f"S:\t{S} ->")
        print(f"T:\t{T}")
    return T, q_ratio

def propose_pCN(x, L, C, beta, lower_bound=0., upper_bound=1.):
	r = np.sqrt(1-beta**2)
	y = np.ones(L)+-np.inf                         # initialize (for enforcing boundary condition)
	x = x.flatten()         # NEW!!!
	while (y < lower_bound).any() or (y > upper_bound).any():     # this loop will be finite
		y = rng.multivariate_normal(r*x, (beta**2)*C)   # NEW!!!
	y = np.array(y)
	y = y.reshape(1, -1)   # NEW!!!
	return y


def get_energy(s, biases):
    return -np.sum(s*biases)

def get_temperature(epoch):
	cooling_rate = starting_temperature/iterations_to_cool_over # units: degree K/iteration
	temperature = max(1., starting_temperature - cooling_rate*epoch)
	return temperature


def accept(current_energy, proposed_energy, q_ratio=1., temperature=1):
    """
    This is the Metropolis step (see Wikipedia, Metropolis-Hastings algorithm)
    """
    acceptance_criterion = min(1, q_ratio * np.exp((current_energy-proposed_energy)/temperature))
    if rng.random() < acceptance_criterion: 
        return True
    else: 
        return False

def accept_pCN(x, y, H_x, H_y, C, L, beta, temperature=1):
	r = np.sqrt(1-beta**2)
	x = x.reshape(L,)
	y = y.reshape(L,)
	numerator = mvn.mvnun(np.zeros(L), np.ones(L), r*x, C)[0]
	denominator = mvn.mvnun(np.zeros(L), np.ones(L), r*y, C)[0]
	q_ratio = numerator / denominator
	acceptance_criterion = q_ratio * np.exp((H_x-H_y)/temperature)
	rnd = rng.random()
	accept_ = rnd < acceptance_criterion         # acceptance = 'True' or 'False'
	return accept_

def mcmc(current, n, biases, calculate_features,fold=1,max_moment_rank=3, temperature=1):            # mcmc = markov-chain Monte Carlo
    """
    S = current state (a vector)
    n = number of iterations to run
    Tested and works.
    """
    for i in range(n):
        L = current.shape[1]
        proposed, q_ratio = propose(current, L)
        current_energy  = get_energy(calculate_features(current, max_moment_rank, fold), biases)
        proposed_energy = get_energy(calculate_features(proposed, max_moment_rank, fold), biases)
        if accept( current_energy, proposed_energy, q_ratio, temperature):
            current = np.copy(proposed)
    return(current)

def mcmc_pCN(current, n, C, beta, biases, calculate_features, fold=1,max_moment_rank=3, temperature=1):            # mcmc = markov-chain Monte Carlo
    """
    S = current state (a vector)
    n = number of iterations to run
    Tested and works.
    """
    for i in range(n):
        L = current.shape[1]
        proposed = propose_pCN(current, L, C, beta)
        current_energy  = get_energy(calculate_features(current, max_moment_rank, fold), biases)
        proposed_energy = get_energy(calculate_features(proposed, max_moment_rank, fold), biases)
        if accept_pCN(current, proposed, current_energy, proposed_energy, C, L, beta, temperature):
            current = np.copy(proposed)
    return(current)

def calculate_features(sample, max_moment_rank,fold=1):
    """
    TO DO
    
    This function is specific for adding the following type of features:
    - sums of each pair of indices within a datapoint
    - distance between the two datapoints
    
    - all non-self pairwise products
    - if order = 2, it's all pairwise products; order = 3 means triples, etc. Recommend not going past 2
    - feature_values_matrix is the *entire sample* appended with values of the pairwise (etc.) features. So each row is a state, and each column is a feature. The first L columns are the given features. The remaining L*(L-1)/2 columns are feature 1 * feature 2, feature 1 * feature 3, etc., in the natural order provided by combinations().
    Tested and works.
    """
    feature_values_matrix = np.copy(sample)
    for moment_rank in range(2, max_moment_rank + 1):
	    for column_tuple in combinations_with_replacement(range(len(sample[0])), moment_rank):
	    	feature_column = sample[:, column_tuple[0]]
	    	for k in range(1, len(column_tuple)):
	    		feature_column = feature_column*sample[:, column_tuple[k]]
	    	feature_column = feature_column.reshape(-1,1)
	    	feature_values_matrix = np.hstack((feature_values_matrix, feature_column))

    return(feature_values_matrix)


def make_seed(L):
    seed = rng.random(size=L)       # initialize
    seed = seed.reshape(1, -1)      # for ease of future concatenation
    return(seed)

def gradient_descent(biases, prev_deltas, rates, observed_expectation_values, sample_expectation_values, factor=1.05, verbose=False):
    deltas = observed_expectation_values - sample_expectation_values
    signs = np.sign(deltas / prev_deltas)  # compare signs to prev_deltas
    rates *= factor**signs
    biases += deltas * rates
    if verbose:
        print("\ttarget EVs:\t", observed_expectation_values)
        print("\tsample EVs:\t", sample_expectation_values)
        print("\tdeltas:    \t", deltas)
        print("\tprv deltas:\t", prev_deltas)
        print("\tsigns:     \t", signs)
        print("\trates:     \t", rates)
        print("\tbias updts:\t", (deltas * rates))
        print("\tnew biases:\t", biases)
    return biases, deltas, rates

def gradient_descent_NEW(biases, prev_deltas, prev_momenta, rates, observed_expectation_values, sample_expectation_values, factor=1.05, verbose=False):
    # See https://cs231n.github.io/neural-networks-3/#sgd 
    deltas = observed_expectation_values - sample_expectation_values
    signs = np.sign(deltas / prev_deltas)  # compare signs to prev_deltas
    rates *= factor**signs
    momenta = mu*prev_momenta + rates*deltas
    biases += momenta

    if verbose:
        print("\ttarget EVs:\t", observed_expectation_values)
        print("\tsample EVs:\t", sample_expectation_values)
        print("\tdeltas:    \t", deltas)
        print("\tprv deltas:\t", prev_deltas)
        print("\tsigns:     \t", signs)
        print("\t prv momenta:\t", prev_momenta)
        print("\trates:     \t", rates)
        print("\tnew biases:\t", biases)
    return biases, deltas, momenta, rates

def find_features_by_rank(chosen_moment_rank, highest_moment_rank, L):
	no_features_by_rank = []

	for running_rank in range(1, highest_moment_rank+1):
		no_features_by_rank.append(len(list(combinations_with_replacement(range(L), r=running_rank))))

	starting_feature_index = sum(no_features_by_rank[:chosen_moment_rank - 1])
	ending_feature_index = starting_feature_index + no_features_by_rank[chosen_moment_rank - 1] - 1

	return no_features_by_rank, starting_feature_index, ending_feature_index


def gradient_descent_by_moment_rank(moment_rank, biases, prev_deltas, prev_momenta, rates, observed_expectation_values, sample_expectation_values, factor=1.05, verbose=False):
	deltas = observed_expectation_values - sample_expectation_values
	signs = np.sign(deltas / prev_deltas)  # compare signs to prev_deltas
	rates *= factor**signs
	momenta = mu*prev_momenta + rates*deltas

	#Filtering out the momenta belonging to other moment ranks before we update the biases
	filtered_momenta = np.zeros(momenta.shape)
	no_features_by_rank, starting_feature_index, ending_feature_index = find_features_by_rank(moment_rank, max_moment_rank,L)
	for i in range(starting_feature_index, ending_feature_index+1):
		filtered_momenta[i] = momenta[i]
	biases += filtered_momenta
	
	if verbose:
		print("\tmoment rank:\t", moment_rank)
		print("\ttarget EVs:\t", observed_expectation_values)
		print("\tsample EVs:\t", sample_expectation_values)
		print("\tdeltas:    \t", deltas)
		print("\tprv deltas:\t", prev_deltas)
		print("\tsigns:     \t", signs)
		print("\trates:     \t", rates)
		print("\tprv momenta: \t", prev_momenta)
		print("\tfiltered momenta: \t", filtered_momenta)
		print("\tnew biases:\t", biases)
	return biases, deltas, momenta, rates


def sample_from_model_NEW(C, beta, biases, sample_size, L, tranches, burn_in_iterations, calculate_features, fold=1, max_moment_rank=3, temperature=1, parallelize=False):
    """
    L:                  the dimensionality of each state
    calculate_features: a generic name for the function that calculates_features; 
                        you can pass specific feature sets
    fold:               the number of datapoints to concatenate together; defines
                        how we think of a state of the system. E.g. if fold = 2,
                        and L = 2, each state consists of: (x1, y1, x2, y2): two
                        (x, y) datapoints concatenated together.
                        
                        *** NOT IMPLEMENTED YET; PROBABLY 1/2 DAY ***
    """

    #
    # sample (in several tranches)
    if parallelize:
    	def sample_per_tranche(tranche):
    		warnings.filterwarnings("ignore")
    		sample = np.zeros(fold*L)
	    	#seed = make_seed(L)
	    	seed = default_rng(seed=tranche**2).random(size=L).reshape(1,-1) 

	    	current = mcmc_pCN(seed, burn_in_iterations, C, beta, biases, calculate_features, fold, max_moment_rank, temperature)
	    	for i in range(int(sample_size/tranches)):
	    		current = mcmc_pCN(current, sample_every, C, beta, biases, calculate_features, fold, max_moment_rank, temperature)
	    		sample = np.vstack((sample, current)) #
	    	sample = sample[1:]

	    	return sample

    	collected_samples = Parallel(n_jobs=6)(delayed(sample_per_tranche)(tranche) for tranche in range(tranches))

    	sample=np.zeros(L)
    	for elt in collected_samples:
    		sample = np.vstack((sample, elt))
    	sample = sample[1:]

    	sample = np.array(sample).reshape(-1,L)


    else:
    	sample = np.zeros(fold*L)
    	for tranche in range(tranches):
    	    #
        	# get seed and burn in
        	seed = make_seed(L)
        	current = mcmc_pCN(seed, burn_in_iterations, C, beta, biases, calculate_features, fold, max_moment_rank, temperature)
        	for i in range(int(sample_size/tranches)):
        		current = mcmc_pCN(current, sample_every, C, beta, biases, calculate_features, fold, max_moment_rank, temperature)
        		sample = np.vstack((sample, current)) #
    	sample = sample[1:] # remove the initialized dummy value

    return sample

#For the legends of plots
def feature_names(L, max_moment_rank):
	feature_name_list = [chr((i + 23) % 26 + 97) for i in range(L)]
	for k in range(2, max_moment_rank + 1):
		for col_indices in combinations_with_replacement(range(L), r=k):
			col_indices = np.array(col_indices)
			feature_name_list.append(''.join([ chr((i + 23) % 26 + 97) for i in col_indices]))
	return feature_name_list

def make_2D_cluster(n, mu, std, cov):
    """
    cov = a scalar; covariance between the x and the y; cannot exceed the geometric mean of the variances
    std = an array-like; the standard deviations in the x and y directions
    tgt = target distribution (the cluster)
    C   = covariance matrix
    """
    mu = np.array(mu)
    C = np.array([[std[0]**2, cov], [cov, std[1]**2]])
    tgt = rng.multivariate_normal(mu, C, size=n)       # target distribution
    # respect boundary conditions by coercing to boundary (could also have deleted)
    tgt[tgt < 0.] = 0.
    tgt[tgt > 1.] = 1.
    return tgt

def make_tgt_2(tgt_n):
    cluster_1 = make_2D_cluster(int(tgt_n/2), [0.6, 0.6], [0.07,  0.07], -0.004)
    cluster_2 = make_2D_cluster(int(tgt_n/2), [0.2, 0.2], [0.02, 0.02], 0.)
    tgt = np.vstack((cluster_1, cluster_2))
    return tgt

    #--------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    outpath = f'maxent_outfiles_{str(datetime.now())}'
    os.mkdir(outpath)

    arguments_str = ''
    for arg_name, arg_value in vars(args).items():
    	arguments_str += f'{arg_name}: {arg_value}\n'
    with open(outpath + '/arguments.txt', 'w') as f:
    	f.write(arguments_str)

    # make up some highly clustered data
    data = make_tgt_2(n)

    # measure features (pairwise correlations)
    data = calculate_features(data, max_moment_rank=max_moment_rank)

    # get expectation values
    observed_expectation_values = np.mean(data, axis=0)
    n_features=len(observed_expectation_values)

    # inspect
    df = pd.DataFrame( data )
    display(df)

    fig = plt.figure(figsize=(3,3))
    plt.scatter(df[0], df[1], s=5, alpha=0.5)
    plt.xlabel("First parameter")
    plt.ylabel("Second parameter")
    fig.savefig(f'{outpath}/target.pdf')

    print("observed_expectation_values:", observed_expectation_values)

    biases = np.zeros(n_features)     # initialize at zero
    #biases = np.array([150.1, 140.2, -250.1, -200.3, -300.4, 500.5, 700.6, 500.7, 750.8, -100, -300.9, -800.9, -150, -300])
    deltas = np.ones(n_features)      # for first sign calculation in gradient descent
    momenta = np.ones(n_features)      # for first sign calculation in gradient descent
    rates  = np.ones(n_features)*learning_rate  # learning rates
    distance = 1.

    biases_vs_epoch = biases
    deltas_vs_epoch = deltas
    rates_vs_epoch = rates
    distance_vs_epoch = [distance]
    temperature_vs_epoch = [temperature]
    sample_size_vs_epoch = [sample_size]
    feature_names = feature_names(L,max_moment_rank)
    C = np.array([[1,0],[0,1]])

    epoch = 0

    for moment_rank in range(1, max_moment_rank+1):
       for i in range(25):
            epoch += 1
            temperature = get_temperature(epoch)
            sample_size *= 1.01

            sample = sample_from_model_NEW(C, beta, biases, sample_size, L, tranches, burn_in_iterations, calculate_features, 1, max_moment_rank, temperature, parallelize=True)
            sample_features = calculate_features(sample, max_moment_rank=max_moment_rank)
            sample_expectation_values = np.mean(sample_features, axis=0)
    	    #
    	    # update biases (and calculate a distance we can monitor)
            biases, deltas, momenta, rates = gradient_descent_by_moment_rank(moment_rank, biases, deltas, momenta, rates, 
    			observed_expectation_values, sample_expectation_values, factor=1.05, verbose=True)
            distance = cosine(sample_expectation_values, observed_expectation_values)
    	        #
    	        # adding the latest parameters to the lists of parameters vs epoch
            biases_vs_epoch = np.vstack((biases_vs_epoch, biases))
            deltas_vs_epoch = np.vstack((deltas_vs_epoch,deltas))
            rates_vs_epoch = np.vstack((rates_vs_epoch, rates))
            distance_vs_epoch.append(distance)
            temperature_vs_epoch.append(temperature)
            sample_size_vs_epoch.append(sample_size)

    	        # 
    	        # reporting
            print(f"\nEpoch {epoch}\t{datetime.now()}")
            print(f"\tdistance: {distance:.2e}")
    	        #
    	        # plots
            fig = plt.figure(figsize=(5,5))
            plt.scatter(sample[:, 0], sample[:, 1], s=2.5, alpha=0.25, c="orange")
            plt.xlabel("First parameter")
            plt.ylabel("Second parameter")
            plt.title(f'Currently fitting moment {moment_rank} \n sample size: {sample_size}')
            fig.savefig(f'{outpath}/epoch_{epoch}.pdf')
            plt.close()

            fig = plt.figure(figsize=(5,5))
            for i in range(n_features):
                plt.plot(range(epoch+1), biases_vs_epoch[:, i])
                plt.text(epoch,biases_vs_epoch[epoch,i],feature_names[i])
            plt.xlabel("Epoch")
            plt.ylabel("Biases")
    		#plt.legend()
            fig.savefig(f'{outpath}/biases_vs_epoch.pdf')
            plt.close()

            fig = plt.figure(figsize=(5,5))
            for i in range(n_features):
                plt.plot(range(epoch+1), deltas_vs_epoch[:, i])
                plt.text(epoch,deltas_vs_epoch[epoch,i],feature_names[i])
            plt.xlabel("Epoch")
            plt.ylabel("Deltas")
    		#plt.legend()
            fig.savefig(f'{outpath}/deltas_vs_epoch.pdf')
            plt.close()
            
            fig = plt.figure(figsize=(5,5))
            for i in range(n_features):
                plt.plot(range(epoch+1), rates_vs_epoch[:, i], label=feature_names[i])
                plt.text(epoch, rates_vs_epoch[epoch,i],feature_names[i])
            plt.xlabel("Epoch")
            plt.ylabel("Rates")
    		#plt.legend()
            fig.savefig(f'{outpath}/rates_vs_epoch.pdf')
            plt.close()

            fig = plt.figure(figsize=(5,5))
            plt.plot(range(epoch+1), distance_vs_epoch)
            plt.xlabel("Epoch")
            plt.ylabel("Distance")
            plt.yscale('log')
            fig.savefig(f'{outpath}/distance_vs_epoch.pdf')
            plt.close()

            fig = plt.figure(figsize=(5,5))
            plt.plot(range(epoch+1), temperature_vs_epoch)
            plt.xlabel("Epoch")
            plt.ylabel("Temperature")
            fig.savefig(f'{outpath}/temperature_vs_epoch.pdf')
            plt.close()

            fig = plt.figure(figsize=(5,5))
            plt.plot(range(epoch+1), sample_size_vs_epoch)
            plt.xlabel("Epoch")
            plt.ylabel("Sample size")
            fig.savefig(f'{outpath}/sample_size_vs_epoch.pdf')
            plt.close()

            np.savetxt(f'{outpath}/biases.csv', biases_vs_epoch)
