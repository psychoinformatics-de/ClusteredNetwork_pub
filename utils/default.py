import numpy as np
# default values for parameters

# number of units
N_E = 4000
N_I = 1000

# connection probabilities
ps = np.array([[0.2, 0.5], [0.5, 0.5]])

# connections strengths
# weights are js/sqrt(N)
# nan means they are calculated
js = np.ones((2, 2)) * np.nan
# factors for inhibitory weights
ge = 1.2
gi = 1.
gie = 1.


# cluster number
Q = 1
# cluster weight ratios
jplus = np.ones((2, 2))

# synaptic delay
delay = 0.1

# factor multiplied with weights
s = 1.
fixed_indegree = False
# neuron parameters
neuron_type = 'iaf_psc_exp'
E_L = 0.
C_m = 1.
tau_E = 20.
tau_I = 10.
t_ref = 5.
V_th_E = 15. # 
V_th_I = 15. #
V_r = 0.
I_xE = 2. #2. # 5.
I_xI = 1. #1. #2.
delta_I_xE = 0.
delta_I_xI = 0.
I_th_E = 2.13 # 5.34  #
I_th_I = 1.24 # 2.61 #
V_m = 'rand'

tau_syn_ex = 3. #3. # 1. # 2. --> 2.9 
tau_syn_in = 2. #1. # 2. # 1. --> 1.9


n_jobs = 1
