import faulthandler; faulthandler.enable()
import nest
import numpy as np
import matplotlib.pyplot as plt
import time
import default
import pylab
import pickle
import os
import signal
import sys
import copy


small = 1e-10


# ********************************************************************************
#                                    Classes
# ********************************************************************************
class TimeoutException(Exception):   # Custom exception class
    pass

# ********************************************************************************
#                                    Functions
# ********************************************************************************


def mergeParams(params):
    return params


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException




def max_PSP_exp(tau_m,tau_syn,C_m = 1.,E_l =0.):
    tmax = np.log(tau_syn/tau_m)/(1/tau_m-1/tau_syn)
    
    B = tau_m*tau_syn/C_m/(tau_syn-tau_m)
    return (E_l-B)*np.exp(-tmax/tau_m)+B*np.exp(-tmax/tau_syn)
    """calculates the maximum psp amplitude for exponential synapses and unit J"""


def calc_js(params):
    N_E = params.get('N_E', default.N_E)                          # excitatory units
    N_I = params.get('N_I', default.N_I)                          # inhibitory units
    N = N_E+N_I                                                  # total units
    ps = params.get('ps', default.ps) # connection probs
    ge = params.get('ge', default.ge)
    gi = params.get('gi', default.gi)
    gie = params.get('gie', default.gie)
    V_th_E = params.get('V_th_E', default.V_th_E)   # threshold voltage
    V_th_I = params.get('V_th_I', default.V_th_I)
    tau_E = params.get('tau_E', default.tau_E)
    tau_I = params.get('tau_I', default.tau_I)
    E_L = params.get('E_L', default.E_L)
    neuron_type = params.get('neuron_type', default.neuron_type)
    if 'iaf_psc_exp' in neuron_type :
        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        amp_EE = max_PSP_exp(tau_E,tau_syn_ex)
        amp_EI = max_PSP_exp(tau_E,tau_syn_in)
        amp_IE = max_PSP_exp(tau_I,tau_syn_ex)
        amp_II = max_PSP_exp(tau_I,tau_syn_in)

    elif 'gif_psc_exp' in neuron_type :
        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        amp_EE = max_PSP_exp(tau_E,tau_syn_ex)
        amp_EI = max_PSP_exp(tau_E,tau_syn_in)
        amp_IE = max_PSP_exp(tau_I,tau_syn_ex)
        amp_II = max_PSP_exp(tau_I,tau_syn_in)

    else:
        amp_EE = 1.
        amp_EI = 1.
        amp_IE = 1.
        amp_II = 1.

    js = np.zeros((2,2))  
    K_EE = N_E*ps[0,0]
    js[0,0] = (V_th_E-E_L)*(K_EE**-0.5)*N**0.5/amp_EE
    js[0,1] = -ge*js[0,0]*ps[0,0]*N_E*amp_EE/(ps[0,1]*N_I*amp_EI)
    K_IE = N_E*ps[1,0]
    js[1,0] = gie * (V_th_I-E_L)*(K_IE**-0.5)*N**0.5/amp_IE 
    #js[1,0]= gie * tau_E/tau_I * ps[0,0]/ps[1,0] * amp_EE/amp_IE * js[0,0]
    js[1,1] = -gi*js[1,0]*ps[1,0]*N_E*amp_IE/(ps[1,1]*N_I*amp_II)
    
    
        
        
    print(js)
    return js


def FPT(tau_m,E_L,I_e,C_m,Vtarget,Vstart):
    """ calculate first pasage time between Vstart and Vtarget."""
    return -tau_m*np.log((Vtarget -E_L - tau_m*I_e/C_m)/(Vstart -E_L - tau_m*I_e/C_m+small))


def V_FPT(tau_m,E_L,I_e,C_m,Ttarget,Vtarget,t_ref):
    """ calculate the initial voltage required to obtain a certain first passage time. """
    return (Vtarget- E_L-tau_m*I_e/C_m)*np.exp((Ttarget)/tau_m)+E_L+tau_m*I_e/C_m


def simulate(params, PathSpikes=None, timeout=None):
    if timeout is not None:
        # Change the behavior of SIGALRM
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)    
        # This try/except loop ensures that 
        #   you'll catch TimeoutException when it's sent.
    try:

        startbuild = time.time()

        nest.ResetKernel()
        nest.set_verbosity('M_WARNING')

        #randseed = params.get('randseed',np.random.randint(1000000))
        randseed = params.get('randseed', 42)
        ##todo: 1. random seed
        
        # simulation parameters
        dt = params.get('dt',0.1)   # integration time step
        simtime = params.get('simtime',1000.)  # simulation time
        warmup = params.get('warmup',0)
        record_voltage = params.get('record_voltage',False) 
        record_from = params.get('record_from','all')
        recording_interval = params.get('recording_interval',dt)
        return_weights = params.get('return_weights',False)
        n_jobs = params.get('n_jobs',1)

        # stimulation
        stim_clusters = params.get('stim_clusters',None) # clusters to be stimulated
        stim_amp = params.get('stim_amp',0.)             # amplitude of the stimulation current in pA
        stim_starts = params.get('stim_starts',[])       # list of stimulation start times
        stim_ends = params.get('stim_ends',[])           # list of stimulation end times

        # multiple stimulations 
        multi_stim_clusters = params.get('multi_stim_clusters',None)
        multi_stim_amps = params.get('multi_stim_amps',[])
        multi_stim_times = params.get('multi_stim_times',[])



        # network parameters
        N_E = params.get('N_E', default.N_E)                          # excitatory units
        N_I = params.get('N_I', default.N_I)                          # inhibitory units
        N = N_E+N_I                                                  # total units


        # connectivity parameters
        ps = params.get('ps', default.ps) # connection probs
        js = params.get('js', default.js)           # connection weights
        ge = params.get('ge', default.ge)
        gi = params.get('gi', default.gi)                             # relative strength of inhibition
        Q = params.get('Q', default.Q)                                 # number of clusters
        jplus = params.get('jplus', default.jplus)            # intra-cluster weight factors
        delay = params.get('delay', default.delay)                       # synaptic delay
        s = params.get('s', default.s)                               # scale factor applied to js
        # make sure number of clusters and units are compatible
        assert N_E%Q == 0, 'N_E needs to be evenly divisible by Q'
        assert N_I%Q == 0, 'N_I needs to be evenly divisible by Q'

        try:
            DistParams=params.get('DistParams', default.DistParams)
        except AttributeError:
            DistParams={'distribution':'normal', 'sigma': 0.0, 'fraction': False}



        fixed_indegree  = params.get('fixed_indegree', default.fixed_indegree)


        # neuron parameters
        neuron_type = params.get('neuron_type', default.neuron_type)
        E_L = params.get('E_L', default.E_L)          # resting potential
        C_m = params.get('C_m', default.C_m)          # membrane capacitance
        tau_E = params.get('tau_E', default.tau_E)     # excitatory membrane time constant
        tau_I = params.get('tau_I', default.tau_I)     # inhibitory membrane time constant
        t_ref = params.get('t_ref', default.t_ref)      # refractory period
        V_th_E = params.get('V_th_E', default.V_th_E)   # threshold voltage
        V_th_I = params.get('V_th_I', default.V_th_E)
        V_r = params.get('V_r', default.V_r)          # reset voltage
        I_th_E = params.get('I_th_E', default.I_th_E)
        if I_th_E is None:
            I_xE = params.get('I_xE', default.I_xE)
        else:
            I_xE = I_th_E * (V_th_E - E_L)/tau_E * C_m

        I_th_I = params.get('I_th_I', default.I_th_I)
        if I_th_I is None:
            I_xI = params.get('I_xI', default.I_xI)
        else:
            I_xI = I_th_I * (V_th_I - E_L)/tau_I * C_m
        #print I_xE,I_xI
        delta_I_xE = params.get('delta_I_xE', default.delta_I_xE)
        delta_I_xI = params.get('delta_I_xI', default.delta_I_xI)
        
        V_m = params.get('V_m', default.V_m)

        if 'gif_psc_exp' in neuron_type:
            E_neuron_params = {'E_L': E_L, 'C_m': C_m, 't_ref': t_ref,  'V_reset': V_r,
                               'I_e': I_xE}
            I_neuron_params = {'E_L': E_L, 'C_m': C_m, 't_ref': t_ref,  'V_reset': V_r,
                               'I_e': I_xI}

            tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
            tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
            lambda_0 = params.get('lambda_0', default.lambda_0)
            g_L_E = params.get('g_L_E', default.g_L_E)
            g_L_I = params.get('g_L_I', default.g_L_I)
            q_sfa = params.get('q_sfa', default.q_sfa)
            tau_sfa = params.get('tau_sfa', default.tau_sfa)
            Delta_V=params.get('Delta_V', default.Delta_V)
            q_stc = params.get('q_stc', default.q_stc)
            tau_stc = params.get('tau_stc', default.tau_stc)

            print(q_stc)

            E_neuron_params['tau_syn_ex'] = tau_syn_ex
            E_neuron_params['tau_syn_in'] = tau_syn_in
            I_neuron_params['tau_syn_in'] = tau_syn_in
            I_neuron_params['tau_syn_ex'] = tau_syn_ex

            E_neuron_params['lambda_0'] = lambda_0
            E_neuron_params['g_L'] = g_L_E
            E_neuron_params['q_sfa'] = q_sfa
            E_neuron_params['tau_sfa'] = tau_sfa
            E_neuron_params['q_stc'] = q_stc
            E_neuron_params['tau_stc'] = tau_stc
            E_neuron_params['V_T_star'] = V_th_E
            E_neuron_params['Delta_V'] = Delta_V

            I_neuron_params['lambda_0'] = lambda_0
            I_neuron_params['g_L'] = g_L_I
            I_neuron_params['q_sfa'] = q_sfa
            I_neuron_params['tau_sfa'] = tau_sfa
            I_neuron_params['q_stc'] = q_stc
            I_neuron_params['tau_stc'] = tau_stc
            I_neuron_params['V_T_star'] = V_th_I
            I_neuron_params['Delta_V'] = Delta_V
        else:
        
            E_neuron_params = {'E_L':E_L,'C_m':C_m,'tau_m':tau_E,'t_ref':t_ref,'V_th':V_th_E,'V_reset':V_r,'I_e':I_xE}
            I_neuron_params = {'E_L':E_L,'C_m':C_m,'tau_m':tau_I,'t_ref':t_ref,'V_th':V_th_I,'V_reset':V_r,'I_e':I_xI}
        if 'iaf_psc_exp' in neuron_type :
            tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
            tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
            E_neuron_params['tau_syn_ex'] = tau_syn_ex
            E_neuron_params['tau_syn_in'] = tau_syn_in
            I_neuron_params['tau_syn_in'] = tau_syn_in
            I_neuron_params['tau_syn_ex'] = tau_syn_ex

            # iaf_psc_exp allows stochasticity, if not used - ignore
            try:
                delta_ = params.get('delta_', default.delta_)
                rho = params.get('rho', default.rho)
                E_neuron_params['delta'] = delta_
                I_neuron_params['delta'] = delta_
                E_neuron_params['rho'] = rho
                I_neuron_params['rho'] = rho
            except AttributeError:
                pass

        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        if np.isnan(js).any():
           js = calc_js(params)
        js *= s
         
        #print js/np.sqrt(N)

        # jminus is calculated so that row sums remain constant
        if Q>1:
            jminus =(Q-jplus)/float(Q-1)
        else:
            jplus = np.ones((2,2))
            jminus = np.ones((2,2))


        # offgrid_spiking ? 
       
        
        #np.random.seed(randseed)
        np.random.seed(42)
        ##todo: 2. random seed
        randseeds = range(randseed+2,randseed+2+n_jobs)
        nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True,'local_num_threads':n_jobs,'grng_seed':randseed+1,'rng_seeds':randseeds})
        print("Building network")


        # create the neuron populations
        E_pops = []
        I_pops = []
        for q in range(Q):
            E_pops.append(nest.Create(neuron_type,int(N_E/Q)))
            nest.SetStatus(E_pops[-1],[E_neuron_params])
        for q in range(Q):
            I_pops.append(nest.Create(neuron_type,int(N_I/Q)))
            nest.SetStatus(I_pops[-1],[I_neuron_params])

        if delta_I_xE>0:
            for E_pop in E_pops:
                I_xEs = nest.GetStatus(E_pop,'I_e')
                nest.SetStatus(E_pop,[{'I_e':(1-0.5*delta_I_xE+np.random.rand()*delta_I_xE)*ixe} for ixe in I_xEs])

        if delta_I_xI>0:
            for I_pop in I_pops:
                I_xIs = nest.GetStatus(I_pop,'I_e')
                nest.SetStatus(I_pop,[{'I_e':(1-0.5*delta_I_xI+np.random.rand()*delta_I_xI)*ixi} for ixi in I_xIs])
        #print nest.GetStatus(E_pops[-1],'I_e'),nest.GetStatus(I_pops[-1],'I_e'),
        # set some random initial value for the membrane voltage
        if V_m == 'rand':
            T_0_E = t_ref +FPT(tau_E,E_L,I_xE,C_m,V_th_E,V_r)
            if np.isnan(T_0_E):
                T_0_E = 10.
            for E_pop in E_pops:
                nest.SetStatus(E_pop,[{'V_m':V_FPT(tau_E,E_L,I_xE,C_m,T_0_E*np.random.rand(),V_th_E,t_ref)} for i in range(len(E_pop))])
            
            T_0_I = t_ref +FPT(tau_I,E_L,I_xI,C_m,V_th_I,V_r)
            if np.isnan(T_0_I):
                T_0_I = 10.
            for I_pop in I_pops:
                nest.SetStatus(I_pop,[{'V_m':V_FPT(tau_I,E_L,I_xI,C_m,T_0_I*np.random.rand(),V_th_E,t_ref)} for i in range(len(I_pop))])
        else:
            nest.SetStatus(tuple(range(1,N+1)),[{'V_m':V_m} for i in range(N)])


        # define the synapses and connect the populations
        # EE
        j_ee =js[0,0]/np.sqrt(N)

        nest.CopyModel("static_synapse","EE_plus",{"weight": jplus[0,0]*j_ee, "delay":delay})
        nest.CopyModel("static_synapse","EE_minus",{"weight":jminus[0,0]*j_ee, "delay":delay})
        if fixed_indegree:
            K_EE = int(ps[0,0] * N_E / Q)
            print('K_EE: ',K_EE)
            conn_params_EE = {'rule': 'fixed_indegree', 'indegree': K_EE,'autapses':False,'multapses':False}

        else:
            conn_params_EE = {'rule': 'pairwise_bernoulli', 'p': ps[0,0],'autapses':False,'multapses':False}
        for i,pre in enumerate(E_pops):
            for j,post in enumerate(E_pops):
                if i==j:
                    # same cluster
                    nest.Connect(pre,post,conn_params_EE,'EE_plus')
                    #print 'EE_plus: ',nest.GetDefaults('EE_plus')
                else:
                    nest.Connect(pre,post,conn_params_EE,'EE_minus')
        # EI
        j_ei =js[0,1]/np.sqrt(N)
        nest.CopyModel("static_synapse","EI_plus",{"weight":j_ei*jplus[0,1], "delay":delay})
        nest.CopyModel("static_synapse","EI_minus",{"weight":j_ei*jminus[0,1], "delay":delay})
        if fixed_indegree:
            K_EI = int(ps[0,1] * N_I / Q)
            print('K_EI: ',K_EI)
            conn_params_EI = {'rule': 'fixed_indegree', 'indegree': K_EI,'autapses':False,'multapses':False}
        else:
            conn_params_EI = {'rule': 'pairwise_bernoulli', 'p': ps[0,1],'autapses':False,'multapses':False}
        for i,pre in enumerate(I_pops):
            for j,post in enumerate(E_pops):
                if i==j:
                    # same cluster
                    nest.Connect(pre,post,conn_params_EI, 'EI_plus')
                    ##print 'EI_plus: ',nest.GetDefaults('EI_plus')
                else:
                    nest.Connect(pre,post,conn_params_EI, 'EI_minus')
        # IE
        j_ie = js[1,0]/np.sqrt(N)
        nest.CopyModel("static_synapse","IE_plus",{"weight":j_ie *jplus[1,0], "delay":delay})
        #IE_plus = {"model": "static_synapse","delay": delay}
        nest.CopyModel("static_synapse","IE_minus",{"weight":j_ie *jminus[1,0], "delay":delay})
        #IE_minus = {"model": "static_synapse", "delay": delay}

        if fixed_indegree:
            K_IE = int(ps[1,0] * N_E / Q)
            print('K_IE: ',K_IE)
            conn_params_IE = {'rule': 'fixed_indegree', 'indegree': K_IE,'autapses':False,'multapses':False}
        else:
            conn_params_IE = {'rule': 'pairwise_bernoulli', 'p': ps[1,0],'autapses':False,'multapses':False}
        for i,pre in enumerate(E_pops):
            for j,post in enumerate(I_pops):
                if i==j:
                    # same cluster
                    nest.Connect(pre,post,conn_params_IE, 'IE_plus')
                    #print 'IE_plus: ',nest.GetDefaults('IE_plus')
                else:
                    nest.Connect(pre,post,conn_params_IE, 'IE_minus')

        # II
        j_ii = js[1,1]/np.sqrt(N)
        nest.CopyModel("static_synapse","II_plus",{"weight":j_ii*jplus[1,1], "delay":delay})
        #II_plus = {"model": "static_synapse","delay": delay}
        nest.CopyModel("static_synapse","II_minus",{"weight":j_ii*jminus[1,1], "delay":delay})
        #II_minus = {"model": "static_synapse", "delay": delay}
        if fixed_indegree:
            K_II = int(ps[1,1] * N_I / Q)
            print('K_II: ',K_II)
            conn_params_II = {'rule': 'fixed_indegree', 'indegree': K_II,'autapses':False,'multapses':False}
        else:
            conn_params_II = {'rule': 'pairwise_bernoulli', 'p': ps[1,1],'autapses':False,'multapses':False}
        for i,pre in enumerate(I_pops):
            for j,post in enumerate(I_pops):
                if i==j:
                    # same cluster
                    nest.Connect(pre,post,conn_params_II, 'II_plus')
                    #print 'II_plus: ',nest.GetDefaults('II_plus')
                else:
                    nest.Connect(pre,post,conn_params_II, 'II_minus')
        print('Js: ',js/np.sqrt(N))
        # set up spike detector
        spike_detector = nest.Create("spike_detector")
        nest.SetStatus(spike_detector,[{'to_file':False,'withtime':True,'withgid':True}])
        all_units = ()
        for E_pop in E_pops:
            all_units += E_pop
        for I_pop in I_pops:
            all_units += I_pop
        nest.Connect(all_units,spike_detector,"all_to_all",'EE_plus')
        
        # set up stimulation
        if stim_clusters is not None:
            current_source = nest.Create('step_current_generator')
            amplitude_values = []
            amplitude_times = []
            for start,end in zip(stim_starts,stim_ends):
                amplitude_times.append(start+warmup)
                amplitude_values.append(stim_amp)
                amplitude_times.append(end+warmup)
                amplitude_values.append(0.)
            nest.SetStatus(current_source,{'amplitude_times':amplitude_times,'amplitude_values':amplitude_values})
            stim_units = []
            for stim_cluster in stim_clusters:
                stim_units+= list(E_pops[stim_cluster])
            nest.Connect(current_source,stim_units)

       # elif multi_stim_clusters is not None:
       #     for stim_clusters,amplitudes,times in zip(multi_stim_clusters,multi_stim_amps,multi_stim_times):
       #         current_source = nest.Create('step_current_generator')
       #         nest.SetStatus(current_source,{'amplitude_times':times,'amplitude_values':amplitudes})
       #         stim_units = []
       #         for stim_cluster in stim_clusters:
       #             stim_units+= list(E_pops[stim_cluster])
       #         nest.Connect(current_source,stim_units)
        endbuild = time.time() 
        endcompile = time.time()
        print("Loading model")
            
        nest.Prepare()
        #nest.Cleanup()    
        
        endLoad = time.time()

        if warmup + simtime<=0.1:
            endsimulate = time.time()
            endPullSpikes= time.time()
            results={'spiketimes': -1, 'e_rate': -1, 'i_rate': -1}

        else:

        
        
            #nest.Simulate(warmup + simtime)
            nest.Run(warmup + simtime)
            endsimulate = time.time()
            # get the spiketimes from the detector
            print('extracting spike times')
            events = nest.GetStatus(spike_detector,'events')[0]
            # convert them to the format accepted by spiketools
            spiketimes = np.append(events['times'][None,:],events['senders'][None,:],axis=0)
            spiketimes[1] -=1
            # remove the pre warmup spikes
            spiketimes = spiketimes[:,spiketimes[0]>=warmup]

            spiketimes[0] -= warmup
            endPullSpikes= time.time()
            
            results = {'spiketimes':spiketimes}
                    
            e_count = results['spiketimes'][:, results['spiketimes'][1] < N_E].shape[1]
            i_count = results['spiketimes'][:, results['spiketimes'][1] >= N_E].shape[1]
            e_rate = e_count / float(N_E) / float(simtime) * 1000.
            i_rate = i_count / float(N_I) / float(simtime) * 1000.
            results['e_rate'] = e_rate
            results['i_rate'] = i_rate

            if PathSpikes is not None:
                with open(PathSpikes, 'wb') as outfile:
                    for ii, res in enumerate(results):
                        pickle.dump(results['spiketimes']
                        , outfile)

        nest.Cleanup()

        build_time = endbuild - startbuild
        compile_time = 0.0
        load_time = endLoad - endcompile
        sim_time = endsimulate - endLoad
        down_time = endPullSpikes - endsimulate

    

        

        # print("\nTiming:")
        # print("Build time     : %.4f s" % build_time)
        # print("Compile time   : %.4f s" % compile_time)
        # print("Load model time   : %.4f s" % load_time)
        # print("Simulation time   : %.4f s" % sim_time)
        # print("Download and process spikes time   : %.4f s\n" % down_time)

        Timing={'Build': build_time, 'Compile': compile_time, 'Load': load_time, 'Sim': sim_time, 'Download': down_time }
        results["Params"]=params

        return {'e_rate': results['e_rate'], 'i_rate': results['i_rate'], 'Timing': Timing, 'params': mergeParams(params)}

    except TimeoutException:
        print("Aborted - Timeout")
        Timing={'Build': -1, 'Compile': -1, 'Load': -1, 'Sim': -1, 'Download': -1 }
        if 'endbuild' in locals():
            build_time = endbuild - startbuild
            Timing['Build']= build_time
        if 'endcompile' in locals():
            compile_time = endcompile - startCompile
            Timing['Compile']= compile_time
        if 'endLoad' in locals():
            load_time = endLoad - endcompile
            Timing['Load']= load_time
        if 'endsimulate' in locals():
            sim_time = endsimulate - endLoad
            Timing['Sim']= sim_time
        if 'endPullSpikes' in locals():
            down_time = endPullSpikes - endsimulate
            Timing['Download']= down_time 
        return {'e_rate': -1, 'i_rate': -1, 'Timing': Timing, 'params': mergeParams(params)}
    


#######################################################################################################################################
# main#
#######################################################################################################################################

if __name__ == '__main__':
    if len(sys.argv)==2:
        FactorSize=int(sys.argv[1])
        FactorTime=int(sys.argv[1])

    elif len(sys.argv)==3:
        FactorSize=int(sys.argv[1])
        FactorTime=int(sys.argv[2])
    elif len(sys.argv)>=3:
        FactorSize=int(sys.argv[1])
        FactorTime=int(sys.argv[2])
        print("Too many arguments")
    else:
        FactorSize=1
        FactorTime=1

    print("FactorSize: " + str(FactorSize) + " FactorTime: " +  str(FactorTime))
        


    startTime=time.time()
    js_maz = np.array([[1.77, -3.18], [1.06, -4.24]])


    baseline={'N_E': 400, 'N_I': 100,  # number of E/I neurons -> typical 4:1
              'simtime': 900,'warmup': 100}

    params = {'n_jobs': 20,
              'N_E': FactorSize*baseline['N_E'], 'N_I': FactorSize*baseline['N_I'],  # number of E/I neurons -> typical 4:1
              'dt': 0.1,
              'neuron_type': 'iaf_psc_exp',
              'simtime': FactorTime*baseline['simtime'],  # simulation time
              'delta_I_xE': 0., 'delta_I_xI': 0.,
              'record_voltage': False, 'record_from': 1, 'warmup': FactorTime*baseline['warmup']}

    N = params['N_E'] + params['N_I']

    params['Q'] = 20  # number of clusters
    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation

    jep = 10.0  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])


    I_ths = [2.13,
             1.24]  # 3,5,Hz        #background stimulation of E/I neurons -> sets firing rates and changes behavior to some degree # I_ths = [5.34,2.61] # 10,15,Hzh
 
    # print('result: ', I_ths)
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]

    #timeout=28800 # 8h
    timeout=7200 # 1h 

    params['matrixType']= "PROCEDURAL_GLOBALG"
    #params['matrixType']="SPARSE_GLOBALG"

    
    Result = Simulate(params, timeout=timeout)
    print(Result)
    stopTime=time.time()
    Result['Timing']['Total']=stopTime-startTime
    print("Total time     : %.4f s" % Result['Timing']['Total'])

    # print(Result['Timing'])
    # print(Result['e_rate'])
    # print(Result['i_rate'])

    # plt.subplot(1, 1, 1)
    # plt.plot(Result['spiketimes'][0,:], Result['spiketimes'][1,:], '.k', markersize=1)
    # plt.show()

    with open("Results.pkl", 'ab') as outfile:
             pickle.dump(Result, outfile)






