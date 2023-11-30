from pygenn import genn_model, genn_wrapper

from Helper import ModelDefinition
from Helper import ClusterHelper
from Helper import GeneralHelper
from Defaults import defaultSimulate as default





def create_Model(params, batch_size=1, default, NModel= None):
    ## To-Do expand genn_model by adding the list recorded Populations with getter and setter
# simulation parameters
    dt = params.get('dt', 0.1)  # integration time step
    simtime = params.get('simtime', 1000.)  # simulation time
    warmup = params.get('warmup', 0)
    record_voltage = params.get('record_voltage', False)
    record_from = params.get('record_from', 'all')
    recording_interval = params.get('recording_interval', dt)
    return_weights = params.get('return_weights', False)

    # create Model
    model = genn_model.GeNNModel("float", "Gridsearch", generateEmptyStatePushPull=False, generateExtraGlobalParamPull=False)
    model.dT = dt
    model._model.set_merge_postsynaptic_models(True)
    model._model.set_default_narrow_sparse_ind_enabled(True)
    assert batch_size>=1, "batch_size has to be 1 or greater"
    if batch_size>1:
        model.batch_size = batch_size
    model.timing_enabled = MEASURE_TIMING
    #model.default_var_location = genn_wrapper.VarLocation_DEVICE
    #model.default_sparse_connectivity_location = genn_wrapper.VarLocation_DEVICE


    cluster_stimulus=ModelDefinition.define_ClusterStim()


    # stimulation
    stim_clusters = params.get('stim_clusters', None)  # clusters to be stimulated
    stim_amp = params.get('stim_amp', 0.)  # amplitude of the stimulation current in pA
    stim_starts = params.get('stim_starts', [])  # list of stimulation start times
    stim_ends = params.get('stim_ends', [])  # list of stimulation end times

    # multiple stimulations
    multi_stim_clusters = params.get('multi_stim_clusters', None)
    multi_stim_amps = params.get('multi_stim_amps', [])
    multi_stim_times = params.get('multi_stim_times', [])

    # network parameters
    N_E = params.get('N_E', default.N_E)  # excitatory units
    N_I = params.get('N_I', default.N_I)  # inhibitory units
    N = N_E + N_I  # total units

    # connectivity parameters
    ps = params.get('ps', default.ps)  # connection probs
    js = params.get('js', default.js)  # connection weights
    ge = params.get('ge', default.ge)
    gi = params.get('gi', default.gi)  # relative strength of inhibition
    Q = params.get('Q', default.Q)  # number of clusters
    jplus = params.get('jplus', default.jplus)  # intra-cluster weight factors
    delay = params.get('delay', default.delay)  # synaptic delay
    delaySteps=int((delay+0.5*model.dT)//model.dT)
    s = params.get('s', default.s)  # scale factor applied to js
    # make sure number of clusters and units are compatible
    assert N_E % Q == 0, 'N_E needs to be evenly divisible by Q'
    assert N_I % Q == 0, 'N_I needs to be evenly divisible by Q'

    #if Distribution wanted - implement!
    try:
        DistParams = params.get('DistParams', default.DistParams)
    except AttributeError:
        DistParams = {'distribution': 'normal', 'sigma': 0.0, 'fraction': False}

    fixed_indegree = params.get('fixed_indegree', default.fixed_indegree)

    # neuron parameters
    neuron_type = params.get('neuron_type', default.neuron_type)
    E_L = params.get('E_L', default.E_L)  # resting potential
    C_m = params.get('C_m', default.C_m)  # membrane capacitance
    tau_E = params.get('tau_E', default.tau_E)  # excitatory membrane time constant
    tau_I = params.get('tau_I', default.tau_I)  # inhibitory membrane time constant
    t_ref = params.get('t_ref', default.t_ref)  # refractory period
    V_th_E = params.get('V_th_E', default.V_th_E)  # threshold voltage
    V_th_I = params.get('V_th_I', default.V_th_E)
    V_r = params.get('V_r', default.V_r)  # reset voltage
    I_th_E = params.get('I_th_E', default.I_th_E)
    if I_th_E is None:
        I_xE = params.get('I_xE', default.I_xE)
    else:
        I_xE = I_th_E * (V_th_E - E_L) / tau_E * C_m

    I_th_I = params.get('I_th_I', default.I_th_I)
    if I_th_I is None:
        I_xI = params.get('I_xI', default.I_xI)
    else:
        I_xI = I_th_I * (V_th_I - E_L) / tau_I * C_m
    # print I_xE,I_xI
    delta_I_xE = params.get('delta_I_xE', default.delta_I_xE)
    delta_I_xI = params.get('delta_I_xI', default.delta_I_xI)

    V_m = params.get('V_m', default.V_m)

    if 'gif_psc_exp' in neuron_type:
        if NModel is not None:
            neuron_model=NModel
        else:
                # import NeuronModel + StimulationModel
            neuron_model=ModelDefinition.define_GIF()
        
        E_neuron_params = {'E_L': E_L, 'C_m': C_m, 't_ref': t_ref, 'V_reset': V_r,
                           'I_e': I_xE}

        E_neuron_init= {"sfa": 0,
            "TH": V_th_E, "stc": 0,
           # "V": genn_model.init_var("Normal", {"mean": 12, "sd": 4}),
            "RefracTime": -0.1,
            "lambda": 0,
            "u": 1}
        I_neuron_params = {'E_L': E_L, 'C_m': C_m, 't_ref': t_ref, 'V_reset': V_r,
                           'I_e': I_xI}
        I_neuron_init = {"sfa": 0,
                         "TH": V_th_E, "stc": 0,
                         # "V": genn_model.init_var("Normal", {"mean": 12, "sd": 4}),
                         "RefracTime": -0.1,
                         "lambda": 0,
                         "u": 1}



        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        lambda_0 = params.get('lambda_0', default.lambda_0)
        g_L_E = params.get('g_L_E', default.g_L_E)
        g_L_I = params.get('g_L_I', default.g_L_I)
        q_sfa = params.get('q_sfa', default.q_sfa)
        tau_sfa = params.get('tau_sfa', default.tau_sfa)
        Delta_V = params.get('Delta_V', default.Delta_V)
        q_stc = params.get('q_stc', default.q_stc)
        tau_stc = params.get('tau_stc', default.tau_stc)

        #print(q_stc)

        psc_E = {"tau": tau_syn_ex}  # synaptic time constant
        psc_I = {"tau": tau_syn_in}  # synaptic time constant

        E_neuron_params['lambda_0'] = lambda_0
        E_neuron_params['g_L'] = g_L_E
        E_neuron_params['q_sfa'] = q_sfa[0]
        E_neuron_params['tau_sfa'] = tau_sfa[0]
        E_neuron_params['q_stc'] = q_stc[0]
        E_neuron_params['tau_stc'] = tau_stc[0]
        E_neuron_params['V_T_star'] = V_th_E
        E_neuron_params['Delta_V'] = Delta_V

        I_neuron_params['lambda_0'] = lambda_0
        I_neuron_params['g_L'] = g_L_I
        I_neuron_params['q_sfa'] = q_sfa[0]
        I_neuron_params['tau_sfa'] = tau_sfa[0]
        I_neuron_params['q_stc'] = q_stc[0]
        I_neuron_params['tau_stc'] = tau_stc[0]
        I_neuron_params['V_T_star'] = V_th_I
        I_neuron_params['Delta_V'] = Delta_V
    else:
        if NModel is not None: 
            neuron_model=NModel
        else: 
            neuron_model="LIF"
        E_neuron_params = {'Vrest': E_L , 'C': C_m, 'TauM': tau_E, 'TauRefrac': t_ref, 'Vthresh': V_th_E, 'Vreset': V_r,
                           'Ioffset': I_xE}
        E_neuron_init = {
                         # "V": genn_model.init_var("Normal", {"mean": 12, "sd": 4}),
                         "RefracTime": -0.1
                         }
        I_neuron_params = {'Vrest': E_L , 'C': C_m, 'TauM': tau_I, 'TauRefrac': t_ref, 'Vthresh': V_th_I, 'Vreset': V_r,
                           'Ioffset': I_xI}
        I_neuron_init = {
            # "V": genn_model.init_var("Normal", {"mean": 12, "sd": 4}),
            "RefracTime": -0.1
        }
    if 'iaf_psc_exp' in neuron_type:
        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        psc_E = {"tau": tau_syn_ex}  # synaptic time constant
        psc_I = {"tau": tau_syn_in}  # synaptic time constant


    # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
    if np.isnan(js).any():
        js = ClusterHelper.calc_js(params)
    js *= s

    # print js/np.sqrt(N)

    # jminus is calculated so that row sums remain constant
    if Q > 1:
        jminus = (Q - jplus) / float(Q - 1)
    else:
        jplus = np.ones((2, 2))
        jminus = np.ones((2, 2))



    #print("Building network")

    if V_m == 'rand':


        T_0_E = t_ref + ClusterHelper.FPT(tau_E, E_L, I_xE, C_m, V_th_E, V_r)
        if np.isnan(T_0_E):
            T_0_E = 10.

        T_0_I = t_ref +ClusterHelper.FPT(tau_I,E_L,I_xI,C_m,V_th_I,V_r)
        if np.isnan(T_0_I):
            T_0_I = 10.

 
    else:
        E_neuron_init["V"] = V_m
        I_neuron_init["V"] = V_m

    # create the neuron populations
    E_pops = []
    I_pops = []


    for q in range(Q):
        if V_m == 'rand':
            E_neuron_init["V"] =  [ClusterHelper.V_FPT(tau_E,E_L,I_xE,C_m,T_0_E*np.random.rand(),V_th_E,t_ref) for i in range(int(N_E / Q))]
        E_pops.append(model.add_neuron_population("Egif"+str(q), int(N_E / Q), neuron_model, E_neuron_params, E_neuron_init))
    for q in range(Q):
        if V_m == 'rand':
            I_neuron_init["V"] =  [ClusterHelper.V_FPT(tau_I,E_L,I_xI,C_m,T_0_I*np.random.rand(),V_th_E,t_ref) for i in range(int(N_I / Q))]
        I_pops.append(model.add_neuron_population("Igif" + str(q), int(N_I / Q), neuron_model, I_neuron_params, I_neuron_init))
   

    # define the synapses and connect the populations
    # EE
    j_ee = js[0, 0] / np.sqrt(N)
    EE_plus = {"model": "static_synapse", "delay": delay}
    EE_minus = {"model": "static_synapse", "delay": delay}

    if fixed_indegree:
        K_EE = int(ps[0, 0] * N_E / Q)
        print('K_EE: ', K_EE)
        conn_params_EE_same= genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                 {"colLength": K_EE})
        conn_params_EE_different= genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                 {"colLength": K_EE})
    else:

        conn_params_EE_same = genn_model.init_connectivity("FixedProbabilityNoAutapse",
                              {"prob": ps[0, 0]})
        conn_params_EE_different = genn_model.init_connectivity("FixedProbability",
                              {"prob": ps[0, 0]})


    for i, pre in enumerate(E_pops):
        for j, post in enumerate(E_pops):
            if i == j:
                # same cluster
                model.add_synapse_population(str(i) +"EE" +str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": jplus[0, 0] * j_ee}, {}, {},
                                             "ExpCurr", psc_E, {},conn_params_EE_same
                                             )
            else:
                model.add_synapse_population(str(i) +"EE" +str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": jminus[0, 0] * j_ee}, {}, {},
                                             "ExpCurr", psc_E, {}, conn_params_EE_different
                                             )


    # EI
    j_ei = js[0, 1] / np.sqrt(N)
    EI_plus = {"model": "static_synapse", "delay": delay}
    EI_minus = {"model": "static_synapse", "delay": delay}

    if fixed_indegree:
        K_EI = int(ps[0, 1] * N_I / Q)
        print('K_EI: ', K_EI)
        conn_params_EI = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                      {"colLength": K_EI})
    else:
        conn_params_EI = genn_model.init_connectivity("FixedProbability",
                                                      {"prob": ps[0, 1]})


    for i, pre in enumerate(I_pops):
        for j, post in enumerate(E_pops):
            if i == j:
                # same cluster
                model.add_synapse_population(str(i) + "EI" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": jplus[0, 1]*j_ei}, {}, {},
                                             "ExpCurr", psc_I, {}, conn_params_EI
                                             )
            else:
                model.add_synapse_population(str(i) + "EI" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": jminus[0, 1] * j_ei}, {}, {},
                                             "ExpCurr", psc_I, {}, conn_params_EI
                                             )
    # IE
    j_ie = js[1, 0] / np.sqrt(N)
    IE_plus = {"model": "static_synapse", "delay": delay}
    IE_minus = {"model": "static_synapse", "delay": delay}

    if fixed_indegree:
        K_IE = int(ps[1, 0] * N_E / Q)
        print('K_IE: ', K_IE)
        conn_params_IE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                      {"colLength": K_IE})
    else:
        conn_params_IE = genn_model.init_connectivity("FixedProbability",
                                                      {"prob": ps[1, 0]})


    for i, pre in enumerate(E_pops):
        for j, post in enumerate(I_pops):
            if i == j:
                # same cluster
                model.add_synapse_population(str(i) + "IE" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": jplus[1, 0] * j_ie}, {}, {},
                                             "ExpCurr", psc_E, {}, conn_params_IE
                                             )
            else:
                model.add_synapse_population(str(i) + "IE" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": j_ie * jminus[1, 0]}, {}, {},
                                             "ExpCurr", psc_E, {}, conn_params_IE
                                             )

    # II
    j_ii = js[1, 1] / np.sqrt(N)
    II_plus = {"model": "static_synapse", "delay": delay}
    II_minus = {"model": "static_synapse", "delay": delay}

 

    if fixed_indegree:
        K_II = int(ps[1, 1] * N_I / Q)
        print('K_II: ', K_II)
        conn_params_II_different = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                      {"colLength": K_II})
        conn_params_II_same = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                      {"colLength": K_II})
    else:
        conn_params_II_different = genn_model.init_connectivity("FixedProbability",
                                                      {"prob": ps[1, 1]})
        conn_params_II_same = genn_model.init_connectivity("FixedProbabilityNoAutapse",
                                                      {"prob": ps[1, 1]})



    for i, pre in enumerate(I_pops):
        for j, post in enumerate(I_pops):
            if i == j:
                # same cluster
                model.add_synapse_population(str(i) + "II" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": j_ii * jplus[1, 1]}, {}, {},
                                             "ExpCurr", psc_I, {}, conn_params_II_same
                                             )
            else:
                model.add_synapse_population(str(i) + "II" + str(j), "SPARSE_GLOBALG", delaySteps,
                                             pre, post,
                                             "StaticPulse", {}, {"g": j_ii * jminus[1, 1]}, {}, {},
                                             "ExpCurr", psc_I, {}, conn_params_II_different
                                             )
    print('Js: ', js / np.sqrt(N))


    # set up stimulation
    if stim_clusters is not None:
        amplitude_values = []
        amplitude_times = []
        for ii, (start, end) in enumerate(zip(stim_starts, stim_ends)):
            for jj, stim_cluster in enumerate(stim_clusters):
                model.add_current_source(str(ii)+"_StimE_" + str(jj), cluster_stimulus, E_pops[stim_cluster], {"t_onset":start + warmup, "t_offset":end + warmup, "strength":stim_amp}, {})


    #elif multi_stim_clusters is not None:
    #    for ii,(stim_clusters, amplitudes, times) in enumerate(zip(multi_stim_clusters, multi_stim_amps, multi_stim_times)):
            #current_source = nest.Create('step_current_generator')
            #nest.SetStatus(current_source, {'amplitude_times': times, 'amplitude_values': amplitudes})
            #stim_units = []
            #for jj, stim_cluster in enumerate(stim_clusters):
            #     model.add_current_source(str(ii)+"_StimE_" + str(jj), cluster_stimulus, E_pops[stim_cluster], {"t_onset":times[0] + warmup, "t_offset":times[1] + warmup, "strength":stim_amp}, {})
                 #wrong implementation (implement as stepcurrents with timepoints and amplitudes )
                #stim_units += list(E_pops[stim_cluster])
            #nest.Connect(current_source, stim_units)


    # Enable spike recording
    for Pop in E_pops:
        Pop.spike_recording_enabled = True
    for Pop in I_pops:
        Pop.spike_recording_enabled = True

    RecordedPops=(SuperPopulation(E_pops, "Exc."), SuperPopulation(I_pops, "Inh."))

    return model, RecordedPops
