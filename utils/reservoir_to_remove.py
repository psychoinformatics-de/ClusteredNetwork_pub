import nest
import numpy as np
from scipy.optimize import fmin
import time
from copy import deepcopy
import default
import pylab
import global_params


small = 1e-10


def max_PSP_exp(tau_m, tau_syn, C_m=1., E_l=0.):
    tmax = np.log(tau_syn / tau_m) / (1 / tau_m - 1 / tau_syn)

    B = tau_m * tau_syn / C_m / (tau_syn - tau_m)
    return (E_l - B) * np.exp(-tmax / tau_m) + B * np.exp(-tmax / tau_syn)
    """calculates the maximum psp amplitude for exponential synapses and unit J"""


def calc_js(params):
    # excitatory units
    N_E = params.get('N_E', default.N_E)
    # inhibitory units
    N_I = params.get('N_I', default.N_I)
    N = N_E + N_I                                                  # total units
    ps = params.get('ps', default.ps)  # connection probs
    ge = params.get('ge', default.ge)
    gi = params.get('gi', default.gi)
    gie = params.get('gie', default.gie)
    V_th_E = params.get('V_th_E', default.V_th_E)   # threshold voltage
    V_th_I = params.get('V_th_I', default.V_th_I)
    tau_E = params.get('tau_E', default.tau_E)
    tau_I = params.get('tau_I', default.tau_I)
    E_L = params.get('E_L', default.E_L)
    neuron_type = params.get('neuron_type', default.neuron_type)
    if 'iaf_psc_exp' in neuron_type:
        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        amp_EE = max_PSP_exp(tau_E, tau_syn_ex)
        amp_EI = max_PSP_exp(tau_E, tau_syn_in)
        amp_IE = max_PSP_exp(tau_I, tau_syn_ex)
        amp_II = max_PSP_exp(tau_I, tau_syn_in)

    else:
        amp_EE = 1.
        amp_EI = 1.
        amp_IE = 1.
        amp_II = 1.

    js = np.zeros((2, 2))
    K_EE = N_E * ps[0, 0]
    js[0, 0] = (V_th_E - E_L) * (K_EE**-0.5) * N**0.5 / amp_EE
    js[0, 1] = -ge * js[0, 0] * ps[0, 0] * \
        N_E * amp_EE / (ps[0, 1] * N_I * amp_EI)
    K_IE = N_E * ps[1, 0]
    js[1, 0] = gie * (V_th_I - E_L) * (K_IE**-0.5) * N**0.5 / amp_IE
    #js[1,0]= gie * tau_E/tau_I * ps[0,0]/ps[1,0] * amp_EE/amp_IE * js[0,0]
    js[1, 1] = -gi * js[1, 0] * ps[1, 0] * \
        N_E * amp_IE / (ps[1, 1] * N_I * amp_II)

    return js


def FPT(tau_m, E_L, I_e, C_m, Vtarget, Vstart):
    """ calculate first pasage time between Vstart and Vtarget."""
    return -tau_m * np.log((Vtarget - E_L - tau_m * I_e / C_m) / (Vstart - E_L - tau_m * I_e / C_m + small))


def V_FPT(tau_m, E_L, I_e, C_m, Ttarget, Vtarget, t_ref):
    """ calculate the initial voltage required to obtain a certain first passage time. """
    return (Vtarget - E_L - tau_m * I_e / C_m) * np.exp((Ttarget) / tau_m) + E_L + tau_m * I_e / C_m


def build(params):
    startbuild = time.time()
    nest.set_verbosity('M_WARNING')

    # simulation parameters
    dt = params.get('dt', 0.1)   # integration time step
    simtime = params.get('simtime', 1000.)  # simulation time
    warmup = params.get('warmup', 0)
    record_voltage = params.get('record_voltage', False)
    record_from = params.get('record_from', 'all')
    recording_interval = params.get('recording_interval', dt)
    return_weights = params.get('return_weights', False)
    n_jobs = params.get('n_jobs', 1)
    fn_sd = params.get('fn_sd', "sd")

    # stimulation
    # clusters to be stimulated
    stim_clusters = params.get('stim_clusters', None)
    # amplitude of the stimulation current in pA
    stim_amp = params.get('stim_amp', 0.)
    # list of stimulation start times
    stim_starts = params.get('stim_starts', [])
    # list of stimulation end times
    stim_ends = params.get('stim_ends', [])

    # multiple stimulations
    multi_stim_clusters = params.get('multi_stim_clusters', None)
    multi_stim_amps = params.get('multi_stim_amps', [])
    multi_stim_times = params.get('multi_stim_times', [])

    # network parameters
    # excitatory units
    N_E = params.get('N_E', default.N_E)
    # inhibitory units
    N_I = params.get('N_I', default.N_I)
    N = N_E + N_I                                                  # total units

    # connectivity parameters
    ps = params.get('ps', default.ps)  # connection probs
    js = params.get('js', default.js)           # connection weights
    ge = params.get('ge', default.ge)
    # relative strength of inhibition
    gi = params.get('gi', default.gi)
    # number of clusters
    Q = params.get('Q', default.Q)
    # intra-cluster weight factors
    jplus = params.get('jplus', default.jplus)
    # synaptic delay
    delay = params.get('delay', default.delay)
    # scale factor applied to js
    s = params.get('s', default.s)
    # make sure number of clusters and units are compatible
    assert int(N_E % Q) == 0, 'N_E needs to be evenly divisible by Q'
    assert int(N_I % Q) == 0, 'N_I needs to be evenly divisible by Q'

    fixed_indegree = params.get('fixed_indegree', default.fixed_indegree)

    # neuron parameters
    neuron_type = params.get('neuron_type', default.neuron_type)
    E_L = params.get('E_L', default.E_L)          # resting potential
    C_m = params.get('C_m', default.C_m)          # membrane capacitance
    # excitatory membrane time constant
    tau_E = params.get('tau_E', default.tau_E)
    # inhibitory membrane time constant
    tau_I = params.get('tau_I', default.tau_I)
    t_ref = params.get('t_ref', default.t_ref)      # refractory period
    V_th_E = params.get('V_th_E', default.V_th_E)   # threshold voltage
    V_th_I = params.get('V_th_I', default.V_th_E)
    V_r = params.get('V_r', default.V_r)          # reset voltage
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

    delta_I_xE = params.get('delta_I_xE', default.delta_I_xE)
    delta_I_xI = params.get('delta_I_xI', default.delta_I_xI)

    V_m = params.get('V_m', default.V_m)

    E_neuron_params = {'E_L': E_L, 'C_m': C_m, 'tau_m': tau_E,
                       't_ref': t_ref, 'V_th': V_th_E, 'V_reset': V_r, 'I_e': global_params.ratio_bg_inp * I_xE}
    I_neuron_params = {'E_L': E_L, 'C_m': C_m, 'tau_m': tau_I,
                       't_ref': t_ref, 'V_th': V_th_I, 'V_reset': V_r, 'I_e': global_params.ratio_bg_inp * I_xI}
    if 'iaf_psc_exp' in neuron_type:
        tau_syn_ex = params.get('tau_syn_ex', default.tau_syn_ex)
        tau_syn_in = params.get('tau_syn_in', default.tau_syn_in)
        E_neuron_params['tau_syn_ex'] = tau_syn_ex
        E_neuron_params['tau_syn_in'] = tau_syn_in
        I_neuron_params['tau_syn_in'] = tau_syn_in
        I_neuron_params['tau_syn_ex'] = tau_syn_ex

    # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
    if np.isnan(js).any():
        js = calc_js(params)
    js *= s


    # jminus is calculated so that row sums remain constant
    if Q > 1:
        jminus = (Q - jplus) / float(Q - 1)
    else:
        jplus = np.ones((2, 2))
        jminus = np.ones((2, 2))

    # offgrid_spiking ?

    print("Building network with {} clusters".format(Q))

    # create the neuron populations
    E_pops = []
    I_pops = []
    for q in range(Q):
        E_pops.append(nest.Create(neuron_type, int(N_E / Q)))
        nest.SetStatus(E_pops[-1], [E_neuron_params])
    for q in range(Q):
        I_pops.append(nest.Create(neuron_type, int(N_I / Q)))
        nest.SetStatus(I_pops[-1], [I_neuron_params])

    if delta_I_xE > 0:
        for E_pop in E_pops:
            I_xEs = nest.GetStatus(E_pop, 'I_e')
            nest.SetStatus(E_pop, [{'I_e': (
                1 - 0.5 * delta_I_xE + np.random.rand() * delta_I_xE) * global_params.ratio_bg_inp * ixe} for ixe in I_xEs])

    if delta_I_xI > 0:
        for I_pop in I_pops:
            I_xIs = nest.GetStatus(I_pop, 'I_e')
            nest.SetStatus(I_pop, [{'I_e': (
                1 - 0.5 * delta_I_xI + np.random.rand() * delta_I_xI) * global_params.ratio_bg_inp * ixi} for ixi in I_xIs])
    # set some random initial value for the membrane voltage
    if V_m == 'rand':
        T_0_E = t_ref + FPT(tau_E, E_L, I_xE, C_m, V_th_E, V_r)
        if np.isnan(T_0_E):
            T_0_E = 10.
        for E_pop in E_pops:
            nest.SetStatus(E_pop, [{'V_m': V_FPT(
                tau_E, E_L, I_xE, C_m, T_0_E * np.random.rand(), V_th_E, t_ref)} for i in range(len(E_pop))])

        T_0_I = t_ref + FPT(tau_I, E_L, I_xI, C_m, V_th_I, V_r)
        if np.isnan(T_0_I):
            T_0_I = 10.
        for I_pop in I_pops:
            nest.SetStatus(I_pop, [{'V_m': V_FPT(
                tau_I, E_L, I_xI, C_m, T_0_I * np.random.rand(), V_th_E, t_ref)} for i in range(len(I_pop))])
    else:
        nest.SetStatus(tuple(range(1, N + 1)),
                       [{'V_m': V_m} for i in range(N)])
    # nest.SetStatus([2],[{'I_e':0.}])
    # define the synapses and connect the populations
    # EE
    j_ee = js[0, 0] / np.sqrt(N)

    nest.CopyModel("static_synapse", "EE_plus", {
                   "weight": jplus[0, 0] * j_ee, "delay": delay})
    nest.CopyModel("static_synapse", "EE_minus", {
                   "weight": jminus[0, 0] * j_ee, "delay": delay})
    if fixed_indegree:
        K_EE = int(ps[0, 0] * int(N_E / Q))
        conn_params_EE = {'rule': 'fixed_indegree',
                          'indegree': K_EE, 'autapses': False, 'multapses': False}
    else:
        conn_params_EE = {'rule': 'pairwise_bernoulli',
                          'p': ps[0, 0], 'autapses': False, 'multapses': False}
    for i, pre in enumerate(E_pops):
        for j, post in enumerate(E_pops):
            if i == j:
                # same cluster
                nest.Connect(pre, post, conn_params_EE, 'EE_plus')
            else:
                nest.Connect(pre, post, conn_params_EE, 'EE_minus')

    # EI
    j_ei = js[0, 1] / np.sqrt(N)

    #nest.CopyModel("vogels_sprekeler_synapse", "EI_plus", {
    #    "weight": j_ei * jplus[0, 1], "delay": delay, "Wmax": -10., 'alpha': 0.3, 'eta': 0.0001})
    #nest.CopyModel("vogels_sprekeler_synapse", "EI_minus", {
    #               "weight": j_ei * jminus[0, 1], "delay": delay, "Wmax": -10., 'alpha': 0.3, 'eta': 0.0001})

    nest.CopyModel("static_synapse", "EI_plus", {
                   "weight": j_ei * jplus[0, 1], "delay": delay})
    nest.CopyModel("static_synapse", "EI_minus", {
                   "weight": j_ei * jminus[0, 1], "delay": delay})
    if fixed_indegree:
        K_EI = int(ps[0, 1] * int(N_I / Q))
        conn_params_EI = {'rule': 'fixed_indegree',
                          'indegree': K_EI, 'autapses': False, 'multapses': False}
    else:
        conn_params_EI = {'rule': 'pairwise_bernoulli',
                          'p': ps[0, 1], 'autapses': False, 'multapses': False}
    for i, pre in enumerate(I_pops):
        for j, post in enumerate(E_pops):
            if i == j:
                # same cluster
                nest.Connect(pre, post, conn_params_EI, 'EI_plus')
            else:
                nest.Connect(pre, post, conn_params_EI, 'EI_minus')

    # IE
    j_ie = js[1, 0] / np.sqrt(N)
    nest.CopyModel("static_synapse", "IE_plus", {
        "weight": j_ie * jplus[1, 0], "delay": delay})
    nest.CopyModel("static_synapse", "IE_minus", {
                   "weight": j_ie * jminus[1, 0], "delay": delay})

    if fixed_indegree:
        K_IE = int(ps[1, 0] * int(N_E / Q))
        conn_params_IE = {'rule': 'fixed_indegree',
                          'indegree': K_IE, 'autapses': False, 'multapses': False}
    else:
        conn_params_IE = {'rule': 'pairwise_bernoulli',
                          'p': ps[1, 0], 'autapses': False, 'multapses': False}
    for i, pre in enumerate(E_pops):
        for j, post in enumerate(I_pops):
            if i == j:
                # same cluster
                nest.Connect(pre, post, conn_params_IE, 'IE_plus')
            else:
                nest.Connect(pre, post, conn_params_IE, 'IE_minus')

    # II
    j_ii = js[1, 1] / np.sqrt(N)

    #nest.CopyModel("vogels_sprekeler_synapse", "II_plus", {
    #               "weight": j_ii * jplus[1, 1], "delay": delay, "Wmax": -10., 'alpha': 0.2})
    #nest.CopyModel("vogels_sprekeler_synapse", "II_minus", {
    #               "weight": j_ii * jminus[1, 1], "delay": delay, "Wmax": -10., 'alpha': 0.2})

    nest.CopyModel("static_synapse", "II_plus", {
                   "weight": j_ii * jplus[1, 1], "delay": delay})
    nest.CopyModel("static_synapse", "II_minus", {
                   "weight": j_ii * jminus[1, 1], "delay": delay})
    if fixed_indegree:
        K_II = int(ps[1, 1] * int(N_I / Q))
        conn_params_II = {'rule': 'fixed_indegree',
                          'indegree': K_II, 'autapses': False, 'multapses': False}
    else:
        conn_params_II = {'rule': 'pairwise_bernoulli',
                          'p': ps[1, 1], 'autapses': False, 'multapses': False}
    for i, pre in enumerate(I_pops):
        for j, post in enumerate(I_pops):
            if i == j:
                # same cluster
                nest.Connect(pre, post, conn_params_II, 'II_plus')
            else:
                nest.Connect(pre, post, conn_params_II, 'II_minus')

    # set up spike detector
    #spike_detector = nest.Create("spike_detector")
    #nest.SetStatus(spike_detector, [
    #    {'to_file': True, 'withtime': True, 'withgid': True, 'label': fn_sd, 'use_gid_in_filename': False}])
    all_units = ()
    for E_pop in E_pops:
        all_units += E_pop
    for I_pop in I_pops:
        all_units += I_pop
    #nest.Connect(all_units, spike_detector, syn_spec='EE_plus')

    # set up stimulation
    if stim_clusters is not None:
        current_source = nest.Create('step_current_generator')
        amplitude_values = []
        amplitude_times = []
        for start, end in zip(stim_starts, stim_ends):
            amplitude_times.append(start + warmup)
            amplitude_values.append(stim_amp)
            amplitude_times.append(end + warmup)
            amplitude_values.append(0.)
        nest.SetStatus(current_source, {
                       'amplitude_times': amplitude_times, 'amplitude_values': amplitude_values})
        stim_units = []
        for stim_cluster in stim_clusters:
            stim_units += list(E_pops[stim_cluster])
        nest.Connect(current_source, stim_units)

    elif multi_stim_clusters is not None:
        for stim_clusters, amplitudes, times in zip(multi_stim_clusters, multi_stim_amps, multi_stim_times):
            current_source = nest.Create('step_current_generator')
            nest.SetStatus(current_source, {
                           'amplitude_times': times, 'amplitude_values': amplitudes})
            stim_units = []
            for stim_cluster in stim_clusters:
                stim_units += list(E_pops[stim_cluster])
            nest.Connect(current_source, stim_units)

    # set up multimeter if necessary
    if record_voltage:
        recordables = params.get(
            'recordables', [str(r) for r in nest.GetStatus(E_pops[0], 'recordables')[0]])
        voltage_recorder = nest.Create('multimeter', params={
                                       'record_from': recordables, 'interval': recording_interval})
        if record_from != 'all':
            record_units = []
            for E_pop in E_pops:
                record_units += list(E_pop[:record_from])
            for I_pop in I_pops:
                record_units += list(I_pop[:record_from])

        else:
            record_units = [u for u in all_units]
        nest.Connect(voltage_recorder, record_units)

    endbuild = time.time()

    return all_units, E_pops, I_pops

def get_results(params, spike_detector, voltage_recorder, all_units, record_units, E_pops, I_pops, recordables):
    #nest.Simulate(warmup + simtime)
    endsim = time.time()


    # simulation parameters
    dt = params.get('dt', 0.1)   # integration time step
    simtime = params.get('simtime', 1000.)  # simulation time
    warmup = params.get('warmup', 0)
    record_voltage = params.get('record_voltage', False)
    record_from = params.get('record_from', 'all')
    recording_interval = params.get('recording_interval', dt)
    return_weights = params.get('return_weights', False)
    n_jobs = params.get('n_jobs', 1)

    # excitatory units
    N_E = params.get('N_E', default.N_E)
    # inhibitory units
    N_I = params.get('N_I', default.N_I)

    # neuron parameters
    neuron_type = params.get('neuron_type', default.neuron_type)
    E_L = params.get('E_L', default.E_L)          # resting potential
    C_m = params.get('C_m', default.C_m)          # membrane capacitance
    # excitatory membrane time constant
    tau_E = params.get('tau_E', default.tau_E)
    # inhibitory membrane time constant
    tau_I = params.get('tau_I', default.tau_I)
    t_ref = params.get('t_ref', default.t_ref)      # refractory period
    V_th_E = params.get('V_th_E', default.V_th_E)   # threshold voltage
    V_th_I = params.get('V_th_I', default.V_th_E)
    V_r = params.get('V_r', default.V_r)          # reset voltage
    I_th_E = params.get('I_th_E', default.I_th_E)
    I_th_I = params.get('I_th_I', default.I_th_I)





    # get the spiketimes from the detector
    events = nest.GetStatus(spike_detector, 'events')[0]
    # convert them to the format accepted by spiketools
    spiketimes = np.append(
        events['times'][None, :], events['senders'][None, :], axis=0)
    spiketimes[1] -= 1
    # remove the pre warmup spikes
    spiketimes = spiketimes[:, spiketimes[0] >= warmup]

    spiketimes[0] -= warmup

    results = {'spiketimes': spiketimes}
    if record_voltage:


        events = nest.GetStatus(voltage_recorder, 'events')[0]

        times = events['times']
        senders = events['senders']
        usenders = np.unique(senders)
        sender_ind_dict = {s: record_units.index(s) for s in usenders}
        sender_inds = [sender_ind_dict[s] for s in senders]

        utimes = np.unique(times)
        time_ind_dict = {t: i for i, t in enumerate(utimes)}
        time_inds = [time_ind_dict[t] for t in times]

        if record_from == 'all':
            n_records = N
        else:
            n_records = record_from * (len(E_pops) + len(I_pops))
        for recordable in recordables:
            t0 = time.time()

            results[recordable] = np.zeros((n_records, len(utimes)))
            results[recordable][sender_inds, time_inds] = events[recordable]

            results[recordable] = results[recordable][:, utimes >= warmup]

        utimes = utimes[utimes >= warmup]
        utimes -= warmup
        results['senders'] = np.array(record_units)
        results['times'] = utimes

    if return_weights:

        connections = nest.GetConnections(all_units, all_units)
        weights = nest.GetStatus(connections, keys='weight')
        pre = [c[0] - 1 for c in connections]
        post = [c[1] - 1 for c in connections]
        weight_mat = np.zeros((N, N))
        weight_mat[post, pre] = weights
        results['weights'] = weight_mat

    e_count = spiketimes[:, spiketimes[1] < N_E].shape[1]
    i_count = spiketimes[:, spiketimes[1] >= N_E].shape[1]
    e_rate = e_count / float(N_E) / float(max(spiketimes[0])) * 1000.
    i_rate = i_count / float(N_I) / float(max(spiketimes[0])) * 1000.

    if I_th_E is None:
        I_xE = params.get('I_xE', default.I_xE)
    else:
        I_xE = I_th_E * (V_th_E - E_L) / tau_E * C_m


    if I_th_I is None:
        I_xI = params.get('I_xI', default.I_xI)
    else:
        I_xI = I_th_I * (V_th_I - E_L) / tau_I * C_m


    results['e_rate'] = e_rate
    results['i_rate'] = i_rate
    results['I_xE'] = I_xE
    results['I_xI'] = I_xI
    endpost = time.time()

    return results


#if __name__ == '__main__':
#    js_maz = pylab.array([[1.77, -3.18], [1.06, -4.24]])
#
#    params = {'n_jobs': 4, 'N_E': 4000, 'N_I': 1000,
#              'dt': 0.1, 'neuron_type': 'iaf_psc_exp',
#              'simtime': 500, 'delta_I_xE': 0., 'delta_I_xI': 0.,
#              'record_voltage': True, 'record_from': 1, 'warmup': 0}
#
#    N = params['N_E'] + params['N_I']
#
#    I_ths = [2.13, 1.24]  # 3,5,Hz
#    # I_ths = [5.34,2.61] # 10,15,Hz
#    params['I_th_E'] = I_ths[0]
#    params['I_th_I'] = I_ths[1]
#
#    params['Q'] = 1
#    jip_ratio = 0.75
#    jep = 5.
#    jip = 1. + (jep - 1) * jip_ratio
#    params['jplus'] = pylab.array([[jep, jip], [jip, jip]])
#
#    results = simulate(params)
#
#    spiketimes = results['spiketimes']
#    pylab.subplot(1, 1, 1)
#    pylab.plot(spiketimes[0], spiketimes[1], '.k', markersize=1)
#    pylab.title(str(results['e_rate']) + ', ' + str(results['i_rate']))
#
#    pylab.figure()
#    E_inds = pylab.find(results['senders'] < params['N_E'])
#    I_inds = pylab.find(results['senders'] >= params['N_E'])
#    pylab.subplot(2, 1, 1)
#    pylab.plot(results['times'], results['V_m'][E_inds].T)
#    pylab.subplot(2, 1, 2)
#    pylab.plot(results['times'], results['V_m'][I_inds].T)
#
#    pylab.show()
