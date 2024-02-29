import organiser

organiser.datapath = '../data/'
datafile = 'model_stimulation_test'




def get_analysed_spiketimes(params,window=400,calc_cv_twos=True,
                                save =False,do_not_simulate=False):
    """get the analysed spiketimes for the given params and window"""
    params = {'sim_params':deepcopy(params),'window':window,'calc_cv_twos':calc_cv_twos}
    #spiketimes = get_spiketimes(params['sim_params'],fname = datafile,save = save)
    if do_not_simulate:
        all_results = pd.read_pickle(os.path.join(organiser.datapath,datafile) + '_analyses')
        key_list = [k for k in sorted(params.keys())]
        key = key_from_params(params,key_list)
        results = all_results[key]

    else:
        result =  organiser.check_and_execute(params,_simulate_analyse,
                                            datafile +'_analyses',
                                          ignore_keys=['n_jobs'],
                                          redo=False, save = save)
    return result


def split_unit_spiketimes(spiketimes,N = None):
    """ splits a spiketimes array containig [spiketimes,trial,unit] into a dictionary
        where keys are unit numbers and entries contain corresponding spiketimes.
        N is the expected total number of units
        """
    trials = pylab.unique(spiketimes[1])
    spike_dict = {}

    order = pylab.argsort(spiketimes[2])
    spiketimes = spiketimes[:,order]
    
    if N is None:
        units = pylab.unique(spiketimes[2])
    else:
        units = pylab.arange(N)

    for unit in units:
        unit_end = bisect_right(spiketimes[2], unit)
        if unit_end>0:
            spike_dict[unit] = spiketimes[:2,:unit_end]
        else:
            spike_dict[unit] = pylab.zeros((2,0))
        missing_trials = list(set(trials).difference(spike_dict[unit][1,:]))
        
        for mt in missing_trials:
            spike_dict[unit] = pylab.append(spike_dict[unit], pylab.array([[pylab.nan],[mt]]),axis=1)
        spiketimes = spiketimes[:,unit_end:]

    return spike_dict

def get_spiketimes(params,save = True,fname = datafile):
    """get the spiketimes for the given params"""
    return organiser.check_and_execute(
        params,_simulate_stimulate,fname +'_spiketimes',
        ignore_keys=['n_jobs'],save = save)


def _simulate_analyse(params):
    """simulate and analyse the spiketimes"""
    pylab.seed(None)
    sim_params = deepcopy(params['sim_params'])
    try:
        sim_params['n_jobs'] = params['n_jobs']
    except:
        pass
    n_jobs = sim_params['n_jobs']
    save_spiketimes = params.get('save_spiketimes',True)
    if save_spiketimes is False and 'randseed' not in list(sim_params.keys()):
        sim_params['randseed'] = pylab.randint(0,1000000,1)[0]
        print('randseed:' ,sim_params['randseed'])
        fname = 'model_stimulation_no_save'
    else:
        fname = datafile
    spiketimes = get_spiketimes(sim_params,save = save_spiketimes,fname = fname)
    N_E = params['sim_params'].get('N_E',4000)
    Q = params['sim_params'].get('Q',1)
    tlim = params['sim_params']['cut_window']
    unit_spiketimes = split_unit_spiketimes(spiketimes,N = N_E)
    
    cluster_size = int(N_E/Q)
    cluster_inds = [list(range(i*cluster_size,(i+1)*cluster_size)) for i in range(Q)]


    ff_result = Parallel(n_jobs,verbose = 2)(
        delayed(spiketools.kernel_fano)(
            unit_spiketimes[u],window = params['window'],tlim = tlim) for u in list(unit_spiketimes.keys()))
    all_ffs = pylab.array([r[0] for r in ff_result])
    cluster_ffs = pylab.array([pylab.nanmean(all_ffs[ci],axis=0) for ci in cluster_inds])
    results = {'ffs':cluster_ffs,'t_ff':ff_result[0][1]}
    # mean matched ff
    var_mean_result = Parallel(n_jobs,verbose = 2)(delayed(spiketools.kernel_fano)(
        unit_spiketimes[u],window = params['window'],
        tlim = tlim, components=True) for u in list(unit_spiketimes.keys()))
    all_var = pylab.array([r[0] for r in var_mean_result])
    all_mean = pylab.array([r[1] for r in var_mean_result])
    #cluster_ffs = pylab.array([pylab.nanmean(all_ffs[ci],axis=0) for ci in inds])
    for cnt, ci in enumerate(cluster_inds):
        mask = all_mean[ci] < np.max(all_mean[ci,:300])
        ff_all = (all_var[ci]*mask)/(all_mean[ci]*mask)
        if cnt == 0:
            cluster_ff_mm = np.nanmean(ff_all, 0)
        else:
            cluster_ff_mm = np.vstack((cluster_ff_mm, np.nanmean(ff_all, 0) ))

    results.update({'ff_mm':cluster_ff_mm, 
                        'time':ff_result[0][1]})


    cv_two_result = Parallel(n_jobs,verbose = 2)(
        delayed(spiketools.time_resolved_cv_two)(
            unit_spiketimes[u],window = params['window'],tlim = tlim,min_vals = params['sim_params']['min_vals_cv_two']) for u in list(unit_spiketimes.keys()))
    all_cv_twos = pylab.array([r[0] for r in cv_two_result])
    cluster_cv_twos = pylab.array([pylab.nanmean(all_cv_twos[ci],axis=0) for ci in cluster_inds])
    results.update({'cv_twos':cluster_cv_twos,'t_cv_two':cv_two_result[0][1]})
   
    
    kernel = spiketools.triangular_kernel(sigma=params['sim_params']['rate_kernel'])
    rate_result = Parallel(n_jobs,verbose = 2)(
        delayed(spiketools.kernel_rate)(
            unit_spiketimes[u],kernel = kernel,tlim = tlim) for u in list(unit_spiketimes.keys()))
    all_rates = pylab.array([r[0][0] for r in rate_result])
    cluster_rates = pylab.array([pylab.nanmean(all_rates[ci],axis=0) for ci in cluster_inds])
    results.update({'rates':cluster_rates,'t_rate':rate_result[0][1]})
    

    return results