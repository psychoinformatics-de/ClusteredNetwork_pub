class SuperPopulation:
    def __init__(self, Populations: [], name: str):
        self.Populations = Populations  
        self.name=name

    def get_Populations(self):
        return self.Populations
    
    def get_name(self):
      return self.name

    def set_global_Param(self, GlobalParam, Value):
        for pop in self.Populations:
            pop.extra_global_params[GlobalParam].view[:]=Value
            pop.push_extra_global_param_to_device(GlobalParam)


def prepareModelForLoad(model):
    GlobalParams=False
    for popN in model.neuron_populations:         
        pop=model.neuron_populations[popN]
        try:
            popGlobVar = list(pop.extra_global_params.keys())
            for globPar in popGlobVar:         
                pop.set_extra_global_param(globPar, [0.0 for ii in range(model.batch_size)])
            GlobalParams=True
        except:
            print(pop.name + "has no global parameter")        
    return GlobalParams


# has to be changed to deal with batch_size 1 and >1
def extractPopSpikes(num, NQ, Pop, batch_size, NOffset=0, warmup=0, timeZero=0):
    DownloadedData = Pop.spike_recording_data
    spiketimes=[np.vstack(
        (np.array(DownloadedData[jj][0][DownloadedData[jj][0]>= (warmup+timeZero)])-(warmup+timeZero),
         np.array(DownloadedData[jj][1][DownloadedData[jj][0]>= (warmup+timeZero)]) + num * NQ + NOffset)) for jj in range(batch_size)]
    return spiketimes




# Differences in last part have to be solved
def extractResults(results, model,params, E_pops, I_pops, IDMap=None ,timeZero=0):
    ## TO-DO: Change to accept a general number of Populations
    ## To-DO: Change to reveive a function which is then executed as quick results - + names for each Population measure (in our case e_rate + i_rate)
    
    # network parameters - Change to extraction from Populations
    N_E = params.get('N_E', default.N_E)  # excitatory units
    N_I = params.get('N_I', default.N_I)  # inhibitory units
    Q = params.get('Q', default.Q)  # number of clusters


    if IDMap is None:
        batch_size=model.batch_size
    else:
        batch_size=len(IDMap)

    warmup = params.get('warmup', 0)
    simtime = params.get('simtime', 1000.)  # simulation time

    # Download recording data
    model.pull_recording_buffers_from_device()

    with Pool(params[n_jobs]) as p:
        resE=p.map(lambda x: extractPopSpikes(x[0], int(N_E/Q), x[1], batch_size, 0,warmup,timeZero), enumerate(E_pops.get_Populations()))
        resI=p.map(lambda x: extractPopSpikes(x[0], int(N_I/Q), x[1], batch_size, N_E,warmup,timeZero), enumerate(I_pops.get_Populations()))
   
    TuplesConcat=[tuple(res[jj] for res in resE+resI) for jj in range(batch_size)] 
    spiketimesLoc=[np.hstack(TuplesConcat[jj]) for jj in range(batch_size)]

    #sortstart=time.time()
    spiketimesLoc=[spiketimesLoc[jj][:, spiketimesLoc[jj][0,:].argsort()] for jj in range(batch_size)]

    for jj in range(batch_size):
        e_count = spiketimesLoc[jj][:, spiketimesLoc[jj][1] < N_E].shape[1]
        i_count = spiketimesLoc[jj][:, spiketimesLoc[jj][1] >= N_E].shape[1]
        e_rate = e_count / float(N_E) / float(simtime) * 1000.
        i_rate = i_count / float(N_I) / float(simtime) * 1000.
        if IDMap is None:
            results.append({'spiketimes': spiketimesLoc[jj].copy(), 'e_rate': e_rate, 'i_rate': i_rate, 'ID': jj})#, 'I_xE': I_xE, 'I_xI': I_xI})
        else:
            results.append({'spiketimes': spiketimesLoc[jj].copy(), 'e_rate': e_rate, 'i_rate': i_rate, 'ID': IDMap[jj]})#, 'I_xE': I_xE, 'I_xI': I_xI})
    return results