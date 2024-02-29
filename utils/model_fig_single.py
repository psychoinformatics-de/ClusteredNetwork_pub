#import sys;sys.path += ['..'];sys.path.append('../tuning');sys.path.append('../monkey_dat')
import matplotlib;matplotlib.use('Agg')

import plotting_functions as plotting
import organiser
from organiser import *
from copy import deepcopy
import pylab
import numpy as np
import spiketools
from sim_nest import simulate

#from Helper import ClusterModelNEST
#from Defaults import defaultSimulate as default

import analyse_nest
import default
import cv_bias
from joblib import Parallel,delayed
#from fig04_cvs import do_plot as do_cv_plot
import os
#import input_currents
#from bisect import bisect_right
#import tune_jep
#import data_ff_cv2
import global_params
import pandas as pd

organiser.datapath = '../data/'
datafile = 'model_stimulation_test'



def _simulate_stimulate(original_params):
    params = deepcopy(original_params)
    pylab.seed(params.get('randseed',None))

    # make stimulus start and end lists

    stim_starts = [float(params.get('warmup',0)+params['isi'])]
    stim_ends = [stim_starts[-1] +params['stim_length']]

    for i in range(params['trials']-1):
        stim_starts += [stim_ends[-1]+params['isi']+np.int16(pylab.rand()*params['isi_vari'])]
        stim_ends += [stim_starts[-1] +params['stim_length']]

    params['stim_starts'] = stim_starts
    params['stim_ends'] = stim_ends
     
    #print stim_starts
    #print stim_ends
    params['simtime'] = int(max(stim_ends)+params['isi'])
    print('SIMTIME', params['simtime'])
    jep = params['jep']
    jipfactor = params['jipfactor']
    jip = 1. +(jep-1)*jipfactor
    params['jplus'] = pylab.around(pylab.array([[jep,jip],[jip,jip]]),5)
    
    
    # remove all params not relevant to simulation
    sim_params = deepcopy(params)
    drop_keys = ['cut_window','ff_window','cv_ot','stim_length','isi','isi_vari','rate_kernel','jipfactor','jep','trials','min_count_rate','pre_ff_only','ff_only']
    for dk in drop_keys:
        try:
            sim_params.pop(dk)
        except:
            pass
    

    # sim_result = organiser.check_and_execute(sim_params,simulate,datafile+'_spiketimes')
    print('simparamssss', sim_params)
    #EI_Network = ClusterModelNEST.ClusteredNetwork(default, sim_params)
    # Creates object which creates the EI clustered network in NEST
    #sim_result = EI_Network.get_simulation() 
    sim_result = simulate(sim_params)
    spiketimes =  sim_result['spiketimes']
    
    trial_spiketimes = analyse_nest.cut_trials(spiketimes, 
                                    stim_starts, params['cut_window'])
    
    return trial_spiketimes
    

def get_spiketimes(params,save = True,fname = datafile):
    
    return organiser.check_and_execute(params,_simulate_stimulate,
                                       fname +'_spiketimes',ignore_keys=['n_jobs'],
                                       save = save)


def _simulate_analyse(params):
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
    unit_spiketimes = analyse_nest.split_unit_spiketimes(spiketimes,N = N_E)
    
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
            unit_spiketimes[u],window = params['window'],
            tlim = tlim,
            min_vals = params['sim_params']['min_vals_cv2']) for u in list(unit_spiketimes.keys()))
    all_cv2s = pylab.array([r[0] for r in cv_two_result])
    cluster_cv2s = pylab.array([pylab.nanmean(all_cv2s[ci],axis=0) for ci in cluster_inds])
    results.update({'cv2s':cluster_cv2s,'t_cv2':cv_two_result[0][1]})
   
    
    kernel = spiketools.triangular_kernel(sigma=params['sim_params']['rate_kernel'])
    rate_result = Parallel(n_jobs,verbose = 2)(
        delayed(spiketools.kernel_rate)(
            unit_spiketimes[u],kernel = kernel,tlim = tlim) for u in list(unit_spiketimes.keys()))
    all_rates = pylab.array([r[0][0] for r in rate_result])
    cluster_rates = pylab.array([pylab.nanmean(all_rates[ci],axis=0) for ci in cluster_inds])
    results.update({'rates':cluster_rates,'t_rate':rate_result[0][1]})
    

    return results



def _simulate_analyse_subset(params):
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
    #unit_spiketimes = analyse_nest.split_unit_spiketimes(spiketimes,N = N_E)
    unit_spiketimes = analyse_nest.split_unit_spiketimes(spiketimes,N = N_E + 1000)
    
    #cluster_size = int(N_E/Q)
    #cluster_inds = [range(i*cluster_size,(i+1)*cluster_size) for i in range(Q)]
    pylab.seed(0)
    inds = [i for i in pylab.randint(0,5000,100)]

    ff_result = Parallel(n_jobs,verbose = 2)(delayed(spiketools.kernel_fano)(
        unit_spiketimes[u],window = params['window'],tlim = tlim) for u in list(unit_spiketimes.keys()))
    all_ffs = pylab.array([r[0] for r in ff_result])
    #cluster_ffs = pylab.array([pylab.nanmean(all_ffs[ci],axis=0) for ci in inds])
    cluster_ffs = pylab.array([all_ffs[ci] for ci in inds])
    results = {'ffs':cluster_ffs,'t_ff':ff_result[0][1]}
    ff_mm_result = Parallel(n_jobs,verbose = 2)(delayed(spiketools.kernel_fano)(
        unit_spiketimes[u],window = params['window'],tlim = tlim, mean_matched=True) for u in list(unit_spiketimes.keys()))
    all_ff_mms = pylab.array([r[0] for r in ff_mm_result])
    #cluster_ffs = pylab.array([pylab.nanmean(all_ffs[ci],axis=0) for ci in inds])
    cluster_ff_mms = pylab.array([all_ff_mms[ci] for ci in inds])
    results.update({'ff_mms':cluster_ff_mms,'t_ff_mm':ff_mm_result[0][1]})

    cv_two_result = Parallel(n_jobs,verbose = 2)(delayed(spiketools.time_resolved_cv_two)(unit_spiketimes[u],window = params['window'],tlim = tlim,min_vals = params['sim_params']['min_vals_cv2']) for u in list(unit_spiketimes.keys()))
    all_cv2s = pylab.array([r[0] for r in cv_two_result])
    #cluster_cv2s = pylab.array([pylab.nanmean(all_cv2s[ci],axis=0) for ci in inds])
    cluster_cv2s = pylab.array([all_cv2s[ci] for ci in inds])
    results.update({'cv2s':cluster_cv2s,'t_cv2':cv_two_result[0][1]})
   
    
    kernel = spiketools.triangular_kernel(sigma=params['sim_params']['rate_kernel'])
    rate_result = Parallel(n_jobs,verbose = 2)(delayed(spiketools.kernel_rate)(unit_spiketimes[u],kernel = kernel,tlim = tlim) for u in list(unit_spiketimes.keys()))
    all_rates = pylab.array([r[0][0] for r in rate_result])
    #cluster_rates = pylab.array([pylab.nanmean(all_rates[ci],axis=0) for ci in inds])
    cluster_rates = pylab.array([all_rates[ci] for ci in inds])
    results.update({'rates':cluster_rates,'t_rate':rate_result[0][1]})
    

    return results

def get_analysed_spiketimes(params,window=400,calc_cv2s=True,
                                save =False,do_not_simulate=False):
    params = {'sim_params':deepcopy(params),'window':window,'calc_cv2s':calc_cv2s}
    #spiketimes = get_spiketimes(params['sim_params'],fname = datafile,save = save)
    if do_not_simulate:
        all_results = pd.read_pickle(os.path.join(organiser.datapath,datafile) + '_analyses')
        key_list = [k for k in sorted(params.keys())]
        key = key_from_params(params,key_list)
        results = all_results[key]
        #result_keys =  [key+'_'+str(r) for r in range(reps)]
        #results = [all_results[result_key] for result_key in result_keys]

    else:
        result =  organiser.check_and_execute(params,_simulate_analyse,
                                            datafile +'_analyses',
                                          ignore_keys=['n_jobs'],
                                          redo=False, save = save)
    #result =  organiser.check_and_execute(params,_simulate_analyse,datafile +'_analyses',ignore_keys=['n_jobs'],save = save)
    #result['spiketimes'] = spiketimes
    #result['cluster_inds'] = analyse_nest.get_cluster_inds(params['sim_params'])
    return result



    
def make_plot(params,axes = None,plot = True,ff_plotargs={},cvtwo_plotargs = {},calc_cv2s = True,t_offset  =0,save= True,split_ff_clusters = False,split_cv2_clusters = False):
    result = get_analysed_spiketimes(params,calc_cv2s=calc_cv2s,save =save)
    stim_clusters = params['stim_clusters']
    
    non_stim_clusters = [i for i in range(params['Q']) if i not in stim_clusters]
    #print result['cluster_inds']
    #
    if axes == None:
        pylab.plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'],axis=0),**ff_plotargs)
        if split_ff_clusters:
            pylab.plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'][stim_clusters],axis=0),linestyle = '--',**ff_plotargs)
            pylab.plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'][non_stim_clusters],axis=0),linestyle = ':',**ff_plotargs)
        if calc_cv2s:
            pylab.plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'],axis=0),label = 'all',**cvtwo_plotargs)
            if split_cv2_clusters:
                pylab.plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'][stim_clusters],axis=0),linestyle = '--',label = 'stim',**cvtwo_plotargs)
                pylab.plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'][non_stim_clusters],axis=0),linestyle = ':',label = 'non stim',**cvtwo_plotargs)

        return result
def make_plot_ff_cv2(params,axes = None,plot = True,ff_plotargs={},cvtwo_plotargs = {},calc_cv2s = True,t_offset  =0,save= False,split_ff_clusters = False,split_cv2_clusters = False, ylim_ff=[0.,2.5], ylim_cv2 = [0.,1.3], xlim = [0,2000]):
    result = get_analysed_spiketimes(params,calc_cv2s=calc_cv2s,save =save,do_not_simulate=True)
    print('got the resultssss')
    stim_clusters = params['stim_clusters']
    non_stim_clusters = [i for i in range(params['Q']) if i not in stim_clusters]
    axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'][stim_clusters],axis=0),**ff_plotargs)
    axes[0].set_ylim(ylim_ff)
    axes[0].set_xlim(xlim)    
    
    #axes.plot(result['t_rate']+t_offset,pylab.nanmean(result['rates'],axis=0),**ff_plotargs)        
    if split_ff_clusters:
        axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'][stim_clusters],axis=0),linestyle = '--',**ff_plotargs)
        axes[0].plot(result['t_ff']+t_offset,pylab.nanmean(result['ffs'][non_stim_clusters],axis=0),linestyle = ':',**ff_plotargs)
    if calc_cv2s:
        axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'][stim_clusters],axis=0),label = 'all',**cvtwo_plotargs)
        axes[1].set_ylim(ylim_cv2)
        axes[1].set_xlim(xlim)            
        if split_cv2_clusters:
            axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'][stim_clusters],axis=0),linestyle = '--',label = 'stim',**cvtwo_plotargs)
            axes[1].plot(result['t_cv2']+t_offset,pylab.nanmean(result['cv2s'][non_stim_clusters],axis=0),linestyle = ':',label = 'non stim',**cvtwo_plotargs)

        return result

"""
if __name__ == '__main__':

    params = {'Q':20,'N_E':4000,'N_I':1000,'I_th_E':2.14,'I_th_I':1.26,'ff_window':400,'min_vals_cv2':1,
              'stim_length':1000,'isi':1000,'isi_vari':200,'cut_window':[-500,1500],'rate_kernel':50.,'warmup':2000,
              'trials':20}

    
    n_jobs = 20
    save = True
    plot = True
    redo  =False
    trials = 100
    
    params['n_jobs'] = n_jobs
    
    target_ff = round(pylab.nanmean(data_ff_cv2.get_pre_post_ff()[0]),2)
    target_rates = [3.,5.]
    print('target ff', target_ff)


    settings = [{'randseed':0,'Q':50,'jipfactor':0.75,'jep':None,'stim_clusters':None,'stim_amp':0.31, 'portion_I':1}]
    
    
   # datafile += '_experiments'
  #  randseeds = pylab.arange(50)
 #   jeps = pylab.arange(3.7,5,0.1)
#    stim_amps = pylab.arange(0.29,0.32,0.01)
   # new_settings = []
  #  for setting in settings:
 #       for randseed in randseeds:
#            for stim_amp in stim_amps:
         #       for jep in jeps:
        #            new_settings.append(deepcopy(setting))
       #             #new_settings[-1]['randseed'] = randseed
      #              #new_settings[-1]['jep'] = jep
     #               new_settings[-1]['stim_amp'] = stim_amp


    
    #settings = new_settings
    #pylab.shuffle(settings)
    
    params['fixed_indegree'] = False
    params['trials'] = trials
   
    plot_cv2s = True
    
    for setno,setting in enumerate(settings):
        print(setting)

        plotting.nice_figure(latex_page=global_params.text_width_pts)

        for k in list(setting.keys()):
            params[k] = setting[k]
        
        if params['stim_clusters'] is None:
            params['stim_clusters'] = list(range(int(params['Q']/10.)))
        
        if params['jep'] is None:
            print('JA!')
            jep,ff = tune_jep.tune_jep(params,target_ff,jipfactor=params['jipfactor'],min_jep=7.,max_jep=9., reps = 1)
            print('JEP FF', jep,ff)
            params['jep'] = jep
        

    
        
        make_plot(params,save = save)
        fname = 'single_'+str(params['Q'])+'_' +str(params['jep'])+'_'+str(params['jipfactor'])+'_'+str(params['stim_amp'])+'_'+str(params['trials'])+'_'+str(params['randseed'])
        pylab.savefig(fname+'.png',dpi  =300)
        

    pylab.show()
"""        
