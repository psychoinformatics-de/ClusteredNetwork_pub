import sys;sys.path.append('../utils/')
#import sim_nest
import pylab
import plotting_functions as plotting
import spiketools
import organiser
import default
from copy import deepcopy
from bisect import bisect_right
import global_params
from matplotlib.ticker import MaxNLocator
from general_func import *
from Helper import ClusterModelNEST
from Defaults import defaultSimulate as default
import pickle
import pandas as pd


datapath = '../data/'
datafile = 'fig02_cluster_dynamics_new'
datafile1 = 'ff_cv2_spontaneous_new'

def simulate_spontaneous(params):
    pylab.seed()
    trials = params['trials']
    trial_length = params['trial_length']
    sim_params = deepcopy(params)
    sim_params['simtime'] = trials*trial_length
    ff_window = params['ff_window']
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, sim_params)
    # Creates object which creates the EI clustered network in NEST
    result = EI_Network.get_simulation() 
    long_spiketimes = result['spiketimes']
    order = pylab.argsort(long_spiketimes[0])
    long_spiketimes = long_spiketimes[:,order]
    # cut into trial pieces
    spiketimes = pylab.zeros((3,0))
    trial_start = 0
    
    for trial in range(trials):

        trial_end = bisect_right(long_spiketimes[0], trial_length)
        trial_spikes = long_spiketimes[:,:trial_end].copy()
        long_spiketimes = long_spiketimes[:,trial_end:]
        trial_spikes = pylab.concatenate([trial_spikes[[0],:],pylab.ones((1,trial_spikes.shape[1]))*trial,trial_spikes[[1],:]],axis=0)
        spiketimes = pylab.append(spiketimes, trial_spikes,axis=1)
        long_spiketimes[0]-= trial_length

    order = pylab.argsort(spiketimes[2])
    spiketimes = spiketimes[:,order]
    N_E = params.get('N_E',default.N_E)
    ffs = []
    cv2s = []
    counts = []
    for unit in range(N_E):
        unit_end = bisect_right(spiketimes[2], unit)
        
        unit_spikes = spiketimes[:2,:unit_end]
        spiketimes = spiketimes[:,unit_end:]
        counts.append(unit_spikes.shape[1])
        if unit_spikes.shape[1]>0:
            window_ffs = []
            tlim = pylab.array([0,ff_window])
            while tlim[0]<trial_length:
                window_ffs.append(spiketools.ff(unit_spikes,tlim = tlim))
                tlim+=ff_window
            ffs.append(pylab.nanmean(window_ffs))
            cv2s.append(spiketools.cv2(unit_spikes,pool = False))
        else:
            ffs.append(pylab.nan)
            cv2s.append(pylab.nan)
    print('ff',pylab.nanmean(ffs))
    print('cv2',pylab.nanmean(cv2s))
    return pylab.nanmean(ffs),pylab.nanmean(cv2s),pylab.nanmean(counts)

def get_spikes_fig2(params):
    EI_Network = ClusterModelNEST.ClusteredNetwork(default, params)
    # Creates object which creates the EI clustered network in NEST
    result = EI_Network.get_simulation() 
    return result

        
    
def plot_ff_jep_vs_Q_Litwin(params,jep_range=pylab.linspace(1,4,41),
                            Q_range = pylab.arange(2,20,2),jipfactor = 1,
                            reps = 40,plot = True,vrange = [0,15],redo = False):
    
    
    try:
        ffs = pd.read_pickle(datapath + "ffs_supplyfig1")
    except:
        ffs = pylab.zeros((len(jep_range),len(Q_range),reps))
        counts = pylab.zeros((len(jep_range),len(Q_range),reps))
        cvs = pylab.zeros((len(jep_range),len(Q_range),reps))
        for i,jep_ in enumerate(jep_range):
            print(jep_,'-------------------------------------------------------------------------')
            for j,Q in enumerate(Q_range):
                jep = float(min(jep_,Q))
                if jipfactor == 0.:
                    params['portion_I'] = Q
                else:
                    params['portion_I'] = 1
                jip = 1. +(jep-1)*jipfactor

                params['jplus'] = pylab.around(pylab.array([[jep,1.0],[jip,1.0]]),5)
                params['Q'] = int(Q)
                # adjust for devisable N and Q
                params['N_E'] = default.N_E - default.N_E%params['Q']
                params['N_I'] = default.N_I - default.N_I%params['Q']
                results = organiser.check_and_execute(params, simulate_spontaneous, datafile1,
                                    reps = reps,ignore_keys=['n_jobs'],redo = redo)
                ff = [r[0] for r in results]
                count=[r[2] for r in results]
                cv=[r[1] for r in results]

                counts[i,j,:] = count
                cvs[i,j,:] = cv
                ffs[i,j,:] = ff

                
                if jep_>Q:
                    ffs[i,j,:] = pylab.nan
        pickle.dump(ffs,open(datapath + "ffs_supplyfig1",'wb'))


    if plot:
        print(pylab.nanmean(ffs,axis=2).T)
        pylab.contourf(jep_range,Q_range,pylab.nanmean(ffs,axis=2).T,
                       levels = [0.5, 1.,1.5,2.],extend = 'both',cmap = 'Greys')
        x = pylab.linspace(Q_range.min(), jep_range.max(),1000)
        y1 = pylab.ones_like(x)*Q_range.min()
        y2 = x
        pylab.fill_between(x,y1, y2,facecolor = 'w',hatch = '\\\\\\',edgecolor = global_params.colors['orange'])
        pylab.xlabel('$J_{E+}$',size = 14)
        pylab.ylabel('$Q$', size = 14)
        pylab.axis('tight')


             
if __name__ == '__main__':
    
    n_jobs = 22
    settings = [{'jipfactor':0.75,'fixed_indegree':False, 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                    'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26}]  #3,5  hz
    
    plot = True
    reps = 20
    x_label_val = -0.25
    num_row, num_col = 1,1
    if plot:
        fig  =plotting.nice_figure(ratio = 0.8,latex_page=global_params.text_width_pts)
        fig.subplots_adjust(bottom = 0.15,hspace = 0.4,wspace = 0.3)

    for i,params in enumerate(settings):
        row = 0
        col= 0
        if True:
            jipfactor = params.pop('jipfactor')
            jep_step = 0.5
            jep_range = pylab.arange(1.,15.+0.5*jep_step,jep_step)
            q_step = 1
            Q_range = pylab.arange(q_step,60+0.5*q_step,q_step)
            
            if plot:
                ax = plotting.simpleaxis(pylab.subplot2grid((num_row,num_col),(row, col)),labelsize = 10)          
                #plotting.ax_label1(ax, labels[i], x=x_label_val)
            plot_ff_jep_vs_Q_Litwin(params,jep_range,Q_range,jipfactor,plot=plot, redo=False)
            if plot:
                cbar = pylab.colorbar()
                cbar.set_label('FF', rotation=90,size = 14)
    pylab.savefig('suppl_fig1.pdf')
    #pylab.savefig('suppl_fig1.png', dpi=300)    
    pylab.show()



