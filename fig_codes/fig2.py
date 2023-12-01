import sys;sys.path.append('../utils/')
#import sim_nest
import pylab
import plotting_functions as plotting
import spiketools
import organiser
import default
import time
import sim_nest
from copy import deepcopy
from bisect import bisect_right
import global_params
from matplotlib.ticker import MaxNLocator
from general_func import *
import pandas as pd
import pickle
#from scipy.ndimage import gaussian_filter

datapath = '../data/'
datafile = 'fig02_cluster_dynamics'
datafile1 = 'ff_cv2_spontaneous'

def simulate_spontaneous(params):
    pylab.seed()
    trials = params['trials']
    trial_length = params['trial_length']
    sim_params = deepcopy(params)
    sim_params['simtime'] = trials*trial_length
    ff_window = params['ff_window']
    long_spiketimes = sim_nest.simulate(sim_params)['spiketimes']
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


def plot_ff_cv_vs_jep(params,jep_range=pylab.linspace(1,4,41),jipfactor = 0.,reps = 10,
                      spike_js = [1.,2.,3.,4.],spike_simtime = 1000.,markersize = 0.5,
                      spikealpha = 0.5,plot_units = [0,4000],spike_randseed = 0,
                      plot = True,redo_spiketrains = False):
    #print(params)
    ffs = []
    cv2s = []
    counts = []
    for jep in jep_range:
        print('###############################################################################')
        print(jep,'-----------------------------------------------------------------------')
        print('###############################################################################')
        if jipfactor != 0. or jep<10:
            jip = 1. +(jep-1)*jipfactor
            params['jplus'] = pylab.around(pylab.array([[jep,jip],[jip,jip]]),5)
            #results = load_data(datapath, datafile,params,old_key_code=True,reps=reps)
            results = organiser.check_and_execute(params, simulate_spontaneous, datafile1,reps = reps)
            ff = [r[0] for r in results]
            cv2 = [r[1] for r in results]
            count = [r[2] for r in results]
            ffs.append(ff)
            cv2s.append(cv2)
            counts.append(count)
        else:
            ffs.append(ff)
            cv2s.append(cv2)
            counts.append(count)
            
    ffs = pylab.array(ffs)
    cv2s = pylab.array(cv2s)
    cv2s = pylab.nanmean(cv2s,axis=1)
    
    if plot:
        ffs = pylab.nanmean(ffs,axis=1)
        pylab.plot(jep_range,ffs,'k')
        if jipfactor == 0.:
            pylab.gca().set_ylim(-0.2, 4)
        else:
            pylab.gca().set_ylim(-0.2, 12)
            


        n_boxes = len(spike_js)
        box_bottom = ffs.max()*1.1
        box_top = ffs.max()*1.7
        xlim = [jep_range.min(),jep_range.max()]
        box_sep = 0.05*(xlim[1]-xlim[0])
        box_lim = [xlim[0]+box_sep,xlim[1]-box_sep]
        box_span = box_lim[1]-box_lim[0]
        box_width = (box_span-box_sep*(n_boxes-1))/float(n_boxes)
        box_height = box_top-box_bottom
        for i,j in enumerate(spike_js):
            if j == 'max':
                j = jep_range[pylab.argmax(ffs)]
            box_target_ind = pylab.argmin(pylab.absolute(j-jep_range))
            box_target = [jep_range[box_target_ind],ffs[box_target_ind]]
            box_left = box_lim[0]+i*(box_width+box_sep)
            plotting.draw_box([box_left,box_bottom,box_width,box_height], box_target,[0.6,0.6,0.6,0.6])

            jep = round(j,4)
            
            jip = round(1 + (jep-1)*jipfactor,4)
            print('spikes for ',jep,jip)
            spike_params = deepcopy(params)
            spike_params['jplus'] = pylab.array([[jep,jip],[jip,jip]])
            spike_params['randseed'] = spike_randseed
            spike_params['simtime'] = spike_simtime
            print(spike_params)
            #spiketimes = load_data(datapath, datafile + '_spikes',spike_params,old_key_code=True, reps=None)['spiketimes']
            spiketimes = organiser.check_and_execute(spike_params, sim_nest.simulate, datafile1 +'_spikes'
                                                     ,redo = redo_spiketrains)['spiketimes']
            spiketimes = spiketimes[:,spiketimes[1]<plot_units[1]]
            spiketimes = spiketimes[:,spiketimes[1]>=plot_units[0]]
            spiketimes[1] -= plot_units[0]
            spiketimes[0]/= spiketimes[0].max()
            spiketimes[1]/= spiketimes[1].max()

            spiketimes[0] *= box_width
            spiketimes[0] += box_left
            spiketimes[1] *= box_height
            spiketimes[1] += box_bottom
            
            pylab.plot(spiketimes[0],spiketimes[1],'.k',markersize = markersize,alpha = spikealpha)

            pylab.xlim(jep_range.min(),jep_range.max())
        pylab.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

def plot_ff_jep_vs_Q(params,jep_range=pylab.linspace(1,4,41),
                     Q_range = pylab.arange(2,20,2),jipfactor = 1,reps = 40,
                     plot = True,vrange = [0,15],redo = False):
    
    if jipfactor == 0.:
        model = 'E_clustered'
    else:
        model = 'EI_clustered'
    try:
        ffs = pd.read_pickle(datapath + "ffs_fig2_"+model)
    except:
        ffs = pylab.zeros((len(jep_range),len(Q_range),reps))
        
        #all_results = pd.read_pickle(datapath + datafile1)
        for i,jep_ in enumerate(jep_range):
            for j,Q in enumerate(Q_range):
                jep = float(min(jep_,Q))
                if jipfactor == 0.:
                    params['portion_I'] = Q
                else:
                    params['portion_I'] = 1
                jip = 1. +(jep-1)*jipfactor
                print('#################################################################################')
                print(Q,jep,jip,'-------------------------------------------------------------------------')
                print('#################################################################################')
                params['jplus'] = pylab.around(pylab.array([[jep,jip],[jip,jip]]),5)
                params['Q'] = int(Q)
                results = organiser.check_and_execute(params, simulate_spontaneous, datafile1,
                                reps = reps,ignore_keys=['n_jobs'],redo = redo)
                #key = key_from_params(params, reps=reps,ignore_keys=['n_jobs'])
                #results = [all_results[result_key] for result_key in key]
                ff = [r[0] for r in results]
                ffs[i,j,:] = ff
                if jep_>Q:
                    ffs[i,j,:] = pylab.nan
                
        pickle.dump(ffs,open(datapath + "ffs_fig2_"+model,'wb'))

    if plot:
        print(pylab.nanmean(ffs,axis=2).T)
        pylab.contourf(jep_range,Q_range,pylab.nanmean(ffs,axis=2).T,
                       levels = [0.5, 1.,1.5,2.],extend = 'both', 
                       cmap = 'Greys')#, algorithm = 'threaded',
                       #antialiased=True, nchunk=0, norm="linear")
        x = pylab.linspace(Q_range.min(), jep_range.max(),1000)
        y1 = pylab.ones_like(x)*Q_range.min()
        y2 = x
        pylab.fill_between(x,y1, y2,facecolor = 'w',hatch = '\\\\\\',edgecolor = global_params.colors['orange'])
        pylab.xlabel(r'$J_{E+}$')
        pylab.ylabel(r'$Q$')
        pylab.axis('tight')
    return ffs
    




        
if __name__ == '__main__':
    n_jobs = 12
    settings = [{'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,'n_jobs':n_jobs,'Q':50,'jipfactor':0.,
                 'jep_range':pylab.arange(1,50.001,0.1),'spike_js':[1.,3.,5., 8. ,10.], 'portion_I':50}, 
                {'jipfactor':0.,'fixed_indegree':False, 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                 'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26},
                {'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,'n_jobs':n_jobs,'Q':50,'jipfactor':0.75,
                 'jep_range':pylab.arange(1.001,50.001, 0.1),'spike_js':[1.,8.,10.5,14.,50.], 'portion_I':1},
                {'jipfactor':0.75,'fixed_indegree':False, 'warmup':200,'ff_window':400,'trials':20,'trial_length':400.,
                 'n_jobs':n_jobs,'I_th_E':2.14,'I_th_I':1.26}]  #3,5  hz

    
    
    plot = True
    reps = 20
    x_label_val = -0.25
    num_row, num_col = 2,3
    if plot:
        fig  =plotting.nice_figure(ratio = 0.8,latex_page=global_params.text_width_pts)
        fig.subplots_adjust(bottom = 0.15,hspace = 0.4,wspace = 0.3)
        
        labels = ['a','b','c','d']
    title_left = ['E clustered network','','E/I clustered network']
    for i,params in enumerate(settings):
        row = int(i/2)
        col= int(i%2)
        if plot and i in [0,2]:
            ax = plotting.simpleaxis(pylab.subplot2grid((num_row,num_col),(row, col), colspan=2))
            plotting.ax_label1(ax, labels[i],x=x_label_val)
            pylab.ylabel('FF')
            pylab.xlabel('$J_{E+}$')
            jep_range = params.pop('jep_range')
            spike_js = params.pop('spike_js')
            jipfactor = params.pop('jipfactor')
            plot_ff_cv_vs_jep(params,reps = reps,jipfactor =jipfactor,
                              jep_range = jep_range,spike_js = spike_js,
                              plot = plot,spike_randseed = 3,spike_simtime = 2000.,markersize = 0.1,spikealpha= 0.3)
            pylab.gca().text(-7, i/3.+0.2, title_left[i], rotation=90)#, fontweight='bold')
        else:
            jipfactor = params.pop('jipfactor')
            jep_step = 0.5
            jep_range = pylab.arange(1.,15.+0.5*jep_step,jep_step)
            q_step = 1
            Q_range = pylab.arange(q_step,60+0.5*q_step,q_step)
            
            if plot:
                ax = plotting.simpleaxis(pylab.subplot2grid((num_row,num_col),(row, col+1)))          
                plotting.ax_label1(ax, labels[i], x=x_label_val)
            ffs = plot_ff_jep_vs_Q(params,jep_range,Q_range,jipfactor=jipfactor,plot=plot)
            if plot:
                cbar = pylab.colorbar()
                cbar.set_label('FF', rotation=90)
    pylab.savefig('fig2.pdf')
    pylab.savefig('fig2.png', dpi=300)    
    pylab.show()



