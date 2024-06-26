import sys;sys.path.append('utils')
import plotting_functions as plotting
from copy import deepcopy
import pylab
import spiketools
from sim_nest import simulate
import analyse_nest
import default
from general_func import *
import organiser
datapath = 'cache/'
datafile = 'fig03_upstate_balance_reduced'

def simulate_and_analyse(original_params):
    
    params = deepcopy(original_params)
    # add kernel length to simtime to have rates over the whole interval
    kernel_width = spiketools.triangular_kernel(sigma = params['rate_kernel']).shape[0]
    params['warmup']-= kernel_width/2
    params['simtime'] += kernel_width
    jep = params['jep']
    jipfactor = params['jipfactor']
    jip = 1. +(jep-1)*jipfactor
    params['jplus'] = pylab.around(pylab.array([[jep,jip],[jip,jip]]),5)
    params['record_voltage'] = True
    params['record_from'] = params['focus_unit']

    sim_results = simulate(params)
    print(list(sim_results.keys()))
    spiketimes = sim_results['spiketimes']
    results = {}
    
    # compute average cluster rates
    cluster_rates = []
    cluster_size = params['N_E']/params['Q']
    
    
    cluster_rates,t_rate = analyse_nest.compute_cluster_rates(spiketimes, params['N_E'], params['Q'],kernel_sigma=params['rate_kernel'],tlim = [0,params['simtime']+kernel_width])
    results['cluster_rates'] = cluster_rates
    results['t_rate'] = t_rate-kernel_width/2
    
    # remove added kernel width from spiketimes again
    spiketimes[0,:] -= kernel_width/2
    spiketimes = spiketimes[:,spiketimes[0]>0]
    spiketimes = spiketimes[:,spiketimes[0]<=original_params['simtime']]
    
    results['spiketimes'] = spiketimes
    # extract currents for focus unit

    focus_unit = cluster_size * params['focus_cluster']+params['focus_unit']
    focus_index = find(sim_results['senders'] == focus_unit)
    
    results['current_times'] = sim_results['times']-kernel_width/2
    results['ex_current'] = sim_results['I_syn_ex'][focus_index]
    results['inh_current'] = sim_results['I_syn_in'][focus_index]
    results['Ixe'] = sim_results['I_xE']
    results['V_m'] = sim_results['V_m'][focus_index]
    results['e_rate'] = sim_results['e_rate']
    results['i_rate'] = sim_results['i_rate']
    results['focus_cluster_inds'] = list(range(int(params['focus_cluster']*cluster_size),int((params['focus_cluster']+1)*cluster_size)))
    results['focus_spikes'] = spiketimes[:,spiketimes[1] == focus_unit-1]

    return results

def do_plot(params,axes=None,redo = False,plot = True,markersize = 0.5,spikealpha = 1,box_color = 'k',cmap = 'jet',lw = 0.8,show_clusters = 4,
            legend = False,current_limits = [-15,18],voltage_limits = [-20,42],ylabel = False,rate_ticks = [],V_ticks = [],I_ticks = []):
    #result = load_data(datapath, datafile,params,old_key_code=True)
    result = organiser.check_and_execute(params, simulate_and_analyse, datafile,reps = None)
    spiketimes = result['spiketimes']
    if plot:
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[0]
        pylab.sca(ax)

        pylab.xlabel('time [ms]')
        # draw a box around the focus cluster/interval
        bottom = min(result['focus_cluster_inds'])
        top = max(result['focus_cluster_inds'])
        left = params['focus_interval'][0]
        right = params['focus_interval'][1]
        
        x_margin = 2.
        y_margin = 2.
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],color = "r",lw = 0,alpha=0.1)
        pylab.plot(spiketimes[0],spiketimes[1],'.k',markersize = markersize,alpha = spikealpha)
        pylab.xlim(0,params['simtime'])
        pylab.plot([left-x_margin,right+x_margin,right+x_margin,left-x_margin,left-x_margin],[bottom-y_margin,bottom-y_margin,top+y_margin,top+y_margin,bottom-y_margin],'--',color = box_color,lw = 1.5)
        
        # show only the clusters below and above the focus cluster

        cluster_size = params['N_E']/params['Q']
        for i in range(params['Q']):
            pylab.axhline(i*cluster_size,linestyle = '-',color = 'k',alpha = 0.8)
        pylab.ylim((params['focus_cluster']-(show_clusters))*cluster_size,(params['focus_cluster']+show_clusters+1)*cluster_size)
        pylab.yticks([])
        if ylabel:
            pylab.ylabel('unit')
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[1]
        pylab.sca(ax)
        pylab.xlabel('time [ms]')
        cluster_rates = result['cluster_rates']
        t_rate = result['t_rate']
        if ylabel:
            pylab.ylabel(r'rate [1/s]')

        Q = params['Q']
        colors = plotting.make_color_list(2*show_clusters+1,cmap = cmap)
        for i,q in enumerate(range(params['focus_cluster']-show_clusters,params['focus_cluster']+show_clusters+1)):
            
            
            print(q)    
            pylab.plot(t_rate,cluster_rates[q],lw= lw,color = colors[i])
            if q == params['focus_cluster']:
                
                # find bounding box of that line in the focus interval
                focus_piece = find((t_rate>=left)*(t_rate<=right))
                bottom = cluster_rates[q][focus_piece].min()
                top = cluster_rates[q][focus_piece].max()
                pylab.plot(t_rate[focus_piece],cluster_rates[q][focus_piece],'--r',lw=1.5)
        
        pylab.xlim(0,params['simtime'])
        pylab.yticks(rate_ticks)
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[2]


        pylab.sca(ax)
        bottom = voltage_limits[0]
        top = voltage_limits[1]
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],color = "r",lw = 0, alpha=0.1)        
        pylab.xticks(list(range(left,right+50,100)))
        pylab.xlabel('time [ms]')
        ex_current = result['ex_current'][0]+result['Ixe']
        inh_current = result['inh_current'][0]
        time = result['current_times']
        focus_piece = find((time>=left)*(time<=right))        
        print(ex_current.shape,time.shape)

        pylab.plot(time[focus_piece],ex_current[focus_piece],color = '0.4',label = r'$I_{E} + I_{x}$')
        pylab.plot(time[focus_piece],inh_current[focus_piece],color = '0.65',label = r'$I_{I}$')
        pylab.plot(time[focus_piece],ex_current[focus_piece]+inh_current[focus_piece],color = 'k',label = r'$I_{tot}$')
        pylab.axhline(0,linestyle ='--',color = '0.7')
        if params['jipfactor'] == 0.:
            current_limits = [current_limits[0]+4, current_limits[1]-4]
            I_ticks = [I_ticks[0]+4, I_ticks[1], I_ticks[2]-4]
        pylab.ylim(current_limits)
        pylab.xlim(left,right)
        pylab.yticks(I_ticks)
        if ylabel:
            pylab.ylabel(r'$I_{syn}$ [pA]')


        if legend:
            pylab.legend(loc = 'upper center',frameon = False,fontsize= 6,ncol = 3,handlelength = 1.5,columnspacing = 1.,handletextpad = 0.5,borderaxespad = 0.,borderpad = 0.)
        
        if axes is None:
            pylab.figure()
            ax = pylab.subplot(1,1,1)
        else:
            ax = axes[3]
    
        pylab.sca(ax)
        bottom = voltage_limits[0]
        top = voltage_limits[1]
        pylab.fill([left,right,right,left,left],[bottom,bottom,top,top,bottom],color = "r",lw = 0, alpha=0.1)        

        pylab.xlabel('time [ms]')
        if ylabel:
            pylab.ylabel('$V_{m}$ [mV]')
        V_m = result['V_m'][0]
        pylab.plot(time[focus_piece],V_m[focus_piece],'k')
        V_th_E = params.get('V_th_E',default.V_th_E)
        pylab.axhline(V_th_E,linestyle = '--',color = 'k')
        spikeheight = 20.
        for spike in result['focus_spikes']:
            pylab.plot([spike]*2,[V_th_E,V_th_E+spikeheight],'k')
        pylab.ylim(voltage_limits)
        pylab.xlim(left,right)
        pylab.xticks(list(range(left,right+50,100)))
        pylab.yticks(V_ticks)
        

    
if __name__ == '__main__':


    params = {'simtime':1000.,'n_jobs':12,'Q':50,'rate_kernel':50,'N_E':4000}
    settings = [{'warmup':0, 'jipfactor':0.,'jep':3.7,'randseed':3,'focus_cluster':8,'focus_interval':[200,600],'focus_unit':6}, 
                {'warmup':0, 'jipfactor':0.75,'jep':8.,'randseed':5,'focus_cluster':15,'focus_interval':[700,1000],'focus_unit':12}]  

    plot= True


    
    if plot:
        fig = plotting.nice_figure(ratio = 1.1)
        ncols =2
        nrows = 7
        hspace = 0.6
        gs = pylab.GridSpec(nrows,ncols,top=0.95,bottom=0.08,hspace = 0.01,left = 0.1,right = 0.9,height_ratios = [1.8,hspace,0.7,hspace,1.1,hspace,.7])
        x_label_val = -0.17
        subplotspec = gs.new_subplotspec((0,0), colspan=1,rowspan=1)
        ax1 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax1, 'a', x=x_label_val)
        plotting.ax_label_title(ax1, 'E clustered network')
        subplotspec = gs.new_subplotspec((0,1), colspan=1,rowspan=1)
        ax2 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax2, 'b', x=x_label_val)
        plotting.ax_label_title(ax2, 'E/I clustered network')

        subplotspec = gs.new_subplotspec((2,0), colspan=1,rowspan=1)
        ax3 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax3, 'c', x=x_label_val)

        subplotspec = gs.new_subplotspec((2,1), colspan=1,rowspan=1)
        ax4 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax4, 'd',x=x_label_val)

        subplotspec = gs.new_subplotspec((4,0), colspan=1,rowspan=1)
        ax5 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax5, 'e', x=x_label_val)

        subplotspec = gs.new_subplotspec((4,1), colspan=1,rowspan=1)
        ax6 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax6, 'f',x=x_label_val)

        subplotspec = gs.new_subplotspec((6,0), colspan=1,rowspan=1)
        ax7 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax7, 'g',x=x_label_val)

        subplotspec = gs.new_subplotspec((6,1), colspan=1,rowspan=1)
        ax8 =plotting.simpleaxis(pylab.subplot(subplotspec))
        plotting.ax_label1(ax8, 'h',x=x_label_val)
        
        axes = [[ax1,ax3,ax5,ax7],[ax2,ax4,ax6,ax8]]
    for setno,setting in enumerate(settings):
        for k in list(setting.keys()):
            params[k] = setting[k]
        if setno == 0:
            legend = True
            ylabel = True
            rate_ticks = [0,30,60,90]
        else:
            legend = False
            ylabel = False
            rate_ticks = [0,10,20,30]
        print(params)
        do_plot(params,redo = False,plot = plot,axes = axes[setno],box_color = 'r',cmap = 'Greys',legend = legend,ylabel = ylabel,
                rate_ticks = rate_ticks,V_ticks = [-20,0,20,40],I_ticks = [-15,0,15])
    pylab.savefig('fig3.pdf')

