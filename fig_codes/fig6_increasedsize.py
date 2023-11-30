import sys;sys.path.append('../utils')
import analyse_model
import pylab
import organiser
from organiser import memoized
from copy import deepcopy
from simulate_experiment import get_simulated_data
import plotting_functions as plotting
import spiketools
import analyses
import joe_and_lili
from general_func import *

from reaction_times_func import *






    

if __name__ == '__main__':
    #sim_params = {'randseed':8721,'trials':150,'N_E':1200,'N_I':300,'I_th_E':1.25,'I_th_I':0.78,'Q':6,'rs_stim_amp':0,'n_jobs':12,'conditions':[1,2,3]}
    sim_params = {'randseed':8721,'trials':800,'N_E':2400,'N_I':600,'I_th_E':1.25,'I_th_I':0.78,'Q':6,'rs_stim_amp':0,'n_jobs':20,'conditions':[1,2,3]}

    # settings = [{'randseed':7745,'jep':3.3,'jipratio':0.75,'condition_stim_amps':[0.15,0.15,0.15],'rs_stim_amp':0.15,'rs_length':400},
    #             {'randseed':5362,'jep':2.8,'jipratio':0.75,'condition_stim_amps':[0.15,0.15,0.15],'rs_stim_amp':0.15,'rs_length':400}]

    # settings = [{'randseed':7745,'jep':3.3,
    #              'jipratio':0.75,'condition_stim_amps':[0.15,0.15,0.15],
    #              'rs_stim_amp':0.15,'rs_length':400}]

    # settings = [{'randseed':7745,'jep':3.3,'jipratio':0.75,
    #              'condition_stim_amps':[0.15,0.15,0.15],
    #              'rs_stim_amp':0.15,'rs_length':400,'trials':2000}]
    settings = [{'randseed':7745,'jep':3.2,
                 'jipratio':0.75,'condition_stim_amps':[0.1,0.1,0.1],
                 'rs_stim_amp':0.1,'rs_length':400}]


    x_label_val=-0.5                            
    fig = plotting.nice_figure(ratio = 1.)
    nrows = 2
    ncols = 2
    gs = pylab.GridSpec(nrows,ncols,top=0.9,bottom=0.1,hspace = 0.4,wspace = 0.9,left = 0.2,right = 0.88,height_ratios = [2,1])
    subplotspec = gs.new_subplotspec((1,1), colspan=1,rowspan=1)
    ax2 = plotting.ax_label1(plotting.simpleaxis(pylab.subplot(subplotspec)),'c',x=x_label_val)
    ax2.set_title('Behaving monkey')
    labelsize=5
    cond_colors = ['navy','royalblue','lightskyblue']
    reaction_time_plot('joe', condition_colors = cond_colors)
    pylab.xlabel('reaction time [ms]')
    pylab.ylabel('p.d.f')
    pylab.axvline(1500,linestyle = '-',color = 'k',lw = 0.5)
    pylab.ylim(0,0.015)    
    pylab.yticks([0,0.004,0.008,0.012])
    pylab.legend(frameon = False,fontsize = 6,loc = 'upper right', bbox_to_anchor=(1.45, 1.1))
    pylab.xticks([1500,1600,1700,1800,1900,2000])
    ax2.set_xticklabels(['RS', '100','200','300','400','500'])
    condition_alpha = 1.
                
    condition_colors = [[0,0,0,condition_alpha],[0.4,0.4,0.4,condition_alpha],[0.6,0.6,0.6,condition_alpha]]
    tlim = [-500,2000]

    for setno,setting in enumerate(settings[:1]):
        for k in list(setting.keys()):
            sim_params[k] = setting[k]

        for tau in [50.]:
            for threshold_per_condition in [False]:
                for integrate_from_go in [False]:
                    for min_count_rate in [7.5]:
                        for align_ts in [False]:

                            params = {'sim_params':sim_params}
                            result = get_reaction_time_analysis(params,tlim  =tlim,redo = False,
                                                    tau  =tau,integrate_from_go = integrate_from_go,
                                                    normalise_across_clusters=True,
                                                    threshold_per_condition = threshold_per_condition)
                            print(result['integrals'].shape)
                            print(list(result.keys()))
                            
                            rts = result['rts']
                            conditions = result['conditions']
                            directions = result['directions']
                            predictions = result['predictions']
                            print(predictions)
                            print(directions)
                            correct= directions == predictions
                            correct_inds = find(correct)
                            incorrect_inds = find(correct==False)
                            subplotspec = gs.new_subplotspec((0,0), colspan=2,rowspan=1)
                            ax1 = plotting.ax_label1(plotting.simpleaxis(
                                pylab.subplot(subplotspec)),'a',x=x_label_val/3)
                            pylab.suptitle('Example trial of motor cortical attractor model')
                            
                            print('find correct cond==3', find(correct*(conditions==3)))
                            plot_trial = find(correct*(conditions==3))[8]#[6]
                            print('plot trial',plot_trial)
                            pylab.xticks([-500,0,1000])
                            pylab.gca().set_xticklabels(['0', '500','1500'])

                            print('direction prediction condition',
                                  directions[plot_trial],
                                  predictions[plot_trial])

                            data  = get_simulated_data(params['sim_params'],
                                            datafile = 'simulated_data_fig6')
                            print('got the data!!!')
                            cut_window = [-500,2000]
                            trial_starts = data['trial_starts']
                            spiketimes = spiketools.cut_spiketimes(data['spiketimes'],tlim = pylab.array(cut_window)+trial_starts[plot_trial])
                            spiketimes[0] -= trial_starts[plot_trial]
                            pylab.plot(spiketimes[0],spiketimes[1],'.',ms =0.5,color = '0.5')
                            pylab.xlim(cut_window)
                            Q = params['sim_params']['Q']
                            N_E = params['sim_params']['N_E']
                            cluster_size = N_E/Q
                            direction_clusters =  data['direction_clusters']
                            integrals = result['integrals']
                            time = result['time']
                            threshold = result['condition_thresholds'][conditions[plot_trial]]
                            pylab.axvline(1000,linestyle = '--',color ='k',lw = 0.5)
                            direction_clusters = pylab.array(direction_clusters).flatten()
                            print(direction_clusters)
                            for cluster in range(6):

                                pylab.text(2000,(cluster+0.5)*cluster_size,r'\textbf{'+str(cluster+1)+'}',va = 'center',ha = 'left')
                                direction = find(direction_clusters == cluster)[0]
                                
                                
                                pylab.plot(time,cluster*cluster_size +integrals[direction,plot_trial]*cluster_size*0.8,color = 'k')
                                print(integrals[direction,plot_trial])
                                pylab.plot([1000,2000],[cluster*cluster_size +threshold*cluster_size*0.8]*2,'--k',lw =0.5)
                                
                                if (direction+1) == directions[plot_trial]:
                                    try:
                                        crossing = find((integrals[direction,plot_trial]>threshold)*(time>1000))[0]
                                        print(crossing)
                                        pylab.plot(time[crossing],cluster*cluster_size +threshold*cluster_size*0.8,'ok',ms = 4)
                                    except:
                                        print('no crossing')
                                    
                            pylab.xlim(time.min(),time.max())
                            pylab.ylim(0,N_E)
                            pylab.ylabel('unit')
                            pylab.xlabel('time [ms]')
                            pylab.text(-50,1250,'PS')
                            pylab.text(950,1250,'RS')
                            subplotspec = gs.new_subplotspec((1,0), colspan=1,rowspan=1)
                            ax2 = plotting.ax_label1(plotting.simpleaxis(pylab.subplot(subplotspec)),'b',x=x_label_val)
                            ax2.set_title('Attractor model')
                            
              
                            
                            
                            for condition in [1,2,3]:
                                
                                rt = rts[(conditions == condition)*correct]
                                bins = pylab.linspace(0,500,15)
                                print('cond len_rt',condition,len(rt))
                                print('mediann', pylab.median(rt))
                                pylab.hist(rt,bins,histtype = 'step',facecolor = cond_colors[condition-1],
                                           density = True,edgecolor  = cond_colors[condition-1],label = 'condtion '+str(condition))
                                pylab.xlim(1400,2000)
                            import scipy.stats as ss
                            min_len = min(len(rts[(conditions == 2)*correct]),len(rts[(conditions == 3)*correct]))
                            print('sample size model:', min_len)
                            print('wilcoxon test', ss.wilcoxon(rts[(conditions == 2)*correct][:min_len],rts[(conditions == 3)*correct][:min_len]))
                            print('condition 1', rts[(conditions == 1)*correct])

                                
                            pylab.xlim(-100,500)
                            pylab.xticks([0,100,200,300,400,500])
                            pylab.gca().set_xticklabels(['RS', '100','200','300','400','500'])
                            pylab.ylim(0,0.015)
                            pylab.yticks([0,0.004,0.008,0.012])                            
                            pylab.ylabel('p.d.f')
                            pylab.xlabel('reaction time [ms]')
                            pylab.axvline(0,linestyle = '-',color = 'k',lw = 0.5)             
                            
                            
                



    pylab.savefig('fig6_extendedSize.pdf')
    pylab.show()



