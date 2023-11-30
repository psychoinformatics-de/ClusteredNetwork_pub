import sys;sys.path.append('../data')
sys.path.append('..')
import ClusteredNetwork_pub_rev1.utils.analyses_to_remove as analyses_to_remove
import joe_and_lili
import plotting_functions as plotting
import pylab
import numpy as np
import pickle as pickle
from scipy.stats import wilcoxon
import organiser
#from organiser import memoized
from matplotlib.markers import TICKDOWN
from global_params import colors as global_colors


monkey = b'lili'#b'joe' # None
if monkey != None:
    global_gns = pylab.unique(joe_and_lili.get_toc([
        ('monkey','=',monkey)])['global_neuron'])
else:
    global_gns = pylab.unique(joe_and_lili.get_toc()['global_neuron'])

off_gray = global_colors['off_gray']
yellow = global_colors['yellow']
green = global_colors['green']
red = global_colors['red']
def draw_hex_array(center,size=0.3,colors = [[0.5]*3]*7,axes = None,
    radius = 0.1,add = True,show_numbers = False,draw_center = True,lw = 1., epoch=None):
    angles = pylab.array([30,90,150,210,270,330])*pylab.pi/180
    Y = size*pylab.cos(angles)
    X = size*pylab.sin(angles)
    
    
    i = 0
    circs= []
    coords = []
    number = 6
    for x,y in zip(X,Y):
        coords.append((x+center[0],y+center[1]))
        circ = pylab.Circle((x+center[0],y+center[1]), radius=radius,  fc=colors[i],clip_on = False,lw= lw)
        if axes is None:
            axes = pylab.gca()
        if add:
            axes.add_patch(circ)
        circs.append(circ)
        #pylab.text(x,y,str(i),va='center',ha = 'center')
        if show_numbers:
            pylab.text(x+center[0],y+center[1],str(number),size = 6,ha ='center',va = 'center')
            if number == 6:
                number =1
            else:
                number+=1
        i+=1

    if draw_center:
        circ = pylab.Circle((center[0],center[1]), radius=radius,  fc=colors[-1],clip_on = False,lw = lw)
        if axes is None:
            axes = pylab.gca()
        if add:
            axes.add_patch(circ)
    if epoch!=None:
        pylab.text(center[0]-80,center[1]+250, epoch,size = 5,ma='center')

    return circs,coords

def get_stats(gns,min_trials  =10,min_count_rate = 15,min_count=200,
              minvals = 0,alignment = 'TS',tlim = [0,2000],window =400,
              calc_ff=False, calc_cv2=False, calc_lv=False, calc_cv_two=False):
    if gns is None:
        gns = global_gns
    rates = []
    trial_counts = []
    count_rates = []
    ffs = []
    cv2s= []
    
    lvs = []
    cv_twos = []

    for gn in gns:
        for direction in range(1,7):
            #print gn,direction
            rate,trate = analyses_to_remove.get_rate(
                gn,condition = 1,direction = direction,
                tlim  =tlim,alignment = alignment)
            rates.append(rate[0])
            trial_counts.append(analyses_to_remove.get_trial_count(gn,1,direction))
            count_rates.append(analyses_to_remove.get_mean_direction_counts(gn,1,direction,tlim  =tlim,alignment = alignment))
            ff,tff = analyses_to_remove.get_ff(gn,condition = 1,direction = direction,window = window,tlim  =tlim,alignment = alignment)
            ffs.append(ff)
            cv2,tcv2 = analyses_to_remove.get_corrected_cv2(gn,condition = 1,direction = direction,minvals = minvals,tlim  =tlim,alignment = alignment)
            cv2s.append(cv2)

            lv,tlv = analyses_to_remove.get_lv(gn,condition = 1,direction = direction,tlim  =tlim,alignment = alignment)
            lvs.append(lv)
            cv_two,tcv_two = analyses_to_remove.get_cv_two(gn,condition = 1,direction = direction,tlim  =tlim,alignment = alignment)
            cv_twos.append(cv_two)
            

            

    
    trial_counts = pylab.array(trial_counts)
    count_rates = pylab.array(count_rates)

    mask = (trial_counts>=min_trials)*(count_rates>=min_count_rate)
    rates = pylab.array(rates)        
    ffs = pylab.array(ffs)
    lvs = pylab.array(lvs)
    cv_twos = pylab.array(cv_twos)
    return tff,ffs[mask],tlv,lvs[mask],tcv_two,cv_twos[mask], trate, rates[mask]
    
    
def plot_rates(gns = None,plotargs = {},min_trials  =10,min_count_rate = 15,min_count=200,minvals = 0,alignment = 'TS',tlim = [0,2000],window =400):
    tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, count_rates = get_stats(gns,min_trials,min_count_rate,min_count,minvals,alignment,tlim,window)
    print(len(trate))
    print(pylab.shape(count_rates))
    pylab.plot(trate,pylab.nanmean(count_rates,axis = 0),**plotargs)


def plot_ffs(gns = None,plotargs = {},min_trials  =10,min_count_rate = 15,min_count=200,minvals = 0,alignment = 'TS',tlim = [0,2000],window =400):
    tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, count_rates = get_stats(gns,min_trials,min_count_rate,min_count,minvals,alignment,tlim,window)

    pylab.plot(tff,pylab.nanmean(ffs,axis = 0),**plotargs)


def plot_interval_stats(gns=None,plotargs_lv={},plotargs_cv_two = {},min_trials  =10,min_count_rate = 15,min_count=200,minvals = 0,alignment = 'TS',tlim = [0,2000],window =400):
    tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, count_rates = get_stats(gns,min_trials,min_count_rate,min_count,minvals,alignment,tlim,window)
    
    if plotargs_lv is not None:
        pylab.plot(tlv,pylab.nanmean(lvs,axis = 0),**plotargs_lv)
    if plotargs_cv_two is not None:
        pylab.plot(tcv_two,pylab.nanmean(cv_twos,axis = 0),**plotargs_cv_two)

def plot_experiment(size,radius,direction =1,lw = 1.,y_pos = 0,condition = 1,write_epoch=False):

    if write_epoch:
        epoch = 'TS'#'Trial \n Start (TS)'
    else:
        epoch=None
    colors = [off_gray]*6 + [yellow]
    draw_hex_array([0,y_pos],colors = colors,size = size,radius = radius,lw = lw,epoch=epoch)
    colors = [off_gray]*7
    if condition ==1:
        colors[direction] = green
    elif condition == 2:
        if direction in [1,2,3]:
            colors[1] = green
            colors[0] = green
        else:
            colors[4] = green
            colors[5] = green
    elif condition == 3:
        if direction in [1,2,3]:
            colors[1] = green
            colors[2] = green
            colors[0] = green
        else:
            colors[4] = green
            colors[5] = green
            colors[3] = green
    if write_epoch:
        epoch = 'PS'#'Preparatory \n Signal (PS)'
    else:
        epoch=None
    draw_hex_array([500,y_pos],colors = colors,size = size,radius = radius,lw = lw,epoch=epoch)

    colors = [off_gray]*7
    colors[direction] = red

    if write_epoch:
        epoch = 'RS'#'Response \n signal (RS)'
    else:
        epoch=None
    draw_hex_array([1500,y_pos],colors = colors,size = size,radius = radius,lw = lw, epoch=epoch)


def make_condition1_scheme_for_slides():
    fig = plotting.nice_figure(ratio = 0.3)
    pylab.subplot(1,1,1)
    pylab.subplots_adjust(bottom = 0,left = 0,right = 1,top = 1)
    plot_experiment(80, 25,lw =0.5)
    time_y = -150
    pylab.plot([0,500,1500],[time_y]*3,'-k',marker = TICKDOWN,lw = 1,mew = 1,ms = 4)
    pylab.axis('off')
    pylab.box('off')
    pylab.axis('equal')
    pylab.savefig('experiment_condition_1.png',dpi=300)

def make_3condition_scheme_for_slides():
    fig = plotting.nice_figure(ratio = 0.5)
    pylab.subplot(1,1,1)
    pylab.subplots_adjust(bottom = 0,left = 0,right = 1,top = 1)
    plot_experiment(80, 25,lw =0.5)

    plot_experiment(80, 25,lw =0.5,y_pos = -300,condition=2)

    plot_experiment(80, 25,lw =0.5,y_pos = -600,condition=3)
    

    time_y = -750
    pylab.plot([0,500,1500],[time_y]*3,'-k',marker = TICKDOWN,lw = 1,mew = 1,ms = 4)
    pylab.axis('off')
    pylab.box('off')
    pylab.axis('equal')
    pylab.savefig('experiment_3conditions.png',dpi=300)


def make_condition1_data_fig_for_slides(lw = 2,FF_col  ='k',cv_two_col = '0.5'):

    tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, count_rates = get_stats(None,min_trials  =10,min_count_rate = 15,min_count=200,minvals = 0,alignment = 'TS',tlim = [0,2000],window =400)

    fig = plotting.nice_figure()
    gs = pylab.GridSpec(2,1,top=0.98,bottom=0.15,hspace = 0.1,wspace = 0.3,left = 0.1,right = 0.9,height_ratios = [0.5,2.])



    
    subplotspec = gs.new_subplotspec((0,0), colspan=1,rowspan=1)
    ax1 = pylab.subplot(subplotspec)
    plot_experiment(80, 25,lw =0.5)
    pylab.axis('off')
    pylab.box('off')
    pylab.axis('equal')
    pylab.xlim(0,2000)
    
    subplotspec = gs.new_subplotspec((1,0), colspan=1,rowspan=1)
    ax2 = plotting.simpleaxis(pylab.subplot(subplotspec))
    pylab.axvline(500,linestyle = '--',color = 'k',lw = 0.5)
    pylab.axvline(1500,linestyle = '--',color = 'k',lw = 0.5)

    pylab.plot(tff,pylab.nanmean(ffs,axis = 0),color = FF_col,lw = lw,label = '$FF$')
    pylab.xlim(0,2000)
    pylab.ylim(0,2.)
    pylab.xlabel('$t [ms]$')
    pylab.legend(frameon = False,loc = 'upper right')
    pylab.ylabel('$FF$')
    pylab.savefig('data_fig_condition1_1.png',dpi=400)
    pylab.plot(tcv_two,pylab.nanmean(cv_twos,axis=0),color = cv_two_col,lw = lw,label = '$CV_2$')
    pylab.ylabel('$FF$, $CV_2$')
    pylab.legend(frameon = False,loc = 'upper right')
    pylab.xlim(0,2000)
    pylab.ylim(0,2.)
    pylab.savefig('data_fig_condition1_2.png',dpi=400)



if __name__ == '__main__':
    pass
    
    #make_condition1_scheme_for_slides()
    #make_condition1_data_fig_for_slides()
    #make_3condition_scheme_for_slides()
    


    
    

