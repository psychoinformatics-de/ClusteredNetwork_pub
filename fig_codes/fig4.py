import sys; sys.path.append('../utils')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotting_functions as plotting
import global_params
from general_func import *
#from functions_vi import extract_info_from_keys
import scipy.stats as ss

data_path = '../data/'
file_name_st_sw = 'SponEvoked_Q50_q1_50_Stim5cluster_100trial_1s_spiketimes'
file_name_analysis_sw = 'SponEvoked_Q50_q1_50_Stim5cluster_100trial_1s_analyses'
trial = 1
fig_size = (4,3)
fontsize = 18
# plot cv, ff, rate
Anls = pd.read_pickle(data_path + file_name_analysis_sw)
colors = plt.cm.get_cmap('rainbow', lut = 10)
cl_cnt = np.linspace(0, 1, len(list(Anls.keys())))
amp_lst = []
ff_lst = []
CV_lst = []
rate_lst = []
ff_lst_std = []
CV_lst_std = []
rate_lst_std = []
portion_lst = []
ff_lst1 = []
CV_lst1 = []
rate_lst1 = []
ff_lst_std1 = []
CV_lst_std1 = []
rate_lst_std1 = []

portion_selected = [1,50]
for k_cnt, keys in enumerate(Anls.keys()):
    sel_data = Anls[keys]
    params = {}
    params = extract_info_from_keys(params, keys)
    portion_I = params['portion_I']
    jep = params['jep']
    stim = params['stim_clusters']  # ids of stim clusters
    non_stim = [c for c in range(params['Q']) if c not in stim]
    if portion_I in portion_selected:
        amp_lst.append(params['stim_amp'])
        ff_lst.append(np.nanmean(sel_data['ffs'][stim], 0)[2])
        ff_lst_std.append(ss.sem(sel_data['ffs'][stim], 0,nan_policy='omit')[2])
        rate_lst.append(np.nanmean(sel_data['rates'][stim], 0)[0])
        rate_lst_std.append(ss.sem(sel_data['rates'][stim], 0,nan_policy='omit')[0])        
        CV_lst.append(np.nanmean(sel_data['cv2s'][stim], 0)[0])
        CV_lst_std.append(ss.sem(sel_data['cv2s'][stim], 0,nan_policy='omit')[0])        
        ff_lst1.append(np.nanmean(sel_data['ffs'][non_stim], 0)[2])
        ff_lst_std1.append(ss.sem(sel_data['ffs'][non_stim], 0,nan_policy='omit')[2])        
        rate_lst1.append(np.nanmean(sel_data['rates'][non_stim], 0)[0])
        rate_lst_std1.append(ss.sem(sel_data['rates'][non_stim], 0,nan_policy='omit')[0])
        CV_lst1.append(np.nanmean(sel_data['cv2s'][non_stim], 0)[0])
        CV_lst_std1.append(ss.sem(sel_data['cv2s'][non_stim], 0,nan_policy='omit')[0])
        portion_lst.append(portion_I)
        plt.show()

print(np.shape(sel_data['ffs']))

print(params)
print('stim clusters: ',stim)
print('non stim clusters: ',non_stim)
print('portion_I:', np.unique(portion_lst))
amp_uni = np.unique(amp_lst)
portion_uni = np.unique(portion_lst)
min_r, max_r = int(min(rate_lst)), int(max(rate_lst))
ff_mat = np.zeros((len(amp_uni), len(portion_uni)))
CV_mat = np.zeros((len(amp_uni), len(portion_uni)))
rate_mat = np.zeros((len(amp_uni), len(portion_uni)))
ff_mat_std = np.zeros((len(amp_uni), len(portion_uni)))
CV_mat_std = np.zeros((len(amp_uni), len(portion_uni)))
rate_mat_std = np.zeros((len(amp_uni), len(portion_uni)))
ff_mat1 = np.zeros((max_r, len(portion_uni)))
ff_mat_nonstim = np.zeros((len(amp_uni), len(portion_uni)))
CV_mat_nonstim = np.zeros((len(amp_uni), len(portion_uni)))
rate_mat_nonstim = np.zeros((len(amp_uni), len(portion_uni)))
ff_mat_nonstim_std = np.zeros((len(amp_uni), len(portion_uni)))
CV_mat_nonstim_std = np.zeros((len(amp_uni), len(portion_uni)))
rate_mat_nonstim_std = np.zeros((len(amp_uni), len(portion_uni)))

for i_cnt, i in enumerate(amp_lst):
    amp_idx = np.where(amp_uni == i)[0]
    por_idx = np.where(portion_uni == portion_lst[i_cnt])[0]
    rate_idx = int(rate_lst[i_cnt]) - 1
    ff_mat[amp_idx, por_idx] = ff_lst[i_cnt]
    CV_mat[amp_idx, por_idx] = CV_lst[i_cnt]    
    rate_mat[amp_idx, por_idx] = rate_lst[i_cnt]
    ff_mat_std[amp_idx, por_idx] = ff_lst_std[i_cnt]
    CV_mat_std[amp_idx, por_idx] = CV_lst_std[i_cnt]    
    rate_mat_std[amp_idx, por_idx] = rate_lst_std[i_cnt]

    ff_mat1[rate_idx, por_idx] = ff_lst[i_cnt]
    ff_mat_nonstim[amp_idx, por_idx] = ff_lst1[i_cnt]
    CV_mat_nonstim[amp_idx, por_idx] = CV_lst1[i_cnt]    
    rate_mat_nonstim[amp_idx, por_idx] = rate_lst1[i_cnt]
    ff_mat_nonstim_std[amp_idx, por_idx] = ff_lst_std1[i_cnt]
    CV_mat_nonstim_std[amp_idx, por_idx] = CV_lst_std1[i_cnt]    
    rate_mat_nonstim_std[amp_idx, por_idx] = rate_lst_std1[i_cnt]

cmap = plt.cm.get_cmap('RdGy', lut =8)
XTICKS = np.arange(len(portion_uni))

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00',
                  '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']



fig = plotting.nice_figure(ratio = 0.8,latex_page=global_params.text_width_pts)
fig.subplots_adjust(bottom = 0.1,left = 0.1,top=0.85, wspace = 0.3,hspace=0.4)

nrow,ncol = 2,2
ax1 = plotting.simpleaxis(plt.subplot2grid((nrow,ncol), (0,0), colspan=1))
x_label_val=-0.15
plotting.ax_label1(ax1, 'a',x=x_label_val)
plt.axis('off')

ax1 = plotting.simpleaxis(plt.subplot2grid((nrow,ncol), (1,0), rowspan=1))
FF_ylim = (-2,8)
plotting.ax_label1(ax1, 'c',x=x_label_val)

colors = plt.cm.get_cmap('rainbow')
cl_cnt = np.linspace(0, 1, 2)
lw = 1.
ls = '-'
print('ffmat', ff_mat)
FF = ff_mat-ff_mat[0]
FF_std = ff_mat_std-ff_mat_std[0]
FF_nonstim = ff_mat_nonstim-ff_mat_nonstim[0]
FF_nonstim_std = ff_mat_nonstim_std-ff_mat_nonstim_std[0]
fc ='0.9'
plt.plot(amp_uni, FF[:,-1], lw =lw, label = 'E stim. clusters', c = CB_color_cycle[7])
plt.fill_between(amp_uni, FF[:,-1]-FF_std[:,-1], FF[:,-1]+FF_std[:,-1], facecolor='lightcoral',lw=0, alpha=0.5)
plt.plot(amp_uni, FF_nonstim[:,-1], lw =lw, ls=ls, label = 'E non-stim. clusters', c = CB_color_cycle[1])
plt.fill_between(amp_uni, FF_nonstim[:,-1]-FF_nonstim_std[:,-1], FF_nonstim[:,-1]+FF_nonstim_std[:,-1], facecolor='navajowhite', alpha=0.5,lw=0)


plt.gca().set_title('E clustered network')
plt.gca().set_ylabel(r'$\Delta${FF}')
plt.xlabel('Stim. Amplitude [pA]')
plt.axvline(0.4, ls ='--', lw=0.8, color='gray')
plt.axhline(0., ls ='--', lw=0.8, color='gray')
plt.ylim(-2,20)
plt.yticks([-2,0,5,10,15,20])

ax1 = plotting.simpleaxis(plt.subplot2grid((nrow,ncol), (1,1), rowspan=1))
plotting.ax_label1(ax1, 'd',x=x_label_val)

plt.gca().set_title('E/I clustered network')


plt.plot(amp_uni, FF[:,0], lw = lw, label = 'E/I clustered network', c = CB_color_cycle[0])
print(ff_mat_std[0])
plt.fill_between(amp_uni, FF[:,0]-FF_std[:,0], FF[:,0]+FF_std[:,0],
                 facecolor='lightskyblue', alpha=0.5,lw=0)
plt.plot(amp_uni, FF_nonstim[:,0], lw = lw, ls=ls,label = 'E/I non-stim. clusters', c = CB_color_cycle[5])
plt.fill_between(amp_uni, FF_nonstim[:,0]-FF_nonstim_std[:,0], FF_nonstim[:,0]+FF_nonstim_std[:,0],
                 facecolor='plum', alpha=0.5,lw=0)


plt.gca().set_ylabel(r'$\Delta${FF}')
plt.xlabel('Stim. Amplitude [pA]')
plt.axvline(0.4, ls ='--', lw=0.8, color='gray')
plt.axhline(0., ls ='--', lw=0.8, color='gray')
plt.ylim(-2,1)
ax1 = plotting.simpleaxis(plt.subplot2grid((nrow,ncol), (0,1), rowspan=1))
plotting.ax_label1(ax1, 'b',x=x_label_val)

RATE = rate_mat-rate_mat[0]
RATE_nonstim = rate_mat_nonstim-rate_mat_nonstim[0]
RATE_std = rate_mat_std-rate_mat_std[0]
RATE_nonstim_std = rate_mat_nonstim_std-rate_mat_nonstim_std[0]
plt.plot(amp_uni, RATE[:,0], lw = lw, label = 'E/I stim. clusters', c = CB_color_cycle[0])
plt.plot(amp_uni, RATE_nonstim[:,0], lw = lw, ls=ls, label = 'E/I non-stim. clusters', c = CB_color_cycle[5])
plt.plot(amp_uni, RATE[:,-1], lw =lw, label = 'E stim. clusters', c = CB_color_cycle[7])
plt.plot(amp_uni, RATE_nonstim[:,-1], lw =lw, ls=ls, label = 'E non-stim. clusters', c = CB_color_cycle[1])
plt.axhline(0,ls='--',lw=0.8,color='gray')
plt.gca().set_ylabel(r'$\Delta${rate [1/s]}')
plt.xlabel('Stim. Amplitude [pA]')
plt.axvline(0.4, ls='--', lw=0.8, color='gray')
plt.legend(loc=9,ncol=2,bbox_to_anchor=(.5, 1.48))
plt.savefig('../data/fig_StimAmp0.eps')
import pyx
c = pyx.canvas.canvas()
c.insert(pyx.epsfile.epsfile(0, 0.0, "../data/fig_StimAmp0.eps"))
c.insert(pyx.epsfile.epsfile(1.5, 5.2,"../data/sketch_ff.eps"))
c.writeEPSfile("fig4.eps")  
plt.show()
