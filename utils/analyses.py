import sys;sys.path.append('../utils/')
print(sys.path)
import os
import pickle as pickle
import pylab
import spiketools
import organiser
import joe_and_lili
from organiser import memoized_but_forgetful as memoized
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from time import process_time as clock
import cv_bias
from copy import deepcopy
from general_func import *
current_path = os.path.abspath(__file__)
organiser_path = os.path.join(os.path.split(current_path)[0],'..','data')
organiser.datapath =  organiser_path

#@memoized
def load_data(gn,condition,direction  =None,alignment = 'TS'):
    toc = joe_and_lili.get_toc(filters = [])
    file = toc['file'][(toc['global_neuron']==gn)*(toc['condition'] == condition)]
    ppath = joe_and_lili.pickle_path
    data = pickle.load(open(os.path.join(ppath,file[0].decode()),'rb'),encoding='latin-1')
    if direction is not None:
        spiketimes = data['spiketimes']
        direction_inds = find(spiketimes[2]==direction)
        direction_trials = pylab.unique(spiketimes[1,direction_inds])
        spiketimes = spiketimes[:2,direction_inds]
        for nt,ot in enumerate(pylab.unique(spiketimes[1])):
            spiketimes[1,spiketimes[1]==ot] = nt
        data['spiketimes'] = spiketimes
        data['eventtimes'] = data['eventtimes'][direction_trials.astype(int)]
    

    if alignment != 'TS':
        alignment_column = data['eventtimes'][:,data['event_names'] == alignment]
        spiketimes = data['spiketimes']
        for trial,offset in enumerate(alignment_column):
            spiketimes[0,spiketimes[1]==trial] -= offset
            data['eventtimes'][trial,:] -= offset
        data['spiketimes'] = spiketimes

    return data

def get_ff(gn,condition,direction,window = 400,tlim = [0,2000],alignment = 'TS',redo = False):
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'alignment':alignment,
              'tlim':tlim}
    return organiser.check_and_execute(params, _calc_ff, 'ff_file_'+str(condition)+'_'+str(direction),redo = redo)

def _calc_ff(params):
    
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    ff,tff = spiketools.kernel_fano(spiketimes, params['window'],tlim  =params['tlim'])
    return ff,tff



def get_rate(gn,condition,direction,kernel = 'triangular',sigma = 50.,tlim = [0,2000],alignment = 'TS',redo  =False):
    params = {'gn':gn,'condition':condition,'direction':direction,
              'kernel':kernel,'sigma':sigma,'alignment':alignment,
              'tlim':tlim}
    return organiser.check_and_execute(params, _calc_rate, 'rate_file_'+str(condition)+'_'+str(direction),redo  =redo)

def _calc_rate(params):
    print('calc_rate')
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    if params['kernel'] == 'triangular':
        kernel = spiketools.triangular_kernel(params['sigma'])
    elif params['kernel'] == 'gaussian':
        kernel = spiketools.gaussian_kernel(params['sigma'])

    return spiketools.kernel_rate(spiketimes, kernel,tlim  =params['tlim'])





def _calc_direction_counts(params):
    data = load_data(params['gn'],params['condition'],alignment= params['alignment'])

    spiketimes = data['spiketimes']
    
    counts,time = spiketools.sliding_counts(spiketimes[:2], params['window'],tlim = params['tlim'])
    
    trial_directions = pylab.array([spiketimes[2,find(spiketimes[1]==t)[0]] for t in pylab.sort(pylab.unique(spiketimes[1]))])
    
    
    direction_counts = []
    mean_direction_counts = pylab.zeros((6,counts.shape[1]))
   
    for direction in range(1,7):
        direction_counts.append(counts[trial_directions == direction])

    return direction_counts,time

def _get_direction_counts(gn,condition,window = 400,
                          tlim = [0,2000],alignment = 'TS'):

    params = {'gn':gn,'condition':condition,
              'alignment':alignment,'tlim':tlim,'window':window}

    return organiser.check_and_execute(params, _calc_direction_counts, 
                                       'direction_counts_file',
                                       reps=1,save=False)

def tuning_vector(counts):
    angles = 360 /float(len(counts))
    degrees = pylab.arange(0,360,angles)
    radians = pylab.pi/180.*degrees
    y = counts * pylab.sin(radians)
    x = counts * pylab.cos(radians)
    return x.sum(),y.sum()

def get_tuning(gn,condition,window = 400,tlim=[0,2000],alignment = 'TS',redo =False):
    params = {'gn':gn,'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window}
    return organiser.check_and_execute(params, _calc_tuning, 'tuning_score_file_'+str(condition),redo  =redo)

def _calc_tuning(params):
    
    direction_counts,time = _get_direction_counts(params['gn'], 
                                                  params['condition'],params['window'],
                                                  params['tlim'],params['alignment'])
    mean_direction_counts = pylab.zeros((6,direction_counts[0].shape[1]))
    min_direction_trials = 10000
    for direction in range(1,7):
        
        min_direction_trials = min(min_direction_trials,direction_counts[direction-1].shape[0])
        mean_direction_counts[direction-1] = direction_counts[direction-1].mean(axis = 0)
       

   
    
        
    tuning_score = pylab.zeros((len(time)))
    tuning_vectors = pylab.zeros((2,len(time)))
    for t in range(len(time)):
        tuning_vectors[:,t] = pylab.array(tuning_vector(mean_direction_counts[:,t]))
        diffs = []
        for trial in range(min_direction_trials):
            trial_counts = pylab.array([dc[trial,t] for dc in direction_counts])
            
            trial_tuning = pylab.array(tuning_vector(trial_counts))

            diffs.append(((trial_tuning - tuning_vectors[:,t])**2).sum())

        sigma = pylab.array(diffs).mean()**0.5
    
        tune_length = (tuning_vectors[0,t]**2+tuning_vectors[1,t]**2)**0.5
        tuning_score [t] = tune_length/sigma
    

    return tuning_vectors,tuning_score,time

def balanced_accuray(targets,predictions):
    classes = pylab.unique(targets)
    accuracies = pylab.zeros(classes.shape)
    for i in range(len(classes)):
        class_inds= find(targets == classes[i])

        accuracies[i] =(targets[class_inds]==predictions[class_inds]).mean()
    return accuracies.mean()

def _calc_direction_score(params):
    pylab.seed(None)

    direction_counts,time = _get_direction_counts(params['gn'], params['condition'],params['window'],params['tlim'],params['alignment'])
    
    targets = pylab.zeros((0))
    feature_mat = pylab.zeros((0,len(time)))
    for d,c in zip(list(range(1,7)),direction_counts):
        feature_mat = pylab.append(feature_mat, c,axis=0)
        targets = pylab.append(targets,pylab.ones((c.shape[0]))*d)
    
    # shuffle
    order = pylab.arange(len(targets))
    pylab.shuffle(order)
    targets =targets[order]
    feature_mat = feature_mat[order]
    
    score = pylab.zeros_like(time)

    for i in range(len(time)):
        features = feature_mat[:,[i]]
        predictions = pylab.zeros_like(targets)
        for train,test in StratifiedKFold(targets,n_folds = params['folds']):
            cl = eval(params['classifier'])(**params['classifier_args'])
            cl.fit(features[train],targets[train])
            predictions[test] = cl.predict(features[test])
            #print targets[test],predictions[test]
        score[i] = balanced_accuray(targets, predictions)

    return score,time

def get_direction_score(gn,condition,window = 400,folds = 5,
                        tlim = [0,2000],alignment = 'TS',
                        redo = False,reps = 10,
                        classifier = 'LogisticRegression',
                        classifier_args = {},n_jobs = 1):
    params = {'gn':gn,'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window,'classifier':classifier,
              'folds':folds,'classifier_args':classifier_args}
    result = organiser.check_and_execute(params, _calc_direction_score, 
                                         'direction_score_file_'+str(condition),
                                         redo  =redo,reps =reps,n_jobs = n_jobs)

    time = result[0][1]
    scores = pylab.array([r[0] for r in result])
    return scores.mean(axis=0),time


def _calc_cv2(params):
    t0 = clock()
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    rate  = get_rate(params['gn'],params['condition'], params['direction'],sigma = params['kernel_width'],tlim = params['tlim'],alignment = params['alignment'])
    result =  spiketools.time_warped_cv2(spiketimes,ot=params['ot'],rate = rate,tlim = params['tlim'],kernel_width = params['kernel_width'],nstd=params['nstd'],pool = params['pool'],bessel_correction = params['bessel_correction'],interpolate = params['interpolate'],minvals = params.get('minvals',0),return_all = params.get('per_trial_correction',False))
    print('cv2: ',clock()-t0)
    return result
def get_cv2(gn,condition,direction,ot=10.,kernel_width = 50.,nstd = 3.,pool = False,bessel_correction = True,interpolate = True,tlim = [0,2000],alignment = 'TS',redo = False,minvals = 0):
    
    params = {'gn':gn,'condition':condition,'direction':direction,
              'ot':float(ot),'kernel_width':kernel_width,'nstd':nstd,'pool':pool,
              'bessel_correction':bessel_correction,'interpolate':interpolate,
              'alignment':alignment,'tlim':tlim}
    if minvals>0:
        params['minvals'] = minvals
    return organiser.check_and_execute(params, _calc_cv2, 'cv2_file_'+str(condition)+'_'+str(direction),redo  =redo)

def _calc_corrected_cv2(params):
    t0  =clock()
    print('correcting bias of cv2 for ',params['gn'],params['condition'],params['direction'])
    cv2_params = deepcopy(params)
    cv2_params.pop('precission')
    cv2_params.pop('redo')
    cv2,tcv2 = organiser.check_and_execute(cv2_params, _calc_cv2, 'cv2_file_'+str(params['condition'])+'_'+str(params['direction']),redo  =params['redo'])
    for i in range(len(cv2)):
        cv2[i] = cv_bias.unbiased_cv2(cv2[i],params['ot'],params['precission'])
    if params.get('per_trial_correction',False):
        cv2 = pylab.nanmean(cv2,axis=0)
    print('done ',clock()-t0,' seconds')
    return cv2,tcv2

def get_corrected_cv2(gn,condition,direction,ot=10.,kernel_width = 50.,nstd = 3.,pool = False,bessel_correction = True,interpolate = True,tlim = [0,2000],alignment = 'TS',redo = False,precission = 3,minvals = 0,per_trial_correction  =False):
    
    params = {'gn':gn,'condition':condition,'direction':direction,
              'ot':float(ot),'kernel_width':float(kernel_width),'nstd':nstd,'pool':pool,
              'bessel_correction':bessel_correction,'interpolate':interpolate,
              'alignment':alignment,'tlim':tlim,'precission':precission,'redo':redo}
    if minvals>0:
        params['minvals'] = minvals
    if per_trial_correction:
        params['per_trial_correction'] = True
    return organiser.check_and_execute(params, _calc_corrected_cv2, 'corrected_cv2_file_'+str(condition)+'_'+str(direction),redo  =redo)

def _calc_lv(params):
    t0 = clock()
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    result = spiketools.time_resolved(spiketimes, params['window'],spiketools.lv,kwargs = {'min_vals':params['min_vals']},tlim = params['tlim'])
    print('lv: ',clock()-t0)
    return result 
def get_lv(gn,condition,direction,window = 400,min_vals = 20,tlim = [0,2000],alignment = 'TS',redo  =False):
    
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    
    return organiser.check_and_execute(params, _calc_lv, 'lv_file_'+str(condition)+'_'+str(direction),redo  =redo)
    
    

def _calc_cv_two(params):
    t0 = clock()
    data = load_data(params['gn'],params['condition'],params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    
    
    result = spiketools.time_resolved(spiketimes, params['window'],spiketools.cv_two,kwargs = {'min_vals':params['min_vals']},tlim = params['tlim'])
    
    
    print('cv two: ',clock()-t0)
    return result
def get_cv_two(gn,condition,direction,window = 400,min_vals = 20,tlim = [0,2000],alignment = 'TS',redo  =False):
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    
    return  organiser.check_and_execute(params, _calc_cv_two, 'cv_two_file_'+str(condition)+'_'+str(direction),redo  =redo)

def _calc_full_corrected_cv2(params):
    data = load_data(params['gn'],params['condition'],params['direction'],alignment= params['alignment'])

    spiketimes = data['spiketimes']
    rate  = get_rate(params['gn'],params['condition'], params['direction'],sigma = params['kernel_width'],tlim = params['tlim'],alignment = params['alignment'])
    full_cv2,ot = spiketools.rate_warped_analysis(spiketimes, window='full',rate=rate, func = spiketools.cv2, kwargs={'pool':params['pool'],'bessel_correction':params['bessel_correction']}) 
    
    return cv_bias.unbiased_cv2(full_cv2,ot),ot
def get_full_corrected_cv2(gn,condition,direction,kernel_width = 50.,nstd = 3.,pool = False,bessel_correction = True,interpolate = True,tlim = [0,2000],alignment = 'TS',redo = False,precission = 3,minvals = 0):
    
    params = {'gn':gn,'condition':condition,'direction':direction,
              'kernel_width':float(kernel_width),'nstd':nstd,'pool':pool,
              'bessel_correction':bessel_correction,
              'alignment':alignment,'tlim':tlim,'precission':precission,'redo':redo}
    return organiser.check_and_execute(params, _calc_full_corrected_cv2, 'full_corrected_cv2_file_'+str(condition)+'_'+str(direction),redo  =redo)
def _calc_rate_var(params):
    ff,time = get_ff(params['gn'],params['condition'],params['direction'],window = params['window'],tlim = params['tlim'],alignment = params['alignment'])
        
    if params.get('full_cv2',False):
        cv2,ot = get_full_corrected_cv2(params['gn'], params['condition'], params['direction'],kernel_width = params['rate_kernel'],pool = params['pool'],
                           bessel_correction= params['bessel_correction'],tlim = params['tlim'],alignment = params['alignment'])
    else:
        if params.get('bias_correction',False):
            cv2,tcv2 = get_corrected_cv2(params['gn'], params['condition'], params['direction'],ot = params['ot'],kernel_width = params['rate_kernel'],pool = params['pool'],
                           bessel_correction= params['bessel_correction'],interpolate =True,tlim = params['tlim'],alignment = params['alignment'],minvals = params.get('minvals',0))
        else:
            cv2,tcv2 = get_cv2(params['gn'], params['condition'], params['direction'],ot = params['ot'],kernel_width = params['rate_kernel'],pool = params['pool'],
                           bessel_correction= params['bessel_correction'],interpolate =True,tlim = params['tlim'],alignment = params['alignment'],minvals = params.get('minvals',0))
        
        
        cv2 =pylab.interp(time, tcv2, cv2)
    data = load_data(params['gn'],params['condition'],params['direction'],alignment= params['alignment'])

    spiketimes = data['spiketimes']
    
    counts,tcounts = spiketools.sliding_counts(spiketimes[:2], params['window'],tlim = params['tlim'])
    
    
    counts = pylab.interp(time,tcounts,counts.mean(axis=0))


    return (ff-cv2)*counts/(params['window']/1000.)**2,time

def get_rate_var(gn,condition,direction,ff_window = 400,cv_ot=10.,bessel_correction = True,pool = False,rate_kernel = 50.,tlim = [0,2000],alignment = 'TS',redo  =False,
                 bias_correction = True,minvals = 0,full_cv2 = False):
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':ff_window,'ot':cv_ot,'rate_kernel':rate_kernel,
              'bessel_correction':bessel_correction,'pool':pool,
              'alignment':alignment,'tlim':tlim}
    if bias_correction:
        params['bias_correction'] = True
    if minvals>0:
        params['minvals'] = minvals
    if full_cv2:
        params['full_cv2'] = True
    return organiser.check_and_execute(params, _calc_rate_var, 'rate_var_file_'+str(condition)+'_'+str(direction),redo  =redo)



def _calc_trial_count(params):
    data = load_data(params['gn'],params['condition'],params['direction'])
    spiketimes = data['spiketimes']
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    return len(pylab.unique(spiketimes[1]))

def get_trial_count(gn,condition,direction):
    params = {'gn':gn,'condition':condition,'direction':direction}
    return organiser.check_and_execute(params, _calc_trial_count, 'trial_count_file')
    

def _calc_mean_direction_counts(params):
    data = load_data(params['gn'],params['condition'],params['direction'],alignment = params['alignment'])
    spiketimes = spiketools.cut_spiketimes(data['spiketimes'],tlim  =params['tlim'])
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    trials = len(pylab.unique(spiketimes[1]))
    return spiketimes.shape[1]/float(trials)
def get_mean_direction_counts(gn,condition,direction,alignment = 'TS',tlim=[0,2000]):
    params = {'gn':gn,'condition':condition,'direction':direction,'alignment':alignment,'tlim':tlim}
    return organiser.check_and_execute(params, _calc_mean_direction_counts, 'direction_count_file')
    
def _calc_old_corrected_cv2(params):
    data = load_data(params['gn'],params['condition'],params['direction'],alignment = params['alignment'])
    spiketimes = spiketools.cut_spiketimes(data['spiketimes'],tlim  =params['tlim'])
    rate = get_rate(params['gn'],params['condition'],params['direction'],alignment = params['alignment'],tlim = params['tlim'])
    cv2,tcv2 = spiketools.time_warped_cv2(spiketimes,ot = params['ot'],rate=rate,bessel_correction = False,pool = False)

    cv2 = pylab.array([cv_bias_mb.mean_based_cv_correction(cv, params['ot'])[0] for cv in cv2])
    time = pylab.arange(params['tlim'][0],params['tlim'][1])
    
    if len(cv2)>0:
        cv2 = pylab.interp(time, tcv2, cv2,left = pylab.nan,right = pylab.nan)
    else:
        cv2 = pylab.zeros_like(time).astype(float)*pylab.nan
    return cv2,time

def get_old_corrected_cv2(gn,condition,direction,ot = 10,tlim = [0,2000],alignment = 'TS'):
    params = {'gn':gn,'condition':condition,'direction':direction,'ot':ot,'tlim':tlim,'alignment':alignment}
    return organiser.check_and_execute(params, _calc_old_corrected_cv2, 'old_corrected_cv2_file')

def _calc_population_decoding(params):
    pylab.seed(params.get('randseed',None))
    all_direction_counts = []
    min_trials = pylab.ones((6),dtype = int)*100000
    for gn in params['gns']:
        print('gnsssss', gn)
        direction_counts,time = _get_direction_counts(gn, params['condition'],
                                                      params['window'],params['tlim'],
                                                      params['alignment'])
        for i,d in enumerate(direction_counts):
            min_trials[i] = min(min_trials[i],d.shape[0])
        all_direction_counts.append(direction_counts)
    print('--> min_trials', min_trials)
    gns = params['gns']
    feature_mat = pylab.zeros((0,len(gns),len(time)))
    
    targets = []

    for direction in range(6):
        direction_features = pylab.zeros((min_trials[direction],len(gns),len(time)))
        for i,d in enumerate(all_direction_counts):
            counts = d[direction]
            
            order = pylab.arange(counts.shape[0])
            pylab.shuffle(order)
            direction_features[:,i] = counts[order[:min_trials[direction]]]
        targets += [direction+1 for trial in range(int(min_trials[direction]))]
        feature_mat = pylab.append(feature_mat,direction_features,axis = 0)


    targets = pylab.array(targets)
    print(feature_mat.shape)

    score = pylab.zeros_like(time).astype(float)
    for i in range(len(time)):
        features = feature_mat[:,:,i]
        predictions = pylab.zeros_like(targets).astype(float)
        for train,test in StratifiedKFold(targets,n_folds = params['folds']):
            cl = eval(params['classifier'])(**params['classifier_args'])
            cl.fit(features[train],targets[train])
            predictions[test] = cl.predict(features[test])
        print(i)    
        score[i] = balanced_accuray(targets, predictions)

    return score,time
    



    

def get_population_decoding(gns,condition,window =400.,folds = 5,tlim = [0,2000],
                            alignment = 'TS',redo = False,reps = 10, 
                            classifier = 'LogisticRegression',
                            classifier_args = {},n_jobs = 1):
    params = {'gns':tuple(sorted(gns)),'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window,'classifier':classifier,
              'folds':folds,'classifier_args':classifier_args}
    
    return organiser.check_and_execute(params, _calc_population_decoding, 
                                       'population_decoding_file',
                                       redo = redo,save=False,
                                       reps = reps,n_jobs = n_jobs)
if __name__ == '__main__':
    plot = True
    condition_colors = ['0.','0.3','0.6']
    redo_rate_var = False
    alignment = 'TS'
    tlim =[0,2000]
    for minvals in [0]:
        for ff_window in [400]:
            for monkey in ['joe','lili']:
                if plot:
                    pylab.figure()
                toc = joe_and_lili.get_toc(filters = [], extra_filters = [['monkey','=',monkey]])
                gns = pylab.unique(toc['global_neuron'])
                print(len(gns))
                for condition in [1]:
                    rates = []
                    ffs = []
                    cv2s = []
                    corrected_cv2s = []
                    lvs = []
                    cvtwos = []
                    dirscores = []
                    tuningscores = []
                    ratevars = []
                    for gn in gns:
                        print(gn)
                        dirscore,tdirscore = get_direction_score(gn, condition,window = ff_window)
                        tuningvecs,tuningscore,ttuning = get_tuning(gn, condition,window = ff_window,redo  =False)
                        dirscores.append(dirscore)
                        tuningscores.append(tuningscore)
                        for direction in [6]:#range(1,7):
                            print(condition,gn,direction)
                            rate,trate = get_rate(gn, condition, direction,alignment = alignment,tlim  =tlim)
                            ff,tff = get_ff(gn, condition, direction,window = ff_window,alignment = alignment,tlim  =tlim)
                            cv2,tcv2 = get_cv2(gn, condition, direction,alignment = alignment,tlim  =tlim)
                            corrected_cv2,tcv2 = get_corrected_cv2(gn, condition, direction,minvals = minvals,alignment = alignment,tlim  =tlim)
                            lv,tlv = get_lv(gn,condition,direction,redo  =False,window = ff_window,alignment = alignment,tlim  =tlim)
                            cvtwo,tcvtwo = get_cv_two(gn,condition,direction,redo = False,window = ff_window,alignment = alignment,tlim  =tlim)
                            ratevar,tratevar = get_rate_var(gn, condition, direction,ff_window = ff_window,redo  =redo_rate_var,alignment = alignment,tlim  =tlim)

                            
                            rates.append(rate[0])
                            ffs.append(ff)
                            cv2s.append(cv2)
                            corrected_cv2s.append(corrected_cv2)
                            lvs.append(lv)
                            cvtwos.append(cvtwo)
                            ratevars.append(ratevar)
                            

                    if plot:
                        pylab.subplot(5,1,1)
                        rates = pylab.array(rates)
                        pylab.plot(trate,pylab.nanmean(rates,axis=0),color = condition_colors[condition-1])
                        pylab.xlim(0,2000)
                        pylab.subplot(5,1,2)
                        ffs = pylab.array(ffs)
                        pylab.plot(tff,pylab.nanmean(ffs,axis=0),color = condition_colors[condition-1])
                        cv2s = pylab.array(cv2s)
                        pylab.plot(tcv2,pylab.nanmean(cv2s,axis=0),'--',color = condition_colors[condition-1])
                        corrected_cv2s = pylab.array(corrected_cv2s)
                        pylab.plot(tcv2,pylab.nanmean(corrected_cv2s,axis=0),'--o',color = condition_colors[condition-1])
                        lvs = pylab.array(lvs)
                        pylab.plot(tlv,pylab.nanmean(lvs,axis=0),':',color = condition_colors[condition-1])
                        cvtwos = pylab.array(cvtwos)
                        pylab.plot(tcvtwo,pylab.nanmean(cvtwos,axis=0),'-.',color = condition_colors[condition-1])
                        pylab.xlim(0,2000)
                        pylab.subplot(5,1,3)
                        dirscores = pylab.array(dirscores)
                        pylab.plot(tdirscore,pylab.nanmean(dirscores,axis=0),color = condition_colors[condition-1])
                        pylab.axhline(1/6.,color = 'k',linestyle = '--')
                        pylab.xlim(0,2000)
                        pylab.subplot(5,1,4)
                        tuningscores = pylab.array(tuningscores)
                        pylab.plot(ttuning,pylab.nanmean(tuningscores,axis=0),color = condition_colors[condition-1])
                        pylab.xlim(0,2000)
                        pylab.subplot(5,1,5)
                        ratevars = pylab.array(ratevars)
                        not_nancount = (pylab.isnan(ratevars)==False).sum(axis=0)
                        #pylab.plot(tratevar,nancount)
                        ratevars[:,not_nancount<50] = pylab.nan
                        pylab.plot(tratevar,pylab.nanmean(ratevars,axis=0),color = condition_colors[condition-1])
                        pylab.xlim(0,2000)










    pylab.show()


