from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import StratifiedKFold
import sys;sys.path.append('../utils')
import os
abspath = os.path.split(os.path.abspath(__file__))[0]
#abspath = abspath.split('chapters')[0]
print(abspath)
#nest_path = os.path.join(abspath,'chapters/models/nest')
#import sys; sys.path.append(nest_path)
import analyse_nest
import pylab
import spiketools
from simulate_experiment import get_simulated_data
import default
import organiser
from joblib import Parallel,delayed
from general_func import *

from copy import deepcopy


def _calc_fanos(params):

    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    


    unit_spiketimes = result['unit_spiketimes']
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    window = params.get('window',400.)
    tlim = params.get('tlim',[-500,1500])

    alignment = params.get('alignment',None)
    if alignment is not None:
        print('changing alignment')
        # align trials with respect to times given
        events = result['trial_starts']+alignment
        trial_types = pylab.array(result['trial_types'])
        good = pylab.isfinite(events)
        events = events[good]
        trial_types = trial_types[good]
        conditions = trial_types[:,0]
        directions = trial_types[:,1]
        

        trials = pylab.arange(len(directions))


        trial_spiketimes = analyse_nest.cut_trials(result['spiketimes'], events,tlim)
        N_E =sim_params.get('N_E',default.N_E) 
        unit_spiketimes = analyse_nest.split_unit_spiketimes(trial_spiketimes,N_E)
        
        

        trial_directions = dict(list(zip(trials,directions)))
        trial_conditions = dict(list(zip(trials,conditions)))
        
        for u in list(unit_spiketimes.keys()):
            spiketimes = unit_spiketimes[u]
            directions = pylab.array([trial_directions[int(trial)] for trial in spiketimes[1]])
            conditions = pylab.array([trial_conditions[int(trial)] for trial in spiketimes[1]])
            #print spiketimes.shape,directions.shape

            spiketimes = pylab.append(spiketimes, directions[None,:],axis=0)
            spiketimes = pylab.append(spiketimes, conditions[None,:],axis=0)
            unit_spiketimes[u] = spiketimes

        result['unit_spiketimes'] = unit_spiketimes

    T  = tlim[1]-tlim[0]
    print(T)
    time = pylab.arange(tlim[0],tlim[1]).astype(float)
    condition_ffs = {}

    for condition in sim_params['conditions']:
        condition_ffs[condition] = {}
        for unit in list(unit_spiketimes.keys()):
            print(condition,unit)
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition]
            unit_ffs = []
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    unit_ffs.append(pylab.zeros_like(time)*pylab.nan)
                    continue
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    unit_ffs.append(pylab.zeros_like(time)*pylab.nan)
                    continue

                for i,t in enumerate(pylab.sort(direction_trials)):
                    direction_spiketimes[1,direction_spiketimes[1]==t] = i
                
                ff,tff = spiketools.kernel_fano(direction_spiketimes,params.get('ff_window',window),tlim  =tlim)
                unit_ffs.append(pylab.interp(time,tff,ff,left = pylab.nan,right = pylab.nan))
            condition_ffs[condition][unit] = pylab.nanmean(unit_ffs,axis=0)
    condition_ffs['time'] = time
    return condition_ffs


def get_fanos(params,save = True,redo = False,datafile = 'model_fanos'):
    return load_data('../data/', 'model_fanos', params['sim_params'], old_key_code=False, ignore_keys=[''])
    #return organiser.check_and_execute(params, _calc_fanos, datafile,redo  =redo,save = save)

def _calc_cv_two(params):

    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    


    unit_spiketimes = result['unit_spiketimes']
    print(unit_spiketimes)
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    window = params.get('cvtwo_win',400.)
    tlim = params.get('tlim',[-500,1500])
    print('tlim model cv_two', tlim)
    alignment = params.get('alignment',None)
    if alignment is not None:
        print('changing alignment')
        # align trials with respect to times given
        events = result['trial_starts']+alignment
        trial_types = pylab.array(result['trial_types'])
        good = pylab.isfinite(events)
        events = events[good]
        trial_types = trial_types[good]
        conditions = trial_types[:,0]
        directions = trial_types[:,1]
        

        trials = pylab.arange(len(directions))


        trial_spiketimes = analyse_nest.cut_trials(result['spiketimes'], events,tlim)
        N_E =sim_params.get('N_E',default.N_E)
        print('N_E', N_E)
        unit_spiketimes = analyse_nest.split_unit_spiketimes(trial_spiketimes,N_E)
        
        

        trial_directions = dict(list(zip(trials,directions)))
        trial_conditions = dict(list(zip(trials,conditions)))
        
        for u in list(unit_spiketimes.keys()):
            spiketimes = unit_spiketimes[u]
            directions = pylab.array([trial_directions[int(trial)] for trial in spiketimes[1]])
            conditions = pylab.array([trial_conditions[int(trial)] for trial in spiketimes[1]])
            #print spiketimes.shape,directions.shape

            spiketimes = pylab.append(spiketimes, directions[None,:],axis=0)
            spiketimes = pylab.append(spiketimes, conditions[None,:],axis=0)
            unit_spiketimes[u] = spiketimes

        result['unit_spiketimes'] = unit_spiketimes

    T  = tlim[1]-tlim[0]
    print(T)
    time = pylab.arange(tlim[0],tlim[1]).astype(float)
    condition_cv2s = {}

    for condition in sim_params['conditions']:
        condition_cv2s[condition] = {}
        for unit in list(unit_spiketimes.keys()):
            #print condition,unit
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition]
            unit_cv2s = []
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    unit_cv2s.append(pylab.zeros_like(time)*pylab.nan)
                    continue
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    unit_cv2s.append(pylab.zeros_like(time)*pylab.nan)
                    continue

                for i,t in enumerate(pylab.sort(direction_trials)):
                    direction_spiketimes[1,direction_spiketimes[1]==t] = i
                
                cv2,tcv2 = spiketools.time_resolved_cv_two(direction_spiketimes,params.get('cvtwo_win',window),min_vals = 10, tlim  =tlim)
                #cv2,tcv2 = spiketools.time_warped_cv2(direction_spiketimes,tlim =tlim,bessel_correction = True)
                unit_cv2s.append(pylab.interp(time,tcv2,cv2,left = pylab.nan,right = pylab.nan))
            condition_cv2s[condition][unit] = pylab.nanmean(unit_cv2s,axis=0)
    condition_cv2s['time'] = time
    return condition_cv2s





def get_cv_two(params,save = True,redo = False,datafile = 'model_cv2s'):
    return load_data('../data/', datafile, params['sim_params'], old_key_code=False, ignore_keys=[''])
    #return organiser.check_and_execute(params, _calc_cv_two, datafile, redo=redo,save = save)


def _calc_rates(params):

    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    
    unit_spiketimes = result['unit_spiketimes']
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    kernel = params.get('kernel',50.)
    kernel = spiketools.triangular_kernel(kernel)
    tlim = params.get('tlim',[-500,1500])
    T  = tlim[1]-tlim[0]
    print(T)
    time = pylab.arange(tlim[0],tlim[1]).astype(float)
    condition_rates = {}

    for condition in sim_params['conditions']:
        condition_rates[condition] = {}
        for unit in list(unit_spiketimes.keys()):
            #print condition,unit
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition]
            unit_rates = []
            crap = False
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    unit_rates.append(pylab.zeros_like(time)*pylab.nan)
                    continue
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    unit_rates.append(pylab.zeros_like(time)*pylab.nan)
                    continue

                for i,t in enumerate(pylab.sort(direction_trials)):
                    direction_spiketimes[1,direction_spiketimes[1]==t] = i
                
                rate,trate = spiketools.kernel_rate(direction_spiketimes,kernel,tlim  =tlim)
                unit_rates.append(pylab.interp(time,trate,rate[0],left = pylab.nan,right = pylab.nan))
            condition_rates[condition][unit] = pylab.nanmean(unit_rates,axis=0)
    condition_rates['time'] = time
    return condition_rates


def get_rates(params,save = True,redo = False,datafile = 'model_rates'):
    return organiser.check_and_execute(params, _calc_rates, datafile,redo  =redo,save = save)


def balanced_accuray(targets,predictions):
    classes = pylab.unique(targets)
    accuracies = pylab.zeros(classes.shape)
    for i in range(len(classes)):
        class_inds= find(targets == classes[i])

        accuracies[i] =(targets[class_inds]==predictions[class_inds]).mean()
    return accuracies.mean()

"""
def _calc_direction_score(params):
    pylab.seed(None)

    direction_counts,time = _get_direction_counts(params['gn'], params['condition'],params['window'],params['tlim'],params['alignment'])
    
    targets = pylab.zeros((0))
    feature_mat = pylab.zeros((0,len(time)))
    for d,c in zip(range(1,7),direction_counts):
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

"""


def _calc_direction_scores(params):

    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    pylab.seed(None)
    unit_spiketimes = result['unit_spiketimes']
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    window = params.get('window',400.)
    tlim = params.get('tlim',[-500,1500])
    classifier= params.get('classifier',LR)
    classifier_args = params.get('classifier_args',{})
    reps = params.get('reps',10)
    
    folds = params.get('folds',5)
    timestep = params.get('timestep',1)
    T  = tlim[1]-tlim[0]
    

    condition_scores = {}

    for condition in sim_params['conditions']:
        condition_scores[condition] = {}
        for unit in list(unit_spiketimes.keys()):
            
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition][:3].copy()
            condition_trials  = pylab.sort(pylab.unique(condition_spiketimes[1]))
            for i,t in enumerate(condition_trials):
                condition_spiketimes[1,condition_spiketimes[1]==t] = i
            #print condition,condition_spiketimes.shape
            crap = False
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    crap = True
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    crap = True
            if crap:
                print('crap ',unit)
                continue
                
            print(condition,unit)
            trial_directions = pylab.array([condition_spiketimes[2,find(condition_spiketimes[1]==t)[0]] for t in pylab.sort(pylab.unique(condition_spiketimes[1]))])
            
            counts,time = spiketools.sliding_counts(condition_spiketimes[:2], window,tlim = tlim)

            
            targets = trial_directions
            feature_mat = counts

            

            rep_params = {'targets':targets,'feature_mat':feature_mat,'time':time,'classifier':classifier,'classifier_args':classifier_args,'folds':folds,'timestep':timestep}

            unit_scores = Parallel(sim_params.get('n_jobs',1))(delayed(_calc_classification_score)(deepcopy(rep_params)) for r in range(reps))
            #print unit_scores
            time = unit_scores[0][1]
            condition_scores[condition][unit] = pylab.nanmean([u[0] for u in unit_scores],axis=0)
    
    condition_scores['time'] = time
    return condition_scores


def _calc_classification_score(params):
    pylab.seed(None)
    targets = params['targets']
    feature_mat = params['feature_mat']
    time = params['time']
    classifier= params['classifier']
    classifier_args  = params['classifier_args']
    folds = params['folds']

    timestep  =int(params['timestep'])

    order = pylab.arange(len(targets))
    pylab.shuffle(order)
    targets =targets[order]
    feature_mat = feature_mat[order]
    
    score = []
    real_time = []
    for i in range(0,len(time),timestep):
        features = feature_mat[:,[i]]
        predictions = pylab.zeros_like(targets)
        for train,test in StratifiedKFold(n_splits =folds).split(features,targets):
            cl = classifier(**classifier_args)
            
            
            cl.fit(features[train],targets[train])
                           
            predictions[test] = cl.predict(features[test])
        real_time.append(time[i])  
        score.append(balanced_accuray(targets, predictions))
    print('targets info ############')
    print('targets', targets)
    print('shape', pylab.shape(targets))
    return pylab.array(score),pylab.array(real_time)

def get_direction_scores(original_params,save = True,redo = False,datafile = 'direction_score_file'):

    params = deepcopy(original_params)
    
    return organiser.check_and_execute(params, _calc_direction_scores, datafile,redo  =redo,save = save)


def _calc_population_decoding(original_params):
    params  =deepcopy(original_params)
    
    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    pylab.seed(None)
    unit_spiketimes = result['unit_spiketimes']
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    window = params.get('window',400.)
    tlim = params.get('tlim',[-500,1500])
    classifier= params.get('classifier',LR)
    classifier_args = params.get('classifier_args',{})
    reps = params.get('reps',10)
    
    folds = params.get('folds',5)
    timestep = params.get('timestep',1)
    T  = tlim[1]-tlim[0]
    
    condition_scores = {}

    for condition in sim_params['conditions']:
        condition_scores[condition] = {}

        feature_mat =[]
        for unit in list(unit_spiketimes.keys()):
            
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition][:3].copy()
            condition_trials  = pylab.sort(pylab.unique(condition_spiketimes[1]))
            for i,t in enumerate(condition_trials):
                condition_spiketimes[1,condition_spiketimes[1]==t] = i
            #print condition,condition_spiketimes.shape
            crap = False
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    #print 'not enough trials, ',str(unit)
                    crap = True
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    #print 'rate too low: ',unit,count_rateS
                    crap = True
            if crap:
                continue
            print(condition,unit)
            trial_directions = pylab.array([condition_spiketimes[2,find(condition_spiketimes[1]==t)[0]] for t in pylab.sort(pylab.unique(condition_spiketimes[1]))])
            
            counts,time = spiketools.sliding_counts(condition_spiketimes[:2], window,tlim = tlim)

            
            targets = trial_directions
            feature_mat.append(counts)
        
        feature_mat = pylab.array(feature_mat)
        scores = []

        for r in range(reps):
            score = pylab.zeros_like(time).astype(float)
            order = pylab.arange(len(targets))
            pylab.shuffle(order)
            targets = targets[order]
            feature_mat = feature_mat[:,order]
            print(r)
            for i in range(len(time)):
                features = feature_mat[:,:,i].T
                predictions = pylab.zeros_like(targets)
                for train,test in StratifiedKFold(n_splits =folds).split(features,targets):
                    cl = classifier(**classifier_args)
                    cl.fit(features[train],targets[train])
                    predictions[test] = cl.predict(features[test])
                score[i] = balanced_accuray(targets, predictions)
            scores.append(score)
        condition_scores['time'] = time
        condition_scores[condition] = scores    

    return condition_scores
        


def get_population_decoding(params,save = False,redo = False,datafile = 'population_decoding_file'):
    #return organiser.check_and_execute(params, _calc_population_decoding, datafile,redo  =redo,save = save)
    return load_data('../data/', 'population_decoding_file',params['sim_params'], old_key_code=False, ignore_keys=[''])




def _calc_mean_cluster_counts(params):
    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    Q = sim_params.get('Q',6)
    N_E = sim_params.get('N_E',1200)

    cluster_size = N_E/Q
    tlim = params['tlim']

    

    cluster_units = []
    for cluster in range(Q):
        cluster_units.append(list(range(cluster*cluster_size,(cluster+1)*cluster_size)))

    cluster_counts = []

    for units in cluster_units:
        counts,time = spiketools.spiketimes_to_binary(result['unit_spiketimes'][units[0]][:2],tlim = tlim)
        for unit in units[1:]:
            counts += spiketools.spiketimes_to_binary(result['unit_spiketimes'][unit][:2],tlim = tlim)[0]
        cluster_counts.append(counts/float(len(units)))
    direction_clusters = result['direction_clusters']
    
    cluster_counts = [cluster_counts[dc] for dc in direction_clusters ]
    print(cluster_counts[0].shape)
    
    cluster_counts = pylab.array(cluster_counts)
    trial_types = result['trial_types']
    conditions = pylab.array([t[0] for t in trial_types])
    directions = pylab.array([t[1] for t in trial_types])
    return cluster_counts,time,conditions,directions




def get_mean_cluster_counts(params,datafile = 'mean_cluster_count_file'):
    return organiser.check_and_execute(params, _calc_mean_cluster_counts, datafile)



def tuning_vector(counts):
    angles = 360 /float(len(counts))
    degrees = pylab.arange(0,360,angles)
    radians = pylab.pi/180.*degrees
    y = counts * pylab.sin(radians)
    x = counts * pylab.cos(radians)
    return x.sum(),y.sum()

def _calc_tuning(params):

    sim_params = params['sim_params']
    result = get_simulated_data(sim_params)
    pylab.seed(None)
    unit_spiketimes = result['unit_spiketimes']
    min_count_rate = params.get('min_count_rate',7.5)
    min_trials = params.get('min_direction_trials',10)
    window = params.get('window',400.)
    tlim = params.get('tlim',[-500,1500])
    classifier= params.get('classifier',LR)
    classifier_args = params.get('classifier_args',{})
    reps = params.get('reps',10)
    
    folds = params.get('folds',5)
    timestep = params.get('timestep',1)
    T  = tlim[1]-tlim[0]
    
    condition_tunings = {}

    for condition in sim_params['conditions']:
        condition_tunings[condition] = {}
        for unit in list(unit_spiketimes.keys()):
            
            spiketimes = unit_spiketimes[unit]
            condition_spiketimes = spiketimes[:,spiketimes[3]==condition][:3].copy()
            condition_trials  = pylab.sort(pylab.unique(condition_spiketimes[1]))
            for i,t in enumerate(condition_trials):
                condition_spiketimes[1,condition_spiketimes[1]==t] = i
            #print condition,condition_spiketimes.shape
            crap = False
            for direction in range(1,7):
                direction_spiketimes = spiketools.cut_spiketimes(condition_spiketimes[:2,condition_spiketimes[2]==direction],tlim  =tlim)
                direction_trials = pylab.unique(direction_spiketimes[1])
                if len(direction_trials)<min_trials:
                    crap = True
                count_rate = direction_spiketimes.shape[1]/float(len(direction_trials))/float(T)*1000.
                if count_rate<min_count_rate:
                    crap = True
            if crap:
                continue
            print(condition,unit)


            trial_directions = pylab.array([condition_spiketimes[2,find(condition_spiketimes[1]==t)[0]] for t in pylab.sort(pylab.unique(condition_spiketimes[1]))])
            counts,time = spiketools.sliding_counts(condition_spiketimes[:2], window,tlim = tlim)

            direction_counts = []
            mean_direction_counts = []
            min_direction_trials = 10000000000000000
            for direction in range(1,7):
                direction_counts.append(counts[trial_directions==direction])
                mean_direction_counts.append(direction_counts[-1].mean(axis=0))
                min_direction_trials = min(min_trials,direction_counts[-1].shape[0])

            mean_direction_counts = pylab.array(mean_direction_counts)
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

            condition_tunings[condition][unit] = (tuning_vectors,tuning_score)


            
            
    condition_tunings['time'] = time
    return condition_tunings



def get_tuning(params,save = True,redo = False,datafile = 'tuning_file'):
    return organiser.check_and_execute(params, _calc_tuning, datafile,redo  =False,save = save)

if __name__ == '__main__':

    params = {'sim_params':{'randseed':8721,'trials':5}}

    get_fanos(get_population_decoding)

    pylab.show()











