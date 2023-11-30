#import analyses
from matplotlib import pylab
import joe_and_lili
from matplotlib.markers import TICKDOWN
from global_params import colors as global_colors
import organiser
import pickle 
import os
from general_func import find
import spiketools 
from time import process_time as clock
from sklearn.model_selection import StratifiedKFold

off_gray = global_colors['off_gray']
yellow = global_colors['yellow']
green = global_colors['green']
red = global_colors['red']


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


def _calc_cv_two(params):
    """calculate the cv_two for the given params"""
    t0 = clock()
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    
    
    result = spiketools.time_resolved(spiketimes, params['window'],
                                      spiketools.cv_two,
                                      kwargs = {'min_vals':params['min_vals']},
                                      tlim = params['tlim'])
    
    
    print('cv two: ',clock()-t0)
    return result
def get_cv_two(gn,condition,direction,window = 400,min_vals = 20,
               tlim = [0,2000],alignment = 'TS',redo  =False, monkey = b'joe'):
    """try to load the cv_two for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    
    return  organiser.check_and_execute(params, _calc_cv_two, 
                                        monkey.decode("utf-8") + '_cv_two_file_'+str(condition)+'_'+str(direction),redo  =redo)

def get_ff(gn,condition,direction,window = 400,tlim = [0,2000],
           alignment = 'TS',redo = False, monkey = b'joe'):
    """try to load the ff for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'alignment':alignment,
              'tlim':tlim}
    return organiser.check_and_execute(params, _calc_ff, 
                                       monkey.decode("utf-8") + '_ff_file_'+str(condition)+'_'+str(direction),redo = redo)

def _calc_ff(params):
    """calculate the ff for the given params"""
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    ff,tff = spiketools.kernel_fano(spiketimes, params['window'],
                                    tlim  =params['tlim'])
    return ff,tff



def get_rate(gn,condition,direction,kernel = 'triangular',
             sigma = 50.,tlim = [0,2000],alignment = 'TS',
             redo  =False, monkey = b'joe'):
    """try to load the rate for the given gn and direction and 
    alignment if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'kernel':kernel,'sigma':sigma,'alignment':alignment,
              'tlim':tlim}
    return organiser.check_and_execute(
        params, _calc_rate, 
        monkey.decode("utf-8") + '_rate_file_'+str(condition)+'_'+str(direction),
        redo  =redo)

def _calc_rate(params):
    """calculate the rate for the given params"""
    print('calc_rate')
    data = load_data(params['gn'],params['condition'],
        params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    if params['kernel'] == 'triangular':
        kernel = spiketools.triangular_kernel(params['sigma'])
    elif params['kernel'] == 'gaussian':
        kernel = spiketools.gaussian_kernel(params['sigma'])

    return spiketools.kernel_rate(spiketimes, kernel,tlim  =params['tlim'])


def _calc_trial_count(params):
    """calculate the trial count for the given params"""
    data = load_data(params['gn'],params['condition'],params['direction'])
    spiketimes = data['spiketimes']
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    return len(pylab.unique(spiketimes[1]))

def get_trial_count(gn,condition,direction, monkey = b'joe'):
    """try to load the trial count for the given gn and direction and
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction}
    return organiser.check_and_execute(params, 
                                       _calc_trial_count, 
                                       monkey.decode("utf-8") + '_trial_count_file')
    

def _calc_mean_direction_counts(params):
    """calculate the mean direction counts for the given params"""
    data = load_data(params['gn'],params['condition'],
                     params['direction'],alignment = params['alignment'])
    spiketimes = spiketools.cut_spiketimes(data['spiketimes'],tlim  =params['tlim'])
    spiketimes = spiketimes[:,pylab.isfinite(spiketimes[0])]
    trials = len(pylab.unique(spiketimes[1]))
    return spiketimes.shape[1]/float(trials)

def get_mean_direction_counts(gn,condition,direction,
                              alignment = 'TS',tlim=[0,2000], monkey = b'joe'):
    """try to load the mean direction counts for the given gn and direction and
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'alignment':alignment,'tlim':tlim}
    return organiser.check_and_execute(params, _calc_mean_direction_counts, 
                                       monkey.decode("utf-8") + '_direction_count_file')
    
    

def _calc_lv(params):
    """calculate the lv for the given params"""
    t0 = clock()
    data = load_data(params['gn'],params['condition'],
                     params['direction'],params['alignment'])
    spiketimes = data['spiketimes'][:2]
    result = spiketools.time_resolved(spiketimes, params['window'],
                                      spiketools.lv,
                                      kwargs = {'min_vals':params['min_vals']},
                                      tlim = params['tlim'])
    print('lv: ',clock()-t0)
    return result 
def get_lv(gn,condition,direction,window = 400,
           min_vals = 20,tlim = [0,2000],alignment = 'TS',
           redo  =False, monkey = b'joe'):
    """try to load the lv for the given gn and direction and alignment
    if not calculate it"""
    params = {'gn':gn,'condition':condition,'direction':direction,
              'window':window,'min_vals':min_vals,
              'alignment':alignment,'tlim':tlim}
    
    return organiser.check_and_execute(params, _calc_lv, 
                                       monkey.decode("utf-8") + '_lv_file_'+str(condition)+'_'+str(direction),
                                       redo  =redo)
def _calc_direction_counts(params):
    """calculate the direction counts for the given params"""
    data = load_data(params['gn'],params['condition'],
                     alignment= params['alignment'])

    spiketimes = data['spiketimes']
    
    counts,time = spiketools.sliding_counts(spiketimes[:2], 
                                            params['window'],tlim = params['tlim'])
    
    trial_directions = pylab.array(
        [spiketimes[2,find(spiketimes[1]==t)[0]] for t in pylab.sort(
            pylab.unique(spiketimes[1]))])
    direction_counts = []
    mean_direction_counts = pylab.zeros((6,counts.shape[1]))
   
    for direction in range(1,7):
        direction_counts.append(counts[trial_directions == direction])

    return direction_counts,time

def _get_direction_counts(gn,condition,window = 400,tlim = [0,2000],
                          alignment = 'TS'):

    params = {'gn':gn,'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window}

    return organiser.check_and_execute(params, _calc_direction_counts, 
                                       'direction_counts_file',reps = 1)


def _calc_population_decoding(params):
    """calculate the population decoding for the given params"""
    pylab.seed(params.get('randseed',None))
    all_direction_counts = []
    min_trials = pylab.ones((6),dtype = int)*100000
    for gn in params['gns']:
        direction_counts,time = _get_direction_counts(gn, params['condition'],params['window'],params['tlim'],params['alignment'])
        for i,d in enumerate(direction_counts):
            min_trials[i] = min(min_trials[i],d.shape[0])
        all_direction_counts.append(direction_counts)
    print(min_trials)
    gns = params['gns']
    feature_mat = pylab.zeros((0,len(gns),len(time)))
    
    targets = []

    for direction in range(6):
        direction_features = pylab.zeros((min_trials[direction],
                                          len(gns),len(time)))
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
    



    

def get_population_decoding(gns,condition,window =400.,folds = 5,
                            tlim = [0,2000],alignment = 'TS',
                            redo = False,reps = 10, 
                            classifier = 'LogisticRegression',
                            classifier_args = {},n_jobs = 1):
    """try to load the population decoding for the given gn 
    and direction and alignment"""
    params = {'gns':tuple(sorted(gns)),'condition':condition,'alignment':alignment,
              'tlim':tlim,'window':window,'classifier':classifier,
              'folds':folds,'classifier_args':classifier_args}
    return organiser.check_and_execute(
        params, _calc_population_decoding, 'population_decoding_file',
        redo = redo,reps = reps,n_jobs = n_jobs)


def get_stats(gns,min_trials  =10,min_count_rate = 5,min_count=200,
              minvals = 0,alignment = 'TS',tlim = [0,2000],window =400,
              monkey=b'joe'):
    """get the statistics for the given gn and direction and alignment"""
    if monkey != None:
        extra_filters = [('monkey','=',monkey)]
        toc = joe_and_lili.get_toc(extra_filters = extra_filters)
        global_gns = pylab.unique(toc['global_neuron'])

    else:
        global_gns = pylab.unique(joe_and_lili.get_toc()['global_neuron'])
    if gns is None:
        gns = global_gns

    rates = []
    trial_counts = []
    count_rates = []
    ffs = []
    
    lvs = []
    cv_twos = []
    
    count_rate_block = pylab.zeros((len(gns),3,6))
    trial_count_block = pylab.zeros((len(gns),3,6))
    
    for i,gn in enumerate(gns):
        for j,condition in enumerate([1,2,3]):
            for k,direction in enumerate([1,2,3,4,5,6]):
                count_rate_block[i,j,k] =  get_mean_direction_counts(
                    gn,condition,direction,tlim  =tlim,alignment = alignment)
                trial_count_block[i,j,k]  =get_trial_count(gn,condition,direction)
    
    enough_counts = pylab.prod(count_rate_block>=min_count_rate,axis=1)
    enough_trials = pylab.prod(trial_count_block>=min_trials,axis=1)
    
    good_directions = enough_counts * enough_trials
    
    for i,gn in enumerate(gns):
        for j,condition in enumerate([1]):
            for k,direction in enumerate([1,2,3,4,5,6]):
                if good_directions[i,k]:
                    rate,trate = get_rate(gn,condition = 1,direction = direction,
                        tlim  =tlim,alignment = alignment,monkey=monkey)
                    rates.append(rate[0])
                    trial_counts.append(get_trial_count(gn,1,direction,monkey=monkey))
                    count_rates.append(get_mean_direction_counts(gn,1,
                                                                direction,tlim  =tlim,
                                                                alignment = alignment,
                                                                monkey=monkey))
                    ff,tff = get_ff(gn,condition = 1,
                                            direction = direction,window = window,
                                            tlim  =tlim,alignment = alignment
                                            ,monkey=monkey)
                    ffs.append(ff)

                    lv,tlv = get_lv(gn,condition = 1,
                                            direction = direction,tlim  =tlim,
                                            alignment = alignment
                                            ,monkey=monkey)
                    lvs.append(lv)
                    cv_two,tcv_two = get_cv_two(gn,condition = 1,
                                                        direction = direction,tlim  =tlim,
                                                        alignment = alignment
                                                        ,monkey=monkey)
                    cv_twos.append(cv_two)

    rates = pylab.array(rates)        
    ffs = pylab.array(ffs)
    lvs = pylab.array(lvs)
    cv_twos = pylab.array(cv_twos)
    return tff,ffs,tlv,lvs,tcv_two,cv_twos, trate, rates


def draw_hex_array(center,size=0.3,colors = [[0.5]*3]*7,axes = None,
    radius = 0.1,add = True,show_numbers = False,draw_center = True,lw = 1., epoch=None):
    """draw a hexagonal array of circles with the given center and size"""
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

 
    
def plot_experiment(size,radius,direction =1,lw = 1.,y_pos = 0,condition = 1,write_epoch=False):
    """plot the experimental protocl"""
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


