

import pylab

import os
import pickle as pickle
from copy import deepcopy
import unittest
from joblib import Parallel,delayed
import multiprocessing
import collections
import functools
import pandas as pd
import time
from general_func import *


class memoized(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            print('uncacheable: ',args)
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

class memoized_but_forgetful(object):
    '''Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args,**kwargs):

        key = (args, frozenset(sorted(kwargs.items())))
        if not isinstance(key, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args,**kwargs)
        if key in self.cache:
            return self.cache[key]
        else:
            # forget old value
            self.cache = {}
            value = self.func(*args,**kwargs)
            self.cache[key] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)
datapath = '../data/'
#print('datapath', datapath)
@memoized
def get_data_file(filename):
    try:
        all_results = pd.read_pickle(filename)
    except:
        all_results = {}
    return all_results


def params_from_keys(key,key_list):
    items = [eval(i.replace('array','')) for i in key.split('_')[1:]]
    return dict(list(zip(key_list,items)))

def check_and_execute_hetero(param_list,func,datafile,redo = False,n_jobs = 1):
    full_datafile = os.path.join(datapath,datafile)
    all_results = get_data_file(full_datafile)
    keys =[]
    return_keys = []
    for params in param_list:
        key_list = sorted(params.keys())
        keys.append(key_from_params(params,key_list))
        return_keys.append(keys[-1])
    if not redo:
        all_keys = list(all_results.keys())
        drop_inds = []
        for i,k in enumerate(keys):
            if k in all_keys:
                drop_inds.append(i)
        for i in drop_inds[::-1]:
            keys.pop(i)
            param_list.pop(i)
    if len(param_list)>0:
        get_data_file.cache = {}
        print(len(keys),' to generate')
        
        if n_jobs ==1:
            results = list(map (func, param_list))
        else:
            results = Parallel(n_jobs)(delayed(func)(deepcopy(p)) for p in param_list)

        

        for k,r in zip(keys,results):
            all_results[k] = r
        pickle.dump(all_results,open(full_datafile,'wb'),protocol = 2)
    
    return [all_results[k] for k in return_keys]

    
def check_and_execute(params,func,datafile,key_list=None,reps = None,
                    redo = False,backup_file = None,n_jobs = 1,
                    save = True,ignore_keys = []):
    # if 'sim_params' in params.keys():
    #     params = deepcopy(params['sim_params'])
    key = key_from_params(params,reps,ignore_keys)

    datapath = '../data/'
    full_datafile = os.path.join(datapath,datafile)                                                      
    try:
        if redo:
            raise ValueError
        #try:
        all_results = pd.read_pickle(full_datafile)
        if reps is None:
            if key not in all_results.keys():
                key = compare_key(all_results.keys(), params)
            else:
                pass

        if reps is None and not redo:
            results = all_results[key]
            result_keys = [key]
            
        elif not redo:
            result_keys =  key#[key+'_'+str(r) for r in range(reps)]
            results = [all_results[result_key] for result_key in result_keys]
        elif redo:
            raise 
    except:
        cache_key = (full_datafile,)
        if cache_key in list(get_data_file.cache.keys()):
            get_data_file.cache.pop(cache_key)
        try:
            all_results = pd.read_pickle(full_datafile)
        except:
            all_results = {}  
            if save:
                pickle.dump(all_results,open(full_datafile,'wb'),protocol = 2)
        if reps is None:
            results = func(params)
            all_results[key] = results
        else:
            # if not redo:
            #     all_keys = sorted([k for k in list(all_results.keys()) if key in k])
            #     try:
            #         all_keys.remove(key)
            #     except:
            #         pass
            #else:
                #all_keys = []
            all_keys = []
            #### REMOVE COMMEN: added str to key

            #possible_keys = [key+'_'+str(r) for r in range(reps)]
            possible_keys = key#[str(key)+'_'+str(r) for r in range(reps)]
            missing_keys = [k for k in possible_keys if k not in all_keys]

            if n_jobs>1:
                print('n_jobs: ', n_jobs)
                print('careful: in parallel, randseeds need to be set')
                copied_params = [deepcopy(params) for mk in missing_keys]


                # this sometimes breaks, so we need to catch the error and try again
                max_retries = 2
                retries = 0
                while retries < max_retries:
                    try:
                        new_results = Parallel(n_jobs)(
                            delayed(func)(cp) for cp in copied_params)  # Call your parallel task function
                        break  # If successful, break out of the loop
                    except Exception as e:
                        print(f"Attempt {retries + 1} failed:", e)
                        retries += 1
                        if retries < max_retries:
                            print("Retrying...")
                            time.sleep(2)  # Wait for a moment before retrying
    
    

                for mk,nr in zip(missing_keys,new_results):
                    all_results[mk] = nr
            else:
                for mk in missing_keys:
                    print('loop')
                    all_results[mk] = func(deepcopy(params))
                    if save:
                        pickle.dump(all_results,open(full_datafile,'wb'),
                                    protocol = 2)
            
            all_keys = sorted([k for k in list(all_results.keys()) if k in key])
            results = [all_results[k] for k in all_keys[:reps]]
            
        if save:
            print('saving results of ', func)
            pickle.dump(all_results,open(full_datafile,'wb'),protocol = 2)

    if backup_file is not None:
        result_dict = {}
        try:
            if reps is not None:
                for rk,r in zip(result_keys,results):
                    result_dict[rk] = r
            else:
                raise
        except:
            result_dict[key] = results
        pickle.dump(result_dict,open(backup_file,'wb'),protocol= 2)

    return results
    

def _recursive_in(var,val):
    if hasattr(var,'__iter__'):
        return True in [_recursive_in(v,val) for v in var]
    else:
        return var==val

def contains_nans(var):
    if hasattr(var,'__iter__'):
        return [_recursive_in(contains_nans(v),True) for v in var]
        

    return pylab.isnan(var)



class PackagedArray(object):
    def __init__(self,array):
        self.shape = array.shape
        flat_array = array.flatten()
        self.flat_shape = flat_array.shape
        self.inds = find(flat_array!=0).astype(pylab.int32)
        self.data = flat_array[self.inds]
   
    def unpack(self):
        # reconstruct dense array
        flat_array = pylab.zeros(self.flat_shape,dtype=self.data.dtype)
        flat_array[self.inds] = self.data
        return flat_array.reshape(self.shape)


class TestPackagedArray(unittest.TestCase):
    def test_binary(self):
        spikes = pylab.randint(0,2,(10,1000,1000)).astype(bool)
        package = PackagedArray(spikes)
        unpacked = package.unpack()
        self.assertTrue((spikes==unpacked).all())

    def test_float(self):
        spikes = pylab.randint(0,2,(10,20,100)).astype(float)*pylab.randn(10,20,100)
        package = PackagedArray(spikes)
        unpacked = package.unpack()
        self.assertTrue((spikes==unpacked).all())

if __name__ == '__main__':
    unittest.main()
