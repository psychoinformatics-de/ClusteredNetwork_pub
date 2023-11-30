import numpy as np
import pandas as pd
import ast 

def find(condition):
    """old pylab.find function"""
    res, = np.nonzero(np.ravel(condition))
    return res


def key_from_params(params, reps=None,ignore_keys=['']):
    """create string keys from parameters dictionary"""
    key_list = [k for k in sorted(params.keys()) if k not in ignore_keys]
    key=''
    for k in key_list:
        key += '_'+params[k].__repr__()
    key =key.replace(' ','')
    if reps!=None:
        key =  [key+'_'+str(r) for r in range(reps)]
    return key


def compare_key(all_keys, params):
    """change string key to dictionary and compare with a given parametr dict"""
    import ast
    for i in list(all_keys):
        test_string = '{' + i.split('{')[1].split('}')[0] + '}'
        print('test_str',test_string)
        print(params)
        try:
            dict_key_temp = ast.literal_eval(test_string)
            if dict_key_temp == params:
                return i
        except:
            pass




def load_data(datapath, datafile,params, old_key_code=False, 
                ignore_keys=[''], reps=None):
    """load pickle format data using pandas"""
    all_results = pd.read_pickle(datapath + datafile)
    if old_key_code:
        key = key_from_params(params,reps=reps,ignore_keys=ignore_keys)
        if reps != None:
            result = [all_results[result_key] for result_key in key]
        else:
            result = all_results[key]
    else:
        k = compare_key(all_results.keys(), params)
        print('k', k)
        result = all_results[k]
    return result

def extract_info_from_keys(params, keys):
    """extract info from string keys"""
    keys_dic = keys[keys.find('{')+1: keys.find('}')]
    for i, k in enumerate(keys_dic.split(",'")):
        if i ==0:
            params.update(ast.literal_eval("{"+k+'}'))
        else:
            params.update(ast.literal_eval("{'"+k+'}'))

    return params
