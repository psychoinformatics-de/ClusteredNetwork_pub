'''
Created on Oct 23, 2012

@author: thomas
'''

from scipy.stats import gamma
import numpy as np
import unittest
import spiketools


def generate_gamma(k,rates,warmup_isis = 10):
    if warmup_isis is None:
        return _generate_gamma(k, rates)
    else:
        # add a warmup time
        initial_rates = rates[:,0]
        try:
            min_rate = rates[rates>0].min()
            max_isi = 1/min_rate*1000.
            warmup_samples = int(warmup_isis*max_isi)
        except:
            #rates are zero: no point in warming up
            warmup_samples = 1
        # warmup samples should not be more than say 100000
        warmup_samples = int(min(warmup_samples,1e5))
        warmup_rates = np.array([initial_rates]*warmup_samples).T
        full_rates = np.append(warmup_rates, rates,axis=1)
        full_spikes =  _generate_gamma(k,full_rates)
        return full_spikes[:,-rates.shape[1]+1:]


def _generate_gamma(k,rates):
    if np.isscalar(k):
        k = np.array([k])
    if len(k)!=1:
        assert len(k)==rates.shape[0],'k must be a scalar or a vector of length N_trials'
    # inflate k to vector
    if len(k)==1:
        k = np.ones((rates.shape[0],))*k
    # if rates and k are constant across trials, just generate in one go
    if rates.var(axis = 0).sum()==0 and k.var()==0:
        l = rates[0,:]/1000.
        n = rates.shape[0]
        k = k[0]
        return randg_equilibrium(k,l,n)
    else:
        spikes = np.zeros_like(rates)[:,:-1]
        for i in range(rates.shape[0]):
            l = rates[i,:]/1000.
            spikes[i,:] = randg_equilibrium(k[i],l,1)
        
        return spikes

def randg(k,l,n):
    # make sure k is float
    k=float(k)
    # integrate rate
    L = np.cumsum(l)
    N = L[-1]
    
    W = 5. * np.ceil(k)
    M = 5. * np.sqrt(N)
    
    R = gamma.rvs(k,scale = 1/k,size=(np.ceil(N+M),n))
    
    R = np.cumsum(R,axis = 0)
    spikes = np.zeros((n,len(L)-1))
    for i in range(n):
        spikes[i,:] = np.histogram(R[:,i],bins = L)[0]
    return spikes
    

def randg_equilibrium(k,l,n):
    """ the first interval is drawn from UY,
        where U is uniforml distributed in [0,1]
        and  if from gamma_(k+1,l)
        """
    # make sure k is float
    k=float(k)
    # integrate rate
    L = np.cumsum(l)
    N = L[-1]
    
    W = 5. * np.ceil(k)
    M = 5. * np.sqrt(N)
    
    U = np.random.rand(n)
    Y = gamma.rvs(k+1,scale = 1/k,size=(1,n))
    UY = U*Y

    R = gamma.rvs(k,scale = 1/k,size=(int(np.ceil(N+M)),int(n)))
    R = np.append(UY,R[:-1], axis=0)
    R = np.cumsum(R,axis = 0)
    spikes = np.zeros((n,len(L)-1))
    for i in range(n):
        spikes[i,:] = np.histogram(R[:,i],bins = L)[0]
    return spikes
 


class TestRandG(unittest.TestCase):
    def test_count_consistency(self):
        n = 5000
        
        for r in [1,10,100,200]:
            time = np.arange(1000)
            rate = np.ones_like(time)*r
            k=1.
            l = rate/1000.
            spikes = randg(k,l,n) 
            counts = spikes.sum(axis = 1)
            # count should be about right (test for 5% difference)
            self.assertAlmostEqual( counts.mean(), r,delta = r/20.)
    
    def test_ff_consistency(self):
        n = 5000
        
        for ff in [0.0001,0.1,0.5,1.,1.5]:
            time = np.linspace(0,1000,5000)
            rate = np.ones_like(time)*20
            k=1/ff
            l = rate/5000.
            spikes = randg(k,l,n) 
            
            ff_estimate = spiketools.ff(spiketools.binary_to_spiketimes(spikes, time))
            # ff should be approcimately equal to 1/k
            self.assertAlmostEqual( ff_estimate, ff,delta = 0.1)
            
    def test_cv_ff_consistency(self):
        n = 3000
        
        for ff in [0.0001,0.1,0.5,1.,1.5]:
            time = np.linspace(0,1000,5000)
            rate = np.ones_like(time)*10
            k=1/ff
            l = rate/5000.
            spikes = randg(k,l,n) 
            ff_estimate = spiketools.ff(spiketools.binary_to_spiketimes(spikes, time))
            cv2_estimate = spiketools.cv2(spiketools.binary_to_spiketimes(spikes, time))
            # ff should be approcimately equal to cv^2
            self.assertAlmostEqual( ff_estimate, cv2_estimate,delta = 0.1)
    
    def test_rate_modulation(self):
        n = 5000
        
        time = np.arange(1000)
        rate = np.exp(-(time-500)**2/5000.)*10
        k=1.
        l = rate/1000.
        spikes = randg(k,l,n) 
        spiketimes = spiketools.binary_to_spiketimes(spikes, time)
        e_rate,e_time = spiketools.kernel_rate(spiketimes, spiketools.gaussian_kernel(11.),tlim=[0,1000])
        e_rate = e_rate[0,:]
        len_diff = len(rate)-len(e_rate)
        rate = rate[len_diff/2:-len_diff/2]
        # mean squared error between input and estimate should be small
        error=np.mean((e_rate-rate)**2)
        self.assertTrue(error<0.02)


class TestGenerateGamma(unittest.TestCase):
    def test_rates(self):
        tmax_s = 100. 
        rates = np.zeros((5,1000*tmax_s))
        for i in range(rates.shape[0]):
            rates[i,:] = i+1
        # regular 
        k = 1e30
        spikes = generate_gamma(k,rates)
        counts = spikes.sum(axis = 1)
        self.assertTrue( ((counts-rates.mean(axis=1)  *tmax_s)**2).max()<2)     
if __name__ == '__main__':
    unittest.main()
    
    
    