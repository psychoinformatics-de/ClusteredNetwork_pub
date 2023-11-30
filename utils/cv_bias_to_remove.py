import pylab
import os
from scipy.special import gamma
from scipy.integrate import quad
from scipy import vectorize
from gamma_generator import generate_gamma
import spiketools
from organiser import memoized
from scipy.optimize import fmin
import pickle as pickle


current_path = os.path.abspath(__file__)
full_path = os.path.join(os.path.split(current_path)[0],'..','data')
#full_path = os.path.split(os.path.abspath(__file__))[0]
cv_bias_lookup_fname = 'cv2_bias_lookup'

def gamma_pdf(x,rho=1,alpha = 1):
    return 1/gamma(alpha)*rho *(rho*x)**(alpha-1)*pylab.exp(-rho*x)
#@memoized
def eta(T,rho = 1,alpha = 1):
    def func(s):
        return (T-s)*gamma_pdf(s,rho,alpha)
    return quad(func, 0, T)[0]

def gamma_pdf_hat(x,T=10,rho=1,alpha = 1):
    #print eta(T,rho,alpha)
    return (T-x)*gamma_pdf(x,rho,alpha)/eta(T,rho,alpha)


def pdf_mean(pdf,lower,upper,kwargs = {}):
    def func(x):
        return x*pdf(x,**kwargs)
    return quad(func, lower, upper)[0]

def pdf_var(pdf,lower,upper,kwargs = {},mu =None):
    if mu is None:
        mu = pdf_mean(pdf, lower, upper,kwargs=kwargs)
    def func(x):
        return (x-mu)**2 * pdf(x,**kwargs)
    return quad(func, lower, upper)[0]

def gamma_cv2(ot,alpha):
    kwargs = {'alpha':alpha,'rho':alpha,'T':ot}
    m = pdf_mean(gamma_pdf_hat, 0, ot,kwargs)
    v = pdf_var(gamma_pdf_hat, 0, ot,kwargs,mu = m)
    return v/m**2

def correct_cv2(measured_cv2,ot):
    if not pylab.isfinite(measured_cv2):
        return measured_cv2
    def minfunc(alpha):
        return (measured_cv2-gamma_cv2(ot, float(alpha)))**2
    
    result = fmin(minfunc,1/measured_cv2,disp = False)

    return 1/result
#@memoized
def _get_cv_file(fname):
    
    try:
        look_up = pickle.load(open(fname,'rb'),encoding='latin-1')
    except:
        look_up = {}
    return look_up
@vectorize    
def unbiased_cv2(measured_cv2,ot,precission=3):
    """ same as correct_cv2 but with file memory."""
    if not pylab.isfinite(measured_cv2):
        return measured_cv2
    fname = os.path.join(full_path,cv_bias_lookup_fname)
    look_up = _get_cv_file(fname)
    key = (round(measured_cv2,precission),round(ot,precission))
    try:
        return look_up[key]
    except:
        print('adding cv bias entry for ',key)
        try:
            look_up[key] = correct_cv2(key[0], key[1])
        except:
            look_up[key] = pylab.nan
        pickle.dump(look_up,open(fname,'wb'),protocol = 2)
        _get_cv_file.cache = {}
        return look_up[key]

if __name__ == '__main__':
    x = pylab.linspace(0,1, 1000)

    rate = 10.
    isi = 1/rate*1000.
    cv2 = 1.

    alpha = 1/cv2
    rho = alpha




    nbins = 1000*10000
    rates = pylab.ones((100,nbins/100))*rate
    time = pylab.arange(rates.shape[1])
    spikes = generate_gamma(alpha, rates)
    spiketimes = spiketools.binary_to_spiketimes(spikes, time)

    spike_list = spiketools.spiketimes_to_list(spiketimes)

    intervals = pylab.array([item for sublist in [pylab.diff(sl) for sl in spike_list] for item in sublist])/isi
    print(len(intervals))
    bins = pylab.linspace(0,pylab.nanmax(intervals), 100)
    pylab.hist(intervals,bins,normed = True,histtype = 'stepfilled',facecolor = '0.5',alpha = 0.5)

    pylab.plot(bins,gamma_pdf(bins,alpha = alpha,rho = rho),'k',linewidth = 1.5)



    ot = 5
    pylab.axvline(ot,linestyle = '--',color = 'k')
    window = int(ot*isi)
    rates = pylab.ones((nbins/window,window))*rate

    time = pylab.arange(rates.shape[1])
    spikes = generate_gamma(alpha, rates)
    spiketimes = spiketools.binary_to_spiketimes(spikes, time)

    spike_list = spiketools.spiketimes_to_list(spiketimes)

    intervals = pylab.array([item for sublist in [pylab.diff(sl) for sl in spike_list] for item in sublist])/isi
    print(len(intervals))
    pylab.hist(intervals,bins,normed = True,histtype = 'stepfilled',facecolor = '0.2',alpha = 0.5)
    cv2_ot = spiketools.cv2(spiketimes,pool=True,bessel_correction=True)



    x = pylab.linspace(0,ot,1000)
    ot_pdf = gamma_pdf_hat(x,T=ot,alpha = 1/cv2,rho = 1/cv2)
    pylab.plot(x,ot_pdf,'k',linewidth = 1.5)
    pylab.plot(x,ot_pdf/(ot-x)*eta(ot,1/cv2,1/cv2),'r',linewidth = 1.5)


    kwargs = {'T':ot,'alpha':1/cv2,'rho':1/cv2}
    mean = pdf_mean(gamma_pdf_hat, 0, ot,kwargs)
    var = pdf_var(gamma_pdf_hat, 0, ot,kwargs)

    print(mean,var,cv2_ot,var/mean**2)
    pylab.show()
