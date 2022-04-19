import pylab

def random_pick(x,n=1):
    inds = pylab.arange(len(x))
    pylab.shuffle(inds)
    inds = inds[:n]
    if n==1:
        inds= inds[0]
    return x[inds] 


def stimulus_protocol(condition,trials = 150):
    """
       5  6
    4        1
       3  2
    condition 1:
    1-6 
    condition 2:
    1&2 
    3&4
    5&6 
    condition 3:
    6&1&2 
    3&4&5
    """
    condition_dict = {1:[[1],[2],[3],[4],[5],[6]],
                      2:[[1,2],[3,4],[5,6]],
                      3:[[6,1,2],[3,4,5]]}
    stimuli = pylab.zeros((trials,6,2))
    for trial in range(trials):
        stimulus= pylab.array(random_pick(condition_dict[condition]))
        target = pylab.array(random_pick(stimulus))
        stimuli[trial,stimulus-1,0] = 1
        stimuli[trial,target-1,1] = 1
    return stimuli.astype(bool)


if __name__ == '__main__':
    
    
    for condition in [1,2,3]:
        pylab.figure()
        stimuli = stimulus_protocol(condition,20)
        pylab.subplot(1,2,1)
        pylab.pcolor(-stimuli[:,:,0],cmap = 'gray',edgecolor = 'r',vmin  =-10)
        pylab.title('PS')

        pylab.subplot(1,2,2)
        pylab.pcolor(-stimuli[:,:,1],cmap = 'gray',edgecolor = 'r',vmin  =-10)
        pylab.title('RS')

    pylab.show()