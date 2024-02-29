import os
from Helper import GeneralHelper
from Helper import ParamField
import time
import pickle
from pathos.multiprocessing import ProcessPool
import multiprocessing

class Gridsearch:
    """
    General class to setup and run a gridsearch.
    """
    def __init__(self, params, Constructor, measurementVar, default, PathOutput="Measurement.pkl", PathSpikes=None):
        """
        Init Gridsearch object, which contains the parameter field used for the gridsearch, the names of the measurements
        which are reported back. The useds parameters are merged with the default values and saved as well.
        """
        startInitGrid = time.time()
        if os.path.exists(PathOutput):
            os.remove(PathOutput)
        if PathSpikes is not None:
            if os.path.exists(PathSpikes):
                os.remove(PathSpikes)
        self.ParamField = ParamField.ParamField(Constructor)
        self.Parameter = GeneralHelper.mergeParams(params, default)
        self.measurementVar = measurementVar
        self.OutputPath = PathOutput
        self.SpikesPath = PathSpikes
        endInitGrid = time.time()
        self.Timing = {'InitGrid': endInitGrid - startInitGrid}

    def getTiming(self):
        """
        Returns the timing information of the gridsearch.
        """
        return self.Timing

    def getParamField(self):
        """
        Returns the parameter field used in the gridsearch.
        """
        return self.ParamField.get_paramField()


class Gridsearch_NEST(Gridsearch):
    """
        Gridsearch with NEST. Adds Nworkers argument controlling the number of parallel simulations running and the
        simulation function itself.
    """
    def __init__(self, simFun, params, Constructor, measurementVar, default, PathOutput="Measurement.pkl",
                 PathSpikes=None, Nworkers=6):

        startInitGrid = time.time()
        super().__init__(params, Constructor, measurementVar, default, PathOutput, PathSpikes)
        self.Parameter['Nworker']=Nworkers
        self.SimulationFunction=simFun
        with open(self.OutputPath, 'ab') as outfile:
            pickle.dump(self.measurementVar
                        , outfile)
            pickle.dump(self.getParamField()
                        , outfile)
            pickle.dump(self.Parameter, outfile)
        endInitGrid = time.time()
        self.Timing = {'InitGrid': endInitGrid - startInitGrid}

    def search(self):
        """
        Run the Gridsearch and set the timing information.
        """
        Parameterlist = []
        rv, Parm, Ids = self.ParamField.AllSample()
        while rv != -1:
            Parameterlist.append((Parm, Ids))
            rv, Parm, Ids = self.ParamField.AllSample()
        m = multiprocessing.Manager()
        lock = [m.Lock(), m.Lock()]

        with ProcessPool(nodes=self.Parameter['Nworker']) as p:
            TimesL = p.map(lambda x: self.SimulationFunction(self.Parameter, x[0], self.measurementVar, x[1],
                                                             self.OutputPath, lock, timeout=7200), Parameterlist)
        BuildTimes = []
        CompileTimes = []
        LoadTimes = []
        SimTimes = []
        DownloadTimes = []

        for Times in TimesL:
            BuildTimes.append(Times["Build"])
            CompileTimes.append(Times["Compile"])
            LoadTimes.append(Times["Load"])
            SimTimes.append(Times["Sim"])
            DownloadTimes.append(Times["Download"])
        self.Timing['Build'] = BuildTimes
        self.Timing['Compile'] = CompileTimes
        self.Timing['Load'] = LoadTimes
        self.Timing['Simulation'] = SimTimes
        self.Timing['Download'] = DownloadTimes

