import numpy as np


class ParamField:
    """
    Object which contains a meshgrid of N parameters and keeps track of the parameter values which were already sampled
    from it.
    """
    def __init__(self, Constructor):
        """
        Initalizes Parameter field
        Parameter:
           Constructor: Tuple of N-Tuples which describe the parameter to construct the field from.
           Example: NEST
           Constructor = (
            ('Ix', [x1, x2, x3, ... xn], ('I_X_E',)),
            ('Ix', [y1, y2, y3, ... ym], ('I_X_I',)),
            )
            The Ix at the beginning and the I_X_.. are just decriptive to match the format used by GeNN-Gridsearches.
            E.g. Ix could be changed to the unit used for this parameter. I_X_E and I_X_I.
            The order of them is also the order in the outputted parameters if samples are taken from the field.
            Example: GeNN
            Constructor=(
            ('Ix',[x1, x2, x3, ... xn], (Pops[0],)),
            ('Ix',[y1, y2, y3, ... ym], (Pops[1],)),
            )
            The 'Ix' gives the name of the global parameter to be changed in the simulation and the Pops[0] or Pops[1]
            the populations targeted by these global parameters. The list in the middle are the values to construct the
            field from.
        """
        self.shape = tuple([len(dim[1]) for dim in Constructor])
        self.paramField = np.empty(self.shape, dtype=object)
        self.bitfield = np.zeros(self.shape, dtype=bool)
        # Create Indexing array for parameter field and fill it with the parameter tuples.
        for idx in DynIndex(self.shape):
            tempTup = []
            for dimId, elementId in enumerate(idx):
                for Pop in Constructor[dimId][2]:
                    tempTup.append(Constructor[dimId][1][elementId])
            self.paramField[idx] = tempTup
            self.Vectorbase = [(dim[0], Pop) for dim in Constructor for Pop in dim[2]]

    def get_data(self, idx):
        """
        Gets the parameter combination of a specific coordinate, but does not change the status of the bitfield
        -> Coordinate is not marked as sampled afterwards if it hasn't been before.
        """
        return (self.paramField[idx])

    def get_paramField(self):
        """
        Returns the whole parameter field (Array of tuples containing the parameter for that coordinate).
        """
        return self.paramField

    def print(self):
        """
        Prints first the bitfield. (False -> Coordinate not sampled)
        Than the parameter field.
        """
        print(self.bitfield)
        print(self.paramField)

    def get_UnusedIdx(self):
        """
        Returns the coordinates which haven't been sampled from.
        """
        return np.argwhere(self.bitfield == False)

    def get_Params(self, Idxs, Ignore=False):
        """
        Return by Idxs requested parameters. If Ignore is false set coordinates as sampled afterwards and check if they
        were sampled before
        """
        Params = []
        for id in Idxs:
            if Ignore == False:
                assert self.bitfield[id] == False, "Param already pulled"
                self.bitfield[id] = True
            Params.append(self.paramField[id])
        return np.transpose(np.array(Params))

    def get_Vectorbase(self):
        """
        Return vectorbase which is the combination of the name of a global parameter and the population
        for each dimension.
        (NEST returns the combination of the first and third part of the constructor -> Useful for the name and the unit)
        """
        return self.Vectorbase

    def randomSample(self, NumberSamples: int):
        """
        Samples randomly NumberSamples Samples in the parameter field.
        Returns:
            -1: no samples left
             1: Not enough samples left to return the  requested number. THe lower number of samples is returned and
                marked as sampled.
        """
        Ids = self.get_UnusedIdx()
        if Ids.size == 0:
            return -1, ([], []), Ids
        elif len(Ids) < NumberSamples:
            Ids = Ids[np.random.choice(Ids.shape[0], len(Ids), replace=False)]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 1, (self.get_Vectorbase(), Parms), Ids
        else:
            Ids = Ids[np.random.choice(Ids.shape[0], NumberSamples, replace=False)]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 0, (self.get_Vectorbase(), Parms), Ids

    def axisSample(self, NumberSamples: int, axis=0):
        """
        Samples NumberSamples Samples along a given axis in the parameter field.
        Returns:
            -1: no samples left
             1: Not enough samples left to return the  requested number. THe lower number of samples is returned and
                marked as sampled.
        """
        if (self.shape[axis] % NumberSamples):
            print("Number of Samples does not fit to the dimensions of the parameter field - Batches not alligned")
        Ids = self.get_UnusedIdx()

        if Ids.size == 0:
            return -1, ([], []), Ids
        else:
            SortTuple = tuple([Ids[:, (jj + axis) % len(Ids[0])] for jj in range(len(Ids[0]))])
            Ids = Ids[np.lexsort(SortTuple)]

        if len(Ids) < NumberSamples:
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 1, (self.get_Vectorbase(), Parms), Ids
        else:
            Ids = Ids[0:NumberSamples]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 0, (self.get_Vectorbase(), Parms), Ids

    def AllSample(self):
        """
        Returns all samples which are not yet sampled and marks them as sampled.
        """
        Ids = self.get_UnusedIdx()
        if Ids.size == 0:
            return -1, [], Ids
        else:
            Parms = self.get_Params([tuple(Ids[0])])
            return 0, Parms, Ids[0]


def DynIndex(stop):
    """
    Generator function which yields indexing tuples dynamically of a N dimensional array.
    Parameter:
        stop: tuple of N numbers which represent the shape of the N dimensional array
    """
    dims = len(stop)
    if not dims:
        yield ()
        return
    for outer in DynIndex(stop[1:]):
        for inner in range(0, stop[0]):
            yield (inner,) + outer
