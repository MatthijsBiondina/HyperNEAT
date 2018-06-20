import numpy as np


"""
    A single node in a CPPN, designed to dynamically determine its activation within a network

    idx    [int] := number identifying the node in the network
    n_type [str] := Sensor, Hidden, or Output
    func   [lambda] := function describing what this node does with its inputs
"""
class CPPN_Node:
    

    def __init__(self,
                 idx=None,
                 n_type="unknown",
                 func=None
                 ):
        self.parents = []
        self.weights = []
        self.activity = None
        self.idx = idx
        self.n_type = n_type
        self.func = func

    """
    add a parent node to list of parents which output into this node

    node   [CPPN_node] := parent node object
    weight [float] := weight of the link connecting parent node with this node
    """
    def addParent(self,
                  node=None,
                  weight=0
                  ):
        self.parents.append(node)
        self.weights.append(weight)

    """
    activity of input nodes can be manually set to a value based on environment observation

    activity [float] := value to set this node's activity to
    """
    def setActivity(self,
                    activity=None):
        assert self.n_type == "Sensor" #only sensor nodes can be manually set
        self.activity = activity

    def getLambdaFunction(self):
        if self.n_type == "Sensor":
            if self.idx == 0:
                return "_l_in"
            if self.idx == 1:
                return "_l_out"
            else:
                return "c{}".format(self.idx-2)
        elif self.n_type == "Output":
            return None if len(self.parents)==0 else "({}+0.)".format('+'.join([str(weight) + "*" + parent.getLambdaFunction() for weight,parent in zip(self.weights,self.parents)])+'+0.')
        else:
            return "({}+0.)".format(self.func.format('+'.join([str(weight) + "*" + parent.getLambdaFunction() for weight,parent in zip(self.weights,self.parents)])+'+0.'))

    """
    dynamically retrieve this nodes activity in the current activation of the network
    check whether activity has been determined already and return activity
    if activity is not yet known, determine activity as a function of this node's parent nodes'
    activity multiplied with the corresponding link weights
    """
    def getActivity(self):
        if self.activity == None:
            self.activity = self.func(sum( parent.getActivity() for weight,parent in zip(self.weights,self.parents) ))
        return self.activity

    """
    reset activity to None before a new activation of the CPPN
    """
    def flush(self):
        self.activity = None

    """
    Getter method for node id
    """
    def getId(self):
        return self.idx
            
