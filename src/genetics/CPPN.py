from genetics.CPPN_Node import *
import math as m
import traceback
import sys
import numpy as np

from tools.tools import log
from tools.Functions import Functions

"""
    Class CPPN:

    A CPPN is a network of canonical functions that take two sets of coordinates, (x1,y1,z1) and (x2,y2,z2),
    and project these coordinates onto a hypercube in Carthesian space, representing the weight between these
    nodes at these coordinates in the derived ANN (substrate).

    genome [dict] := dictionary containing all required information to construct an individual's CPPN
"""
class CPPN:
    
    
    def __init__(self,
                 genome=None,
                 cppn_output_range=None,
                 ):
        self.cppn_nodes   = {}   #dict of all nodes in the CPPN
        self.sensor_nodes = []   #list of input nodes in the CPPN sorted on node_id
        self.output_node  = None #output node of the CPPN
        self.cppn_output_range=cppn_output_range
        
        self.genome = genome
        self.buildCPPN(genome=genome)

    """
    Construct a CPPN from a genome
    """
    def buildCPPN(self,
                  genome=None
                  ):
        for node in genome['nodes']: # make objects for each node
            self.cppn_nodes[node['id']] = CPPN_Node(idx=node['id'],
                                                    n_type=node['type'],
                                                    func=node['func'])
            if node['type'] == 'Sensor': # add to list of sensor nodes for easy access
                self.sensor_nodes.append(self.cppn_nodes[node['id']])
            elif node['type'] == 'Output': # store output node for easy access
                assert self.output_node == None
                self.output_node = self.cppn_nodes[node['id']]
        self.sensor_nodes.sort(key=lambda node: node.getId())

        # add connections between nodes
        for innov_nr in genome['links']:
            if genome['links'][innov_nr]['enabled']:
                self.cppn_nodes[genome['links'][innov_nr]['out']].addParent(node=self.cppn_nodes[genome['links'][innov_nr]['in']],
                                                                            weight=genome['links'][innov_nr]['weight'])

    def getLambdaFunction(self,
                          shape=None):
        formula = self.output_node.getLambdaFunction()
        if formula == None:
            return None
        else:
            return "lambda _l_in,_l_out,{}:{}".format( ','.join(['c' + str(i) for i in range(len(shape))]),
                                          formula)
        

    """
    Get the output of the CPPN for a set of coordinates
    """
    def queryCPPN(self,
                  coordinate_1=None,
                  coordinate_2=None):
        try:
            assert len(coordinate_1) == len(coordinate_2)
            assert len(coordinate_1) + len(coordinate_2) == len(self.sensor_nodes)
            #flush the CPPN
            for _,node in self.cppn_nodes.items():
                node.flush()

            #insert input into sensor nodes
            for i in range(len(coordinate_1)):
                self.sensor_nodes[i*2  ].setActivity(coordinate_1[i])
                self.sensor_nodes[i*2+1].setActivity(coordinate_2[i])

            #return activity of CPPN's output node (determined dynamically)
            #if activity in range [-0.2,0.2] return 0 to indicate no link
            output_activity = self.output_node.getActivity()
            return output_activity if m.fabs(output_activity) > 0.2 else 0.
        except:
            log( self, traceback.format_exc())
            sys.exit(1)
        

    
        
