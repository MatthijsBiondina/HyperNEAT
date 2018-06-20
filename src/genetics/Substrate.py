import numpy as np

from genetics.CPPN import *
from tools.tools import log

import matplotlib.pyplot as plt

"""
    Substrate Class:

    An ANN derived from a CPPN

    CPPN                [CPPN] := a CPPN describing the weights between network nodes
    input_shape         [tuple] := tuple of integers describing the shape of the input layer
    hidden_shapes       [list(tuple)] := list of tuples of integers describing the shapes of the hidden layers
    output_shape        [int] := integer describing the number of nodes in the output layer. Due to the way openai gym works, the output layer is always 1 dimensional
    activation_function [lambda] := lambda function describing the activation function over nodes
"""
class Substrate:
    

    def __init__(self,
                 CPPN=None,
                 input_shape=None,
                 hidden_shapes=None,
                 output_shape=None,
                 layers=None,
                 weights=None,
                 coords=None,
                 activation_function=None,
                 cppn_output_range=None):
        self.charset = 'abcdefghijklmnopqrstuvwxyz' #set of characters to use in the subscript of np.einsum(...)
        self.multiplication_function = None #multiply layer with corresponding weight
        self.CPPN=CPPN
        self.cppn_output_range=cppn_output_range
        #self.layers  = [] # NN layers containing neuron activations
        self.weights = [] # link-weights between layers

        self.activation_function = activation_function
        self.layers  = layers
        self.coords  = coords

        

        #self.buildLayers(input_shape,hidden_shapes,output_shape)
        self.buildWeights()
        self.buildMultFunc()

    """
    Pass an input matrix to the ANN, and perform a feed-forward loop
    """
    def querySubstrate(self,
                       input_matrix=None):
        assert self.layers[0].shape == input_matrix.shape
        self.layers[0] = input_matrix
        for l in range(len(self.layers)-1):
            self.layers[l+1] = self.activation_function( self.multiplication_function( self.layers[l], self.weights[l] ) )
        return self.layers[-1]

    """
    Construct node layers of the ANN based on specified shapes
    """
    def buildLayers(self,input_shape,hidden_shapes,output_shape):
        self.layers.append( np.zeros( input_shape ) )
        for shape in hidden_shapes:
            self.layers.append(np.zeros( shape ) )
        self.layers.append( np.zeros(output_shape) )

        

    """
    Construct weights between the node layers.
    This is done with the recursive function fillWeightMatrix(...), so the algorithm is adaptable for varying dimensions
    """
    def buildWeights(self):
        w_shape = self.layers[1].shape + self.layers[0].shape
        func_str = self.CPPN.getLambdaFunction(shape=w_shape)
        no_links = (func_str == None)
        if not no_links:
            cppn_func = eval(func_str)
            

        try:
            for l in range(len(self.layers)-1):
                w_shape = self.layers[l+1].shape + self.layers[l].shape
                if( func_str == None ):
                    self.weights.append( np.zeros( w_shape ))
                else:
                    if len(self.layers[l].shape) == 1:
                        self.weights.append( cppn_func(self.coords[l][...,0],
                                                       self.coords[l][...,1],
                                                       self.coords[l][...,2],
                                                       self.coords[l][...,3]) )
                    elif len(self.layers[l].shape) == 2:
                        self.weights.append( cppn_func(self.coords[l][...,0],
                                                       self.coords[l][...,1],
                                                       self.coords[l][...,2],
                                                       self.coords[l][...,3],
                                                       self.coords[l][...,4],
                                                       self.coords[l][...,5]) )
                    elif len(self.layers[l].shape) == 3:
                        self.weights.append( cppn_func(self.coords[l][...,0],
                                                       self.coords[l][...,1],
                                                       self.coords[l][...,2],
                                                       self.coords[l][...,3],
                                                       self.coords[l][...,4],
                                                       self.coords[l][...,5],
                                                       self.coords[l][...,6],
                                                       self.coords[l][...,7]) )
                    if isinstance(self.weights[-1],np.float64):
                        self.weights[-1] = np.ones(w_shape)*self.weights[-1]
                    self.weights[-1][np.logical_and( self.weights[-1]>-0.2, self.weights[-1]<0.2 )] = 0.
        except:
            log( self, self.coords[0].shape )
            log( self, func_str )
            log( self, traceback.format_exc())

        plt.imshow( self.weights[1], cmap='inferno')
        plt.colorbar()
        plt.show()
                                                     
                    
            
        #for l in range(len(self.layers)-1):
        #    w_shape = self.layers[l+1].shape + self.layers[l].shape
        #    self.buildSmart( shape=w_shape, layer=l )
            
            #self.weights.append( np.zeros(w_shape) )
            #self.fillWeightMatrix( layer=l, matrix=self.weights[-1], idx=() )
            #self.weights[-1] = self.buildSmart( layer=l, shape=self.weights[-1].shape)


    def buildSmart(self,shape=None,layer=None):
        l = float(layer)
        try:
            func_str = self.CPPN.getLambdaFunction(shape=shape)
            if( func_str == None ):
                self.weights.append( np.zeros( shape ) )
                return
            else:
                func_str = func_str.replace('_l_in',str(l))
                func_str = func_str.replace('_l_out',str(l+1))
                cppn_func = eval( func_str )
                self.weights.append( np.clip(np.fromfunction( cppn_func, shape, dtype=float ),-self.cppn_output_range,self.cppn_output_range ))
                if isinstance(self.weights[-1],np.float64):
                    self.weights[-1] = np.ones(shape)*self.weights[-1]
                self.weights[-1][np.logical_and( self.weights[-1]>-0.2, self.weights[-1]<0.2 )] = 0.
                #self.weights[-1][self.weights[-1] > -0.2 and self.weights[-1] < 0.2] = 0.
        except:
            log( self, func_str )
            log( self, traceback.format_exc())

    """
    Determine weights between network layers using the CPPN. This function builds the weights recursively, so that it is
    applicable to layers of any number of dimensions, as long as the number of dimensions remain constant within the network
    """
    def fillWeightMatrix(self,
                         layer=None,
                         matrix=None,
                         idx=None):
        if len(idx) < len( matrix.shape ):#nested for loop with varying number of dimensions
            for next_dim_index in range( matrix.shape[ len(idx) ] ): 
                self.fillWeightMatrix( layer=layer, matrix=matrix, idx=( idx + ( next_dim_index, ) ) )
        else:
            #convert the index-positions of nodes to coordinate-positions, so that the origin is roughly in the middle of the matrix
            coordinates = tuple( x[0]-(x[1]-1)/2 for x in zip( idx, matrix.shape ) )
            #query the CPPN to get the weight of the link between nodes
            matrix[idx] = self.CPPN.queryCPPN( coordinate_1 = ((layer,)  +coordinates[int(len(coordinates)/2):]),
                                               coordinate_2 = ((layer+1,)+coordinates[:int(len(coordinates)/2)]))

    """
    This method describes the function used to multiply a node matrix with its weight matrix to determine output for the next layer.
    Since the shape of the layers can vary depending on the environment, the multiplication function is adapted to account for the dimensionality
    of the layers in the given environment.

    For example:
    layer n is of arbitrary shape, say ijk
    layer n+1 is of arbitrary shape, but has the same number of dimensions as layer n, say abc
    then the weight matrix has double the number of dimensions as layer n, namely: abcijk
    > to determine the activation in layer n+1:
    > perform: np.einsum('ijk,abcijk->abc', layer_n, weights_[n->n+1] )
    """
    def buildMultFunc(self):
        args2 = self.charset[:len(self.weights[0].shape) ]
        args1 = args2[int(len(args2)/2):]
        args3 = args2[:int(len(args2)/2)]
        subscripts = args1 + ',' + args2 + '->' + args3
        self.multiplication_function = lambda nodes,weights: np.einsum( subscripts, nodes, weights )
