import os
import os.path
import sys
import json
from genetics.CPPN import *
from genetics.Substrate import *
from tools.MyThread import *
import time
import numpy as np
from numpy.random import RandomState
import multiprocessing
from multiprocessing import Process,Pipe,Lock
from itertools import product

from genetics.Evolution import *

"""
    Worker Class:

    Used for simulating behaviour of genomes in environment, determining fitness and evolution

    model            [Model] := model assigns which individuals this worker must simulate, and communicates fitness of individuals evaluated by other workers
    worker_nr        [int] := number identifying worker
    environment      [openai gym environment] := toolset for simulating environment
    environment_name [str] := name of the environment, for loading and saving population genomes
    pop_size         [int] := size of the population
"""
class Worker(Process):
    GENOMES_BASE_FOLDER = '../res/genomes/' #base directory where pre-trained genomes are stored
    SHOW_BEST_INDIVIDUAL = True
    SHOW = True

    def __init__(self,
                 pipe=None,
                 lock=None,
                 seed=42,
                 worker_nr=None,
                 partition=None,
                 environment=None,
                 environment_name=None,
                 pop_size=0,
                 cppn_output_range=0,
                 input_shape=None,
                 hidden_shapes=None,
                 output_shape=None,
                 c1=1.0,
                 c2=1.0,
                 c3=0.4,
                 dt=3.0,
                 prob_add_link=0.1,
                 prob_add_node=0.03,
                 prob_mut_weights=0.8,
                 prob_mut_uniform=0.9,
                 prob_interspecies_mating=0.001,
                 elitism=0.2,
                 activation_function=None,
                 cppn_weight_range=None,
                 function_set=None
                 ):
        super().__init__()
        self.population = [] #current population of genomes
        self.fitness_scores = None
        self.random = RandomState(seed)
        
        self.pipe = pipe
        self.lock = lock
        self.worker_nr = worker_nr
        self.partition = partition
        self.pop_size = pop_size
        self.population = [None] * pop_size
        self.fitness_scores = np.full((pop_size,),None,dtype=float)
        self.cppn_output_range = cppn_output_range
        self.input_shape=input_shape
        self.hidden_shapes=hidden_shapes
        self.output_shape= (output_shape, )
        self.environment=environment
        if self.SHOW_BEST_INDIVIDUAL or self.SHOW:
            self.environment.render()
            time.sleep(5)
        self.environment_name=environment_name
        while len(self.output_shape) < len(self.input_shape):
            self.output_shape = (1,) + self.output_shape
        self.activation_function=activation_function
        self.loadGenomesFromFiles(self.GENOMES_BASE_FOLDER + environment_name)

        self.c1=c1
        self.c2=c2
        self.c3=c3
        self.dt=dt
        self.prob_add_link=prob_add_link
        self.prob_add_node=prob_add_node
        self.prob_mut_weights=prob_mut_weights
        self.prob_mut_uniform=prob_mut_uniform
        self.prob_interspecies_mating=prob_interspecies_mating
        self.elitism=elitism
        self.cppn_weight_range=cppn_weight_range
        self.function_set = function_set

        #structures
        self.layers = []
        self.weights = []
        self.coords = []

        self.buildStructures( self.input_shape, self.hidden_shapes, self.output_shape )

        self.evolution = Evolution( genomes = self.population,
                                    fitness = self.fitness_scores,
                                    random  = self.random,
                                    c1=self.c1,
                                    c2=self.c2,
                                    c3=self.c3,
                                    dt=self.dt,
                                    prob_add_link=self.prob_add_link,
                                    prob_add_node=self.prob_add_node,
                                    prob_mut_weights=self.prob_mut_weights,
                                    prob_mut_uniform=self.prob_mut_uniform,
                                    prob_interspecies_mating=self.prob_interspecies_mating,
                                    elitism=self.elitism,
                                    weight_range = self.cppn_weight_range,
                                    function_set = self.function_set)

    """
    What to do while running
    """
    def run(self):
        stop = False
        while not stop:
            self.fitness_scores = np.full((self.pop_size,),None,dtype=float)
            for genome_id in self.partition:
                self.fitness_scores[genome_id] = self.evaluate(genome_id)
            self.pipe.send( self.fitness_scores )         
            self.fitness_scores = self.pipe.recv()
            time.sleep(5) #prevent overheating or something?
            self.writeGenomesToFiles(self.GENOMES_BASE_FOLDER + self.environment_name)
            #calculate highscore over multiple trials. This does not count to towards training time
            self.pipe.send( self.calcHighscore() )
            stop = self.pipe.recv()
            
            self.population = self.evolution.evolve(fitness_scores=self.fitness_scores,genomes=self.population)
            
    def calcHighscore(self):
        genome_id = np.nanargmax( self.fitness_scores )
        #build phenotype
        cppn = CPPN(genome=self.population[genome_id],
                    cppn_output_range=self.cppn_output_range)
        substrate = Substrate( CPPN=cppn,
                               layers=self.layers,
                               coords=self.coords,
                               activation_function=self.activation_function,
                               cppn_output_range=self.cppn_output_range)
        highscores = []
        for _ in range( max(1, int( 1 / (len(self.population)/len(self.partition)) ) ) ):
            observation = self.environment.reset()
            score = 0.
            while True:
                activations = substrate.querySubstrate( input_matrix=observation )
                action = np.argmax(activations)
                observation, reward, done, info = self.environment.step(action)
                print(info)
                score += reward
                if self.SHOW_BEST_INDIVIDUAL:
                    self.environment.render()
                    time.sleep(1./60)
                if done:
                    print(info)
                    highscores.append(score)
                    break
        return np.array(highscores,dtype=float)
        

    def evaluate(self,genome_id):
        #build phenotype
        cppn = CPPN(genome=self.population[genome_id],
                    cppn_output_range=self.cppn_output_range)
        substrate = Substrate( CPPN=cppn,
                               layers=self.layers,
                               coords=self.coords,
                               activation_function=self.activation_function,
                               cppn_output_range=self.cppn_output_range)
        #play simulation
        observation = self.environment.reset()
        score = 0.
        while True:
            activations = substrate.querySubstrate( input_matrix=observation )
            action = np.argmax(activations)
            observation, reward, done, info = self.environment.step(action)
            score += reward
            if self.SHOW:
                self.environment.render()
                time.sleep(1./60)
            if done:
                return score
        return 0.

    def show(self):
        genome_id = np.nanargmax( self.fitness_scores )
        #build phenotype
        cppn = CPPN(genome=self.population[genome_id],
                    cppn_output_range=self.cppn_output_range)
        substrate = Substrate( CPPN=cppn,
                               layers=self.layers,
                               coords=self.coords,
                               activation_function=self.activation_function,
                               cppn_output_range=self.cppn_output_range)
        #play simulation
        observation = self.environment.reset()
        while True:
            self.environment.render()
            time.sleep(1./24)
            activations = substrate.querySubstrate( input_matrix=observation )
            action = np.argmax(activations)
            observation, _, done, _ = self.environment.step(action)
            if done:
                self.environment.render()
                return

    """
    Load (pre-trained) genomes from .genome files 
    """
    def loadGenomesFromFiles(self,path):
        self.lock.acquire()
        for genome_file in os.listdir(path):
            if genome_file.endswith(".genome"):
                with open( os.path.join(path,genome_file ) ) as f:
                    try:
                        genome = json.load(f)#.read()
                    except:
                        log(self,traceback.format_exc())
                #genome = lambdaJSON.deserialize(serialized)

                genome['links'] = {int(key):genome['links'][key] for key in genome['links']}
                
                self.population[ int(genome_file.split('.')[0]) ] = genome
        self.lock.release()

    def writeGenomesToFiles(self,path):
        self.lock.acquire()
        for genome_id in self.partition:
            with open(path + '/' + str(genome_id).zfill(3) + '.genome','w+') as jf:
                json.dump(self.population[genome_id],jf)
                #jf.write(lambdaJSON.serialize(self.population[genome_id]))
        self.lock.release()

    def stopCondition(self):
        return False

    def buildStructures(self,input_shape,hidden_shapes,output_shape):
        #layers
        self.layers.append( np.zeros( input_shape ) )
        for shape in hidden_shapes:
            self.layers.append( np.zeros(shape) )
        self.layers.append( np.zeros(output_shape) )
        
        l_coords = []

        #coordinates
        l_axis = np.arange(-1.,1.,2./len(self.layers))+1./len(self.layers)
        if len(input_shape) == 1: #if layers are 1-D
            for l_id in range(len(self.layers)-1):
                w_shape = self.layers[l_id+1].shape + self.layers[l_id].shape
                l_coords.append( np.array( list( product( (np.arange(-1.,1.,2./self.layers[l_id+1].shape[0])+1./self.layers[l_id+1].shape[0]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id+0].shape[0])+1./self.layers[l_id+0].shape[0]).tolist()))).reshape(w_shape + (2,)))
                
                l_coords[-1] = np.concatenate(( np.array([(l_axis[l_id],
                                                           l_axis[l_id+1])]*(l_coords[-1].shape[0]*
                                                                             l_coords[-1].shape[1])).reshape(l_coords[-1].shape[:-1] + (2,)),
                                              l_coords[-1]),
                                              axis=-1)
        elif len(input_shape) == 2: #if layers are 2-D
            for l_id in range(len(self.layers)-1):
                w_shape = self.layers[l_id+1].shape + self.layers[l_id].shape
                l_coords.append( np.array( list( product( (np.arange(-1.,1.,2./self.layers[l_id+1].shape[0])+1./self.layers[l_id+1].shape[0]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id+1].shape[1])+1./self.layers[l_id+1].shape[1]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id  ].shape[0])+1./self.layers[l_id  ].shape[0]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id  ].shape[1])+1./self.layers[l_id  ].shape[1]).tolist()))).reshape(w_shape + (4,)))
                l_coords[-1] = np.concatenate(( np.array([(l_axis[l_id],
                                                           l_axis[l_id+1])]*(l_coords[-1].shape[0]*
                                                                             l_coords[-1].shape[1]*
                                                                             l_coords[-1].shape[2]*
                                                                             l_coords[-1].shape[3])).reshape(l_coords[-1].shape[:-1] + (2,)),
                                              l_coords[-1]),
                                              axis=-1)
        elif len(input_shape) == 3:
            for l_id in range(len(self.layers)-1): #if layers are 3-D
                w_shape = self.layers[l_id+1].shape + self.layers[l_id].shape
                l_coords.append( np.array( list( product( (np.arange(-1.,1.,2./self.layers[l_id+1].shape[0])+1./self.layers[l_id+1].shape[0]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id+1].shape[1])+1./self.layers[l_id+1].shape[1]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id+1].shape[2])+1./self.layers[l_id+1].shape[2]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id  ].shape[0])+1./self.layers[l_id  ].shape[0]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id  ].shape[1])+1./self.layers[l_id  ].shape[1]).tolist(),
                                                          (np.arange(-1.,1.,2./self.layers[l_id  ].shape[2])+1./self.layers[l_id  ].shape[2]).tolist()))).reshape(w_shape + (4,)))
                l_coords[-1] = np.concatenate(( np.array([(l_axis[l_id],
                                                           l_axis[l_id+1])]*(l_coords[-1].shape[0]*
                                                                             l_coords[-1].shape[1]*
                                                                             l_coords[-1].shape[2]*
                                                                             l_coords[-1].shape[3]*
                                                                             l_coords[-1].shape[4]*
                                                                             l_coords[-1].shape[5])).reshape(l_coords[-1].shape[:-1] + (2,)),
                                              l_coords[-1]),
                                              axis=-1)
        self.coords = np.array(l_coords)

    
            
        
