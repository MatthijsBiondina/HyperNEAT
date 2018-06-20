import traceback
import sys
import gym
import gym.spaces
import copy
import os
import json
import time
import multiprocessing
import random
import numpy as np

from multiprocessing import Process, Pipe, Lock


from tools.tools import log
from mvc.model.Worker import *

class Model:

    

    """
    Model:

    environment       [env]   := openai gym environment
    environment_name  [str]   := name of the openai environment
    max_frames        [int]   := max number of frames per trial
    pop_size          [int]   := size of the population
    hidden_shape      [((int,int,int),...,(int,int,int))] := shape of the hidden layers in the substrate
    c1                [float] := hyper-parameter for speciation governing influence of excess genes
    c2                [float] := hyper-parameter for speciation governing influence of disjoint genes
    c3                [float] := hyper-parameter for speciation governing influence of weight differences
    dt                [float] := hyper-parameter delta_t for speciation
    prob_add_link     [float] := probability of adding link to CPPN
    prob_add_node     [float] := probability of adding node to CPPN
    prob_mut_weights  [float] := probability of mutating weights of CPPN
    prob_mut_uniform  [float] := probability of perturbing weights uniformly, o/w assign a new random value
    elitism           [float] := percentage of offspring resulting from mutation without crossover in each generation
    cppn_output_range [float] := output of CPPN is capped off to fall within [-float,float]
    cppn_weight_range [float] := weights in CPPN are capped off to fall within [-float,float]
    function_set      [list(lambda)] := list of functions used in nodes of the CPPN
    """
    def __init__(self,
                 seed=42,
                 environment=None,
                 environment_name=None,
                 max_frames=300,
                 nr_of_workers=1,
                 pop_size=300,
                 hidden_shape=((32,1,1),(16,1,1),(8,1,1)),
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
                 cppn_output_range=1.0,
                 cppn_weight_range=3.0,
                 function_set=None,
                 activation_function=(lambda x: ( 1 / (1 + m.exp(-x)))), #sigmoid
                 max_training_time=24*60*60
                 ):
        """Game Logic"""
        self.environment = None #openai gym environment
        self.fitness_scores = None

        """Multiprocessing"""
        self.lock = None
        
        self.seed = seed
        self.pop_size = pop_size
        self.fitness_scores = np.full((pop_size,),None,dtype=float)
        self.environment = environment
        print("Input shape is:",self.environment.reset().shape)
        self.makeGenomeFolderIfNotExists(environment_name)
        
        self.nr_of_workers=nr_of_workers
        self.environment_name = environment_name
        self.max_frames = max_frames
        
        self.hidden_shape = hidden_shape
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.dt = dt
        self.prob_add_link = prob_add_link
        self.prob_add_node = prob_add_node
        self.prob_mut_weights   = prob_mut_weights
        self.prob_mut_uniform   = prob_mut_uniform
        self.prob_interspecies_mating = prob_interspecies_mating
        self.elitism            = elitism
        self.cppn_output_range  = cppn_output_range
        self.cppn_weight_range  = cppn_weight_range
        self.function_set       = function_set
        self.activation_function = activation_function
        self.max_training_time = max_training_time
        

    """
    Train the population to optimize fitness
    """
    def train(self):
        training_time = 0.
        pipes = self.startup()
        epoch = 0
        while not self.stopCondition(training_time):
            epoch_start = time.time()
            for pipe in pipes:
                self.processRecievedFitnessScores( pipe.recv() )
            highscore = np.max(self.fitness_scores)
            meanscore = np.mean(self.fitness_scores)
            self.lock.acquire()
            for pipe in pipes:
                pipe.send( self.fitness_scores )
            self.lock.release()
            training_time += time.time() - epoch_start

            #determine highscore averaged over multiple trials
            highscores = []
            for pipe in pipes:
                for score in pipe.recv():
                    highscores.append( float(score) )
            with open('../res/highscores/' + self.environment_name + '.txt','a+') as f:
                write_string = ' '.join( ("Epoch",str(epoch),"completed in", "%.2f"%(training_time), "s with highscore", str(np.mean(np.array(highscores)))))
                f.write(write_string + '\n')
                print(write_string)
            self.lock.acquire()
            stop_training = ( training_time >= self.max_training_time )
            for pipe in pipes:
                pipe.send( stop_training )
            self.lock.release()
            epoch += 1
        return

    def test(self):
        return

    """
    Setup workers and communication pipelines from and to workers
    """
    def startup(self):
        #Make a lock that prevents multiple processes from editing the same variable
        self.lock  = Lock()
        workers    = []
        pipes      = []
        #assign partitions of individuals to workers
        partitions = self.makePartitions(pop_size=self.pop_size,nr_of_workers=self.nr_of_workers)
        
        for worker_nr in range(self.nr_of_workers):
            #set up pipe for communcation
            parent_conn, child_conn = Pipe()
            pipes.append(parent_conn)

            #set up a new worker
            worker = Worker(pipe=child_conn,
                            lock=self.lock,
                            seed=self.seed,
                            worker_nr=worker_nr,
                            partition=partitions[worker_nr],
                            environment=copy.deepcopy(self.environment),
                            environment_name=self.environment_name,
                            pop_size=self.pop_size,
                            cppn_output_range=self.cppn_output_range,
                            input_shape=self.environment.reset().shape,
                            hidden_shapes=self.hidden_shape,
                            output_shape=self.environment.action_space.n,
                            c1=self.c1,
                            c2=self.c2,
                            c3=self.c3,
                            dt=self.dt,
                            prob_add_link=self.prob_add_link,
                            prob_add_node=self.prob_add_node,
                            prob_mut_weights=self.prob_mut_weights,
                            prob_mut_uniform=self.prob_mut_uniform,
                            prob_interspecies_mating=self.prob_interspecies_mating,
                            elitism=0.2,
                            activation_function=self.activation_function,
                            cppn_weight_range=self.cppn_weight_range,
                            function_set=self.function_set)
            workers.append(worker)
            worker.start()
        return pipes

    """
    After each worker finishes their epoch, they send their list of fitness scores to the
    Model. For each worker, the Model recieves a np-array containing nan's for individuals
    that were not evaluated by that working, and floats for individuals that were evaluated
    representing the fitness score of that individual

    The list of fitness scores is locked to make sure that no racing accidents happen, and
    the evaluated fitness scores are added to the comprehensive list.
    """
    def processRecievedFitnessScores(self,recieved_scores=None):
        self.lock.acquire()
        mask = ~np.isnan(recieved_scores)
        self.fitness_scores[mask] = recieved_scores[mask]
        self.lock.release()

    """
    Each worker will be assigned a partition of the population; a list of indices that it
    has to train each epoch
    """
    def makePartitions(self,
                       pop_size=300,
                       nr_of_workers=1):
        indices = np.arange( pop_size ).tolist()
        #shuffle indices, because index and fitness is not guaranteed to be independent
        indices = random.sample(indices, len(indices))
        part_size = len(indices) / float(nr_of_workers)
        return [ indices[int(round(part_size*i)):int(round(part_size*(i+1)))] for i in range(nr_of_workers) ]

    """
    Check whether a folder with pretrained genomes exists for this environment.
    If not, make a new folder and initialize the population with empty genomes"""
    def makeGenomeFolderIfNotExists(self, environment_name):
        path = '../res/genomes/' + environment_name
        if not os.path.isdir(path): #check whether the folder exists
            os.makedirs(path)       #if not, make it
            observation = self.environment.reset()

            
            for idx in range(self.pop_size):
                genome = { "nodes":[],
                           "links":{}}
                #make one input node for each axis of both coordinates
                for node_id in range((len(observation.shape)+1)*2):
                    genome['nodes'].append( {'id':node_id,
                                             'func':'none',
                                             'type':'Sensor'
                                             }
                                            )
                #make a single output node
                genome['nodes'].append( {'id':len(genome['nodes']),
                                         'func':'none',
                                         'type':'Output'
                                         }
                                        )
                with open(path + '/' + str(idx).zfill(3) + '.genome','w+') as jf:
                    json.dump(genome,jf)
                    #jf.write(lambdaJSON.serialize(genome))
            
    def stopCondition(self,training_time):
        return training_time >= self.max_training_time
