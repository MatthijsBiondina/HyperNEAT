import gym
import gym.spaces
import math
import sys
import numpy

from tools.tools     import log
from mvc.model.Model import *
from mvc.model.Environment import *

class Main:
    def __init__(self):
        environment_name = "MountainCar-v0"
        environment      = self.makeEnvironment(environment_name=environment_name)
        print(type(environment))
        self.model = Model( seed=42,
                            environment=environment,
                            environment_name=environment_name,
                            max_frames=300,
                            nr_of_workers=1,
                            pop_size=300,
                            hidden_shape=((6,),),
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
                            function_set=[ '0.3989422804*np.exp(np.array(-0.5*({})).clip(min=-100,max=100))', #gaussian
                                           '1/(1+np.exp(np.array({}).clip(min=-100,max=100)))',            #sigmoid
                                           'np.sin({})',                     #sine
                                           'np.absolute({})'                      #absolute value
                                           ],
                            activation_function=(lambda x: x.clip(min=0))      #RELU
                            )

    def makeEnvironment(self,
                        environment_name=None
                        ):
        try:
            return Environment2D(environment_name)
            #return gym.make(environment_name)
        except Exception:
            log(self,traceback.format_exc())
            sys.exit(1)

    def getModel(self):
        return self.model

if __name__ == '__main__':
    m = Main()
    m.getModel().train()
