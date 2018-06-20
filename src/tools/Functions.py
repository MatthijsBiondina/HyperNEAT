import numpy as np
import math as m

class Functions:
    f = {'none'    :None,
         'gaussian':(lambda x: 1 / m.sqrt(2*m.pi) * np.exp(-0.5*x)),
         'sigmoid' :(lambda x: 1 / ( 1 + np.exp(-x))),
         'sine'    :(lambda x: np.sin(x)),
         'absolute':(lambda x: np.absolute(x)),
         'relu'    :(lambda x: x.clip(min=0)),
         'linear'  :(lambda x: x)}

