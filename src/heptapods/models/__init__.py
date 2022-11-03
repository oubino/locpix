"""Init file

Bring all models into namespace

model_choice returns initialised model,
so that outside of this init file,
don't have to specify the location
of the model in models/ just the name!
"""

from .simple_gcn_1 import SimpleGCN1
from .simple_gcn_2 import SimpleGCN2


def model_choice(name, *args):
    if name == 'simplegcn1':
        return SimpleGCN1(*args)
    elif name == 'simplegcn2':
        return SimpleGCN2(*args)
    else:
        raise ValueError(f'{name} is not a supported model')
