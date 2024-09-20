from lanlab.tools.dict_tools import SafeDict
from lanlab.tools.configurable import Configurable

import logging
import pickle
import os

class ModelConfig(SafeDict):
    def __init__(self):
        super().__init__()
        self.add_key('temperature',0.7)
        self.add_key('max_tokens',16)
        self.add_key('top_p',1)
        self.add_key('stop',None)
        self.add_key('logit_bias',{})

class Model(Configurable):
    """ Language Model interface class """
    def __init__(self):
        super().__init__()
    def config_class(self):
        return ModelConfig()
    def complete(self,sequence,config):
        raise NotImplementedError
    @property
    def surname(self):
        """ Name of the model (the short version if possible)"""
        return NotImplementedError
    def save(self,study):
        with open(os.path.join(study.path,'model.p'),'wb') as f:
            pickle.dump(self,f)

    def __enter__(self):
        return self
    
    def __exit__(self,exc_type,exc_value,traceback):
        self.close()

    def start(self):
        pass

    def close(self):
        pass
