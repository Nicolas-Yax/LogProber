from lanlab.studies.basic import BasicCompleteStudy
from lanlab.data_management.loader.dataset_loader.dataset_loader import DatasetLoader, DatasetLoaderConfig
from lanlab.data_management.loader.sequence_loader.text_loader import TextLoader
from lanlab.data_management.batch.batch import BatchArray
from lanlab.studies.basic import BasicStudyConfig
from lanlab.models.hf_models import HFModel

import numpy as np


    
class GenerateStudyConfig(BasicStudyConfig):
    def __init__(self):
        super().__init__()
        self.add_key('max_tokens',v=128)

class GenerateStudy(BasicCompleteStudy):
    def __init__(self,model,prefixs=None,name='generate',reconfigure_model=True):
        dataset = GenerateDatasetLoader(prefixs=prefixs)
        super().__init__(dataset,model,name=name)
        if reconfigure_model:
            model['max_tokens'] = self['max_tokens']
            if isinstance(model,HFModel):
                model['return_logits'] = True
        
    @property
    def config_class(self):
        return GenerateStudyConfig