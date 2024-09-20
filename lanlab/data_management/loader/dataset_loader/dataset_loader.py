import pickle
import os

from lanlab.tools.configurable import Configurable
from lanlab.data_management.loader.sequence_loader.sequence_loader import SequenceLoader, SequenceLoaderConfig

class DatasetLoaderConfig(SequenceLoaderConfig):
    pass

class DatasetLoader(Configurable):
    def __init__(self,name=None):
        super().__init__()
        self._name = name
    @property
    def config_class(self):
        return DatasetLoaderConfig
    def __iter__(self):
        raise NotImplementedError
    def generate(self):
        raise NotImplementedError
    @property
    def name(self):
        if self._name is None:
            assert False #Need name to dataset
        return self._name