from lanlab.tools.configurable import Configurable
from lanlab.tools.dict_tools import SafeDict

class SequenceLoaderConfig(SafeDict):
    pass

class SequenceLoader(Configurable):
    """ Base class for all input generators """
    def config_class(self):
        return SequenceLoaderConfig()
    def generate(self):
        raise NotImplementedError