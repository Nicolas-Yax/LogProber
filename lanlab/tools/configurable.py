class Configurable:
    def __init__(self):
        self.config = self.config_class()
    @property
    def config_class(self):
        raise NotImplementedError
    def set_config(self):
        return self.config
    def __getitem__(self,k):
        if isinstance(k,str):
            return self.config[k]
    def __setitem__(self,k,v):
        if isinstance(k,str):
            self.config[k] = v
    def __str__(self):
        return str(self.__class__)+str(self.config)
    def __repr__(self):
        return str(self)