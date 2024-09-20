from lanlab.data_management.loader.sequence_loader.sequence_loader import SequenceLoader
from lanlab.data_management.data.segment import Segment
from lanlab.data_management.batch.sequence import Sequence

class TextLoader(SequenceLoader):
    """ Generates a segment corresponding to a given text"""
    def __init__(self,t):
        self.t = t

    def generate(self):
        s = Segment()
        s['text'] = self.t
        s['origin'] = "user"
        return Sequence(l=[s])
    
class SystemTextLoader(SequenceLoader):
    """ Generates a segment corresponding to a given text"""
    def __init__(self,t):
        self.t = t

    def generate(self):
        s = Segment()
        s['text'] = self.t
        s['origin'] = "system"
        return Sequence(l=[s])