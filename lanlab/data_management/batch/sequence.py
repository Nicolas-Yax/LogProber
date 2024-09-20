from lanlab.data_management.batch.batch import BatchList
from lanlab.data_management.data.segment import Segment

from typing import Union,List

class Sequence(BatchList):
    """A sequence is a chain of segments (a batch of segment objects) that represent a dialog with a language model """
    def __init__(self,l : Union[List[Segment],None]=None):
        if l is None:
            l = []
        #Verify that all elements in the list are segments or loadable as Segments
        ll = []
        for k in l:
            if not(isinstance(k,Segment)):
                ll.append(Segment(k))
        self.l = l

        self.serialize = self.to_dict

    def to_dict(self):
        return [k.to_dict() for k in self]
    
    def __add__(self,s1 : Union[BatchList,Segment]):
        """ Returns a new sequence that is the concatenation of two sequences or a sequence and a segment"""
        s = Sequence(l=self.l)
        s += s1
        return s

    def __iadd__(self,s : Union[BatchList,Segment]):
        """ Adds a sequence or a segment to the current sequence"""
        if isinstance(s,Sequence):
            self.l += s.l
        elif isinstance(s,Segment):
            self.l.append(s)
        return self
    
    def to_str(self):
        """ Returns the concatenation of all segments' text using newlines as separators between segments"""
        s = ""
        for segment in self:
            s += segment.to_str() + "\n"
        return s
    
    def __str__(self):
        """ Returns the concatenation of all segments' text """
        s = ""
        for segment in self:
            s += segment['text']
        return s