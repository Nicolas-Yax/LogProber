import numpy as np
from typing import Any,Union,List

class BatchList():
    """List of objects with some useful methods."""
    def __init__(self,l: Union[List[Any],None]=None,name : str=None):
        if l is None:
            l = []
        self.l = l

        if name is None:
            name = "Data"
        self.name = name

        self.serialize = self.to_dict

    def __getitem__(self, i : int):
        return self.l[i]
    
    def __setitem__(self, i : int, value : Any):
        self.l[i] = value
    
    def __len__(self):
        return len(self.l)
    
    def __iter__(self):
        return iter(self.l)
    
    def __add__(self,other : Union[List[Any],Any]):
        bl = BatchList(self.l)
        bl += other
        return bl
    
    def __iadd__(self,other : Union[List[Any],Any]):
        if isinstance(other,BatchList):
            self.l += other.l
        else:
            self.l.append(other)
        return self
    
    def map(self,f):
        return BatchList([f(e) for e in self.l])
    
    def filter(self,f):
        return BatchList([e for e in self.l if f(e)])
    
    def to_dict(self):
        return {'name':self.name,'data':[k.serialize() for k in self.l]}
    
    def from_dict(self,d : dict):
        self.l = [k for k in d['data']]
        self.name = d['name']

    def pop_data(self):
        l = self.l
        self.l = []
        return l

    def push_data(self,data : List[Any]):
        self.l = data
    
    
class BatchArray():
    """ 2D Array of objects with some useful methods. """
    def __init__(self,a=None,size=None,name=None):
        #If a is None, create an empty array of the specified size
        if a is None:
            if size is None:
                raise Exception("BatchArray: size must be specified if a is None")
            a = np.empty(size,dtype=object)
        self.a = a
        self.size = size

        #Set the name of the array
        if name is None:
            name = "data"
        self.name = name

        self.serialize = self.to_dict

    @property
    def shape(self):
        return self.a.shape

    def __getitem__(self, i):
        return self.a[i]
    
    def __setitem__(self, i, value):
        self.a[i] = value
    
    def __len__(self):
        return len(self.a)
    
    def __add__(self,other):
        ba = BatchArray(self.a)
        ba += other.a
        return ba
    
    def __iadd__(self,other):
        from lanlab.data_management.batch.sequence import Sequence #To avoid circular imports
        if isinstance(other,BatchArray):
            self.a += other.a
        elif isinstance(other,Sequence):
            for l in self.a:
                for s in l:
                    s += other
        else:
            raise TypeError #Unknown type to be added to BatchArray
        return self
    
    def map(self,f):
        array = np.empty(self.size,dtype=object)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                array[i,j] = f(self.a[i,j])
        out = BatchArray(array)
        return out
    
    def to_dict(self):
        #Return a dictionary with the 2D data in json format
        return {'name':self.name,'data':[[k.serialize() for k in row] for row in self.a]}
    
    def from_dict(self,d):
        #Load a dictionary with the 2D data in json format
        from lanlab.data_management.batch.sequence import Sequence #To avoid circular imports
        self.a = np.array([[Sequence(l=k) for k in row] for row in d['data']])
        self.size = self.a.shape
        self.name = d['name']

    def pop_data(self):
        a = self.a
        self.a = np.empty(self.size,dtype=object)
        return a

    def push_data(self,data):
        self.a = data