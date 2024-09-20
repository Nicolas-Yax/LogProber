from lanlab.data_management.data.object import Object
from lanlab.tools.dict_tools import SafeDict

class Segment(Object):
    """ A part of a sequence that represent a text and the associated logprobabilities of the model's completion of the text."""
    def __init__(self,d=None):
        super().__init__()
        assert isinstance(d,SafeDict) or d is None or isinstance(d,dict), "The argument of a Segment must be a SafeDict or a dict or None"
        
        self.add_key('text','') #The text of the segment
        self.add_key('origin',None) #The origin of the segment (e.g. the assistant, the user, ...)
        self.add_key('tags',[]) #Additional tags of the segment (whether it's a question, ...)

        self.add_key('model',None) #The model that generated the segment
        self.add_key('tokens',[]) #The tokens of the model's completion of the text
        self.add_key('logp',[]) #The logprobabilities of the model's completion of the text
        self.add_key('top_logp',[]) #The top logprobabilities of the model's completion of the text
        self.add_key('finish_reason',None) #The reason why the model stopped generating tokens
        self.add_key('logits',[]) #The logits of the model's completion of the text

        #Load the dictionary
        if d is not None:
            for k in d:
                self[k] = d[k]
            
        self.serialize = self.to_dict

    def __str__(self):
        return self['origin'] + '(' + str(self['model']) + ')' + ': ' + self['text']

