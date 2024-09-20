import logging

from lanlab.tools.dict_tools import SafeDict
from lanlab.data_management.loader.sequence_loader.sequence_loader import SequenceLoader, SequenceLoaderConfig
from lanlab.data_management.data.segment import Segment
from lanlab.data_management.batch.sequence import Sequence

class QuestionSegment(Segment):
    def __init__(self):
        super().__init__()
        self.add_key('keywords',None)
        self.add_key('info',None)

class QuestionConfig(SequenceLoaderConfig):
    def __init__(self):
        super().__init__()
        self.d['question_format'] = '[prompt]\n'
        self.d['format'] = '[question]'

class QuestionLoader(SequenceLoader):
    def __init__(self,q=None):
        self.prompt = None
        self.keywords = None
        self.info = None
        #Set initial config
        super().__init__()
    @property
    def config_class(self):
        return QuestionConfig
    def from_dict(self,q):
        #Set the question propertie
        self.prompt = q['prompt']
        self.keywords = q['keywords']
        self.info = q['info']
    def generate(self):
        #Check if the format contains [prompt]
        if not('[prompt]' in self['question_format']):
            logging.warning("[prompt] isn't detected in your question formatting :"+self['format']+".\n This will result in the actual input not being put in the prompt sent to the model.")
        #if not('[question]' in self['format']):
        #    logging.warning("[question] isn't detected in your question formatting :"+self['format']+".\n This will result in the actual input not being put in the prompt sent to the model.")
        s = str(self['question_format']).replace('[prompt]',self.prompt)
        s = str(self['format']).replace('[question]',s)
        #Create the segment
        seg = QuestionSegment()
        seg['text'] = s
        seg['origin'] = "user"
        seg['keywords'] = self.keywords
        seg['info'] = self.info
        #Return the segment chain with the segment
        return Sequence(l=[seg])