from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader
from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader,QuestionConfig
from lanlab.data_management.data.segment import Segment
from lanlab.data_management.batch.sequence import Sequence
import logging

class MCQSegment(Segment):
    def __init__(self):
        super().__init__()
        self.add_key('keywords',None)
        self.add_key('info',None)
        self.add_key('answers',None)

class MCQConfig(QuestionConfig):
    def __init__(self):
        super().__init__()
        self.d['question_format'] = '[prompt]\n'
        self.d['answer_format'] = '([letter]) [answer]\n'
        self.d['format'] = '[question]'

class MCQLoader(QuestionLoader):
    def __init__(self,q=None):
        self.prompt = None
        self.keywords = None
        self.info = None
        self.answers = None
        #Set initial config
        super().__init__()
    @property
    def config_class(self):
        return MCQConfig
    def from_dict(self,q):
        #Set the question propertie
        self.prompt = q['prompt']
        self.answers = q['answers']
        self.keywords = q['keywords']
        self.info = q['info']
    def generate(self):
        #Check if the format contains [prompt]
        if not('[prompt]' in self['question_format']):
            logging.warning("[prompt] isn't detected in your question formatting :"+self['format']+".\n This will result in the actual input not being put in the prompt sent to the model.")
        if not('[question]' in self['format']):
            logging.warning("[question] isn't detected in your question formatting :"+self['format']+".\n This will result in the actual input not being put in the prompt sent to the model.")
        s = str(self['question_format']).replace('[prompt]',self.prompt)
        for i in range(len(self.answers)):
            s_answer = str(self['answer_format'])
            s_answer = s_answer.replace('[letter]',chr(97+i))
            s_answer = s_answer.replace('[number]',str(i+1))
            s_answer = s_answer.replace('[answer]',self.answers[i])
            s += s_answer
        s = str(self['format']).replace('[question]',s)
        #Create the segment
        seg = MCQSegment()
        seg['text'] = s
        seg['origin'] = "user"
        seg['keywords'] = self.keywords
        seg['info'] = self.info
        seg['answers'] = self.answers
        #Return the segment chain with the segment
        return Sequence(l=[seg])