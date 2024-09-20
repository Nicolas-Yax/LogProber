#Imports
from lanlab.tools.dict_tools import SafeDict
from lanlab.studies.basic import BasicStudy
from lanlab.data_management.loader.sequence_loader.text_loader import TextLoader
from lanlab.models.openai_models import GPT35

class AnswerExtractionStudyConfig(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["nb_run_per_question"] = 1
        self.d["append"] = ''
        self.d["prepend"] = ''
        self.d["default_extraction_format"] = 'The answer written right before is:'

class AnswerExtractionStudy(BasicStudy):
    @property
    def config_class(self):
        return AnswerExtractionStudyConfig
    def _run(self):
        #Answer every question
        super()._run()
        #Get the extraction sequence from the question segments
        extraction_sequence = self.get_extraction_sequence()
        #Adds a new segment with the extraction sequence
        self.data += extraction_sequence
        #Extract the answer with chatGPT (lower cost)
        extraction_model = GPT35()
        extraction_model['temperature'] = 0
        extraction_model['max_tokens'] = 64
        super()._run(model=extraction_model)

    def get_extraction_sequence(self):
        """ Returns the extraction sequence from the question segments"""
        def f(x):
            """ Mapping function that returns the extraction sequence of a question segment"""
            if x[0]['info']['extraction_sequence'] is not None:
                text = TextLoader(x[0]['info']['extraction_sequence']).generate()
            else:
                text = TextLoader(self['default_extraction_format']).generate()
            return text
        return self.data.map(f)
        