from lanlab.studies.basic import BasicStudy
from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader
from lanlab.data_management.loader.dataset_loader.question_dataset_loader import QuestionDatasetLoader
from lanlab.studies.extraction import AnswerExtractionStudy


class NewCRTQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='newcrt')
        self.from_json('inputs/new_crt.json',QuestionLoader)

class OldCRTQuestionDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='oldcrt')
        self.from_json('inputs/old_crt.json',QuestionLoader)

class VanillaStudy(BasicStudy):
    def __init__(self,input_collector,model,qa=True,name=None,reconfigure_model=True,reconfigure_input_collector=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_input_collector and qa:
            input_collector['format'] = 'Q:[question]\nA:'
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 256
        #Minimal OneModelStudy config
        super().__init__(input_collector,model,name=name)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 100

#Diamond heritage better here ?
class VanillaStudyExtraction(AnswerExtractionStudy):
    def __init__(self,input_collector,model,qa=True,name=None,reconfigure_model=True,reconfigure_input_collector=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_input_collector and qa:
            input_collector['format'] = 'Q:[prompt]\nA:'
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 256
        #Minimal OneModelStudy config
        super().__init__(input_collector,model,name=name)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 100
        
class VanillaStudyAlpacaPrompt(VanillaStudy):
    def __init__(self,input_collector,model,qa=True,name=None,reconfigure_model=True,reconfigure_input_collector=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_input_collector and qa:
            input_collector['format'] = (
                'Below is an instruction that describes a task, paired with an input that provides further context.'
                'Write a response that appropriately completes the request.\n\n'
                '### Instruction:\n[question]\n\n### Response:'
            )
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 256
        #Minimal OneModelStudy config
        super().__init__(input_collector,model,name=name,reconfigure_model=False,reconfigure_input_collector=False)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 100
        
class VanillaStudyAlpacaPrompt2(VanillaStudy):
    def __init__(self,input_collector,model,qa=True,name=None,reconfigure_model=True,reconfigure_input_collector=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_input_collector and qa:
            input_collector['format'] = (
                'You will be given an instruction.'
                'Please answer the request.\n\n'
                'Request:\n[question]\n\n Your answer:'
            )
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 256
        #Minimal OneModelStudy config
        super().__init__(input_collector,model,name=name,reconfigure_model=False,reconfigure_input_collector=False)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 100