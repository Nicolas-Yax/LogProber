from lanlab.studies.basic import BasicStudy
from lanlab.data_management.loader.sequence_loader.mcq_loader import MCQLoader
from lanlab.data_management.loader.dataset_loader.question_dataset_loader import QuestionDatasetLoader
from lanlab.studies.extraction import AnswerExtractionStudy


class CFMCQDataset(QuestionDatasetLoader):
    def __init__(self):
        super().__init__(d=None,name='cf')
        self.from_json('inputs/cf.json',MCQLoader)


class VanillaMCQStudy(BasicStudy):
    def __init__(self,input_collector,model,qa=True,name=None,reconfigure_model=True,reconfigure_input_collector=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_input_collector and qa:
            input_collector['format'] = 'Q:[question]\nA:'
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 1
        #Minimal OneModelStudy config
        super().__init__(input_collector,model,name=name)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 1