from lanlab.studies.basic import BasicReadStudy

class LogScoreStudy(BasicReadStudy):
    def __init__(self,dataset,model,name=None,reconfigure_model=True,reconfigure_dataset=True):
        #Reconfigure the input_collector to be minimal
        if reconfigure_dataset:
            dataset['format'] = '[question]'
        #Reconfigure the model to be minimal
        if reconfigure_model:
            model['max_tokens'] = 1
        #Minimal OneModelStudy config
        super().__init__(dataset,model,name=name)
        self['append'] = ''
        self['prepend'] = ''
        self['nb_run_per_question'] = 1