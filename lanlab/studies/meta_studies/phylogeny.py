from lanlab.studies.basic import BasicReadStudy, BasicCompleteStudy
from lanlab.studies.meta_studies.meta_study import MetaStudy, MetaStudyConfig
from lanlab.data_management.loader.dataset_loader.dataset_loader import DatasetLoader, DatasetLoaderConfig
from lanlab.data_management.loader.sequence_loader.text_loader import TextLoader
from lanlab.data_management.batch.batch import BatchArray

class GenerateDatasetConfig(DatasetLoaderConfig):
    def __init__(self):
        super().__init__()
        self.add_key('prefixs',v=[''])

class GenerateDatasetLoader(DatasetLoader):
    """ Generate a dataset that is n copies of the prefix"""
    def __init__(self,prefixs=None,name='empty'):
        super().__init__()
        self._name = 'generate'
        if prefixs is None:
            prefixs = ['']
        self['prefixs'] = prefixs

        self.sequences = [TextLoader(prefix)for prefix in self['prefixs']]

    def __len__(self):
        return len(self.sequences)
    
    def generate(self,n=1):
        l = BatchArray(size=(len(self.sequences),n),name=self.name)
        for i,q in enumerate(self.sequences):
            for j in range(n):
                l[i,j] = q.generate()
        return l

    def __getitem__(self,k):
        if isinstance(k,str):
            return super().__getitem__(k)
        return self.sequences[k]

    def __iter__(self):
        return iter(self.sequences)
    
    @property
    def config_class(self):
        return GenerateDatasetConfig

class PhylogenyStudyConfig(MetaStudyConfig):
    def __init__(self):
        super().__init__()
        self.add_key('max_tokens',v=128)   
        self.add_key('n',v=40) 
        self.add_key('get_logits',v=False)


class PhylogenyStudy(MetaStudy):
    def __init__(self,models,prefixs=None,name='phylogeny',dataset_name='empty'):
        dataset = GenerateDatasetLoader(prefixs=prefixs,name=dataset_name)
        super().__init__(dataset,models)

    @property
    def config_class(self):
        return PhylogenyStudyConfig
    
    def _run(self):
        #Generate input data
        inp_studies = []
        for model in self.models:
            with model:
                generate_study = BasicCompleteStudy(self.dataset,model,name='generate')
                model['max_tokens'] = self['max_tokens']
                if self['get_logits']:
                    model['return_logits'] = True
                generate_study.run()
                inp_studies.append(generate_study)
        #Read the data
        for study_in,model_in in zip(inp_studies,self.models):
            with model_in:
                for model_out in self.models:
                    if model_in != model_out:
                        read_study = BasicReadStudy(study_in.data,model_out,name='read')
                        study_in.data.name = ':::'+model_in.name+'_'+study_in.data.name
                        read_study.run()
