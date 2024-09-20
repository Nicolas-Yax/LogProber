from lanlab.studies.study import Study, StudyConfig

class MetaStudyConfig(StudyConfig):
    pass

class MetaStudy(Study):
    """ Meta study class working on several models"""
    def __init__(self,dataset,models):
        super().__init__(dataset,models)

    @property
    def models(self):
        return self.model

    @property
    def config_class(self):
        return MetaStudyConfig
    
    def _run(self):
        raise NotImplementedError