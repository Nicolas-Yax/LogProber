class Analysis:
    def __init__(self,study):
        self.study = study

        self.run(study)
        #self.save(study)
    def run(self,study):
        raise NotImplementedError
    def save(self):
        pass #TODO