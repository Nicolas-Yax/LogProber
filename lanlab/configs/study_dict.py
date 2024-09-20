from lanlab.tools.dict_tools import SafeDict

class OneModelStudyDataDict(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["prompt"] = None
        self.d["options"] = None
        self.d["keywords"] = None
        self.d["info"] = None
        