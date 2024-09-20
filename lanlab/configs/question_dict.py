from lanlab.tools.dict_tools import SafeDict

class QuestionDataDict(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["prompt"] = None
        self.d["options"] = None
        self.d["keywords"] = None
        self.d["info"] = None
        

class QuestionSetupDict(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["qa_prompt"] = None
        self.d["backslash"] = None

class MCQSetupDict(QuestionSetupDict):
    def __init__(self):
        super().__init__()
        self.d["answers_format"] = None
        self.d["extraction_format"] = None