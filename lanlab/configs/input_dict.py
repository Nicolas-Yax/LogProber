from lanlab.tools.dict_tools import SafeDict

class InputDict(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["prompt"] = None
        self.d["temperature"] = None
        self.d["max_tokens"] = None
        self.d["stop"]  = None
        self.d["top_p"] = None
        self.d["top_k"] = None
        self.d["logit_bias"] = None