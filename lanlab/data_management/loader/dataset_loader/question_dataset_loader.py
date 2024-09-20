import logging
import json
import os
import numpy as np

#Import QuestionLoader and Dataset
from lanlab.data_management.loader.sequence_loader.question_loader import QuestionLoader
from lanlab.data_management.loader.dataset_loader.dataset_loader import DatasetLoader
from lanlab.data_management.batch.batch import BatchArray
from lanlab.data_management.batch.sequence import Sequence

class QuestionDatasetLoader(DatasetLoader):
    """ Dataset class used for Studies. It can be composed of different type of data and each object in the data list should be an instance of Question."""
    def __init__(self,d=None,name=None):
        if d is None:
            self.questions = [] #Empty dataset
        else:
            self.from_list(d) #Init dataset

        self._iter_index = 0

        super().__init__(name=name)

    
    def generate(self,n=1):
        l = BatchArray(size=(len(self.questions),n),name=self.name)
        for i,q in enumerate(self.questions):
            for j in range(n):
                l[i,j] = q.generate()
        return l

    def _verify_question(self,q):
        assert isinstance(q,QuestionLoader) #The elements in the dataset need to be subclasses of QUestion
        assert self.config == q.config #All elements in the dataset don't have the same configuration. In a QuestionDataset all should have the same. 
        

    def append_question(self,q):
        self._verify_question(q)
        self.questions.append(q)

    def from_list(self,d):
        if self.questions != []:
            logging.warning('Loading a dataset on top of a non-empty dataset')
        #Verifications
        assert isinstance(d,list) #The data to be loaded should be a dictionary
        self.config = d[0].config.copy()
        for q in d:
            self.append_question(q)

    def from_json(self,file,questions_class,update_name=True):
        #Load the json file
        json_questions = json.load(open(file,'r'))
        #Build the question list
        question_list = []
        for q in json_questions:
            questions_object = questions_class()
            questions_object.from_dict(q)
            question_list.append(questions_object)
        #Load the question list in the actual dataset
        self.from_list(question_list)
        #Take the same name of the json file if asked
        if update_name:
            self._name = os.path.basename(file).split('.')[0]
        
    def __setitem__(self,k,v):
        #Set a question
        if isinstance(k,int):
            self._verify_question(v)
            self.questions[k] = v
        #Set a config parameter
        elif isinstance(k,str):
            #Apply the setting to every questions in the dataset
            for i in self.questions:
                i[k] = v
            #Apply setting to self
            super().__setitem__(k,v)

    def __getitem__(self,k):
        #Make it indexable by integers to access elements in addition to using str to access the config
        if isinstance(k,int):
            return self.questions[k]
        else:
            return super().__getitem__(k)

    def __iter__(self):
        return iter(self.questions)
    def __len__(self):
        return len(self.questions)
    