import os
import logging
import json
import pickle
import yaml
import copy
import numpy as np

from lanlab.tools.dict_tools import SafeDict
from lanlab.tools.configurable import Configurable
from lanlab.data_management.loader.dataset_loader.dataset_loader import DatasetLoader

from lanlab.data_management.batch.batch import BatchArray

class UnnamedStudy(Exception):
    """ Raised when something tries to access the study name but the name is not defined """
    def __init__(self):
        self.message = "The study has no name. However it is mandatory to name is as it is required to initialize the files related to this study. You can name the study using the name=[your study name] parameter when creating the object or by heritage by overloading the @property name."

class StudyConfig(SafeDict):
    pass

class Study(Configurable):
    def __init__(self,data,model,name=None):
        data = copy.deepcopy(data) #To avoid modifying the original data
        if isinstance(data,DatasetLoader):
            self.dataset = data
            self.data = BatchArray(size=(0,),name=self.dataset.name) #place holder with the name of database
        else:
            self.dataset = None
            self.data = data
        self.model = model
        self._name = name
        super().__init__()
    @property
    def config_class(self):
        return StudyConfig
    def already_exists(self):
        return os.path.isdir(self.path) and len(os.listdir(self.path)) > 0
    def create_folders(self):
        try:
            os.makedirs(self.path)
        except FileExistsError:
            pass
    def run(self,update_objects=False,update_data=False):
        #Run the study with default parameters set to False to not erase data
        self.frun(update_objects=update_objects,update_data=update_data)
    def frun(self,update_objects=True,update_data=True):
        #Check if the study already exists
        logging.info('Checking if the study already exists.')
        if self.already_exists():
            logging.info("Found the study. Trying to load it and the data.")
            #Load the study
            self.load(load_objects=not(update_objects),load_data=not(update_data))
            #Update the objects if desired
            if update_objects:
                self.save_objects()
                logging.info('Updated objects.')
            else:
                #Try to load the objects
                try:
                    self.load_objects()
                #If the objects are not found, update them
                except FileNotFoundError:
                    logging.info("Didn't find the objects. Updating them.")
                    self.save_objects()
                    logging.info('Updated objects.')
            #Update the data if desired
            if update_data:
                self.__run()
                self.save_data()
            else:
                #Try to load the data
                try:
                    self.load_data()
                #If the data is not found, update it
                except FileNotFoundError:
                    logging.info("Didn't find the data. Updating it.")
                    self.__run()
                    self.save_data()
                    logging.info('Updated data.')
        else:
            logging.info("Didn't find the study. Running it.")
            self.create_folders()
            self.__run()
            self.save()
    def __run(self):
        #Run the study and generate the data from the dataset if necessary
        if self.dataset is not None:
            logging.info("Generating the data.")
            self.data = self.dataset.generate(n=self.config['nb_run_per_question'])
            logging.info("Data generated.")
        logging.info("Running the study.")
        self._run()
    @property
    def name(self):
        if self._name is None:
            assert UnnamedStudy() #Name of study not defined.
        return self._name
    @property
    def path(self):
        raise NotImplementedError
    
    def save_data(self):
        with open(os.path.join(self.path,'data.p'),'wb') as f:
            pickle.dump(self.data,f)
    
    def save_objects(self):
        with open(os.path.join(self.path,'study.p'),'wb') as f:
            d = self.data.pop_data()
            m = self.model
            del self.model
            pickle.dump(self,f)
            self.model = m
            self.data.push_data(d)

    def save(self):
        self.save_data()
        self.save_objects()

    def load_data(self):
        with open(os.path.join(self.path,'data.p'),'rb') as f:
            self.data = pickle.load(f)

    def load_objects(self):
        with open(os.path.join(self.path,'study.p'),'rb') as f:
            study_obj = pickle.load(f)
            model = self.model
            self.__dict__ = study_obj.__dict__ #Copy the attributes of the loaded object (do not do this at home)
            self.model = model
    def load(self,load_objects=False,load_data=False):
        #The order is important here.
        if load_objects:
            self.load_objects()
        if load_data:
            self.load_data()

    def _run(self):
        raise NotImplementedError
    