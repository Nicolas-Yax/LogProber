import os

from lanlab.studies.study import Study
from lanlab.tools.dict_tools import SafeDict,to_dict
from threading import Thread
from queue import Queue,Empty
import time
import numpy as np
import logging

class BasicStudyConfig(SafeDict):
    def __init__(self):
        super().__init__()
        self.d["nb_run_per_question"] = 1
        self.d["append"] = ''
        self.d["prepend"] = ''
        self.d['store_logprobs'] = True

#Launch the completions with a thread system to compute them in parallel
BATCH_SIZE = 1 #number of threads

#Gets the input queue and shares it with all the workers so that they can pop the inputs until the queue is empty

class BasicCompleteWorker(Thread):
    def __init__(self,i,inp_queue,out_queue):
        super().__init__()
        self.inp_queue = inp_queue
        self.out_queue = out_queue
        self.i = i
        self.stage = 0

        self.alive = True

    def run(self):
        while self.alive:
            try:
                seq,model,i,j = self.inp_queue.get(timeout=1)
                new_seq = model.complete(seq)
                self.out_queue.put((new_seq,i,j))
            except Empty:
                time.sleep(1)

    def terminate(self):
        self.alive = False

class BasicReadWorker(BasicCompleteWorker):
    def run(self):
        while self.alive:
            try:
                seq,model,i,j = self.inp_queue.get(timeout=1)
                new_seq = model.read(seq)
                self.out_queue.put((new_seq,i,j))
            except Empty:
                time.sleep(1)


#Let's correct it by using a manager

class BasicCompleteStudy(Study):
    """ Basic study class operating on a single model and a 2D array of sequences"""

    @property
    def config_class(self):
        return BasicStudyConfig
    
    @property
    def worker_class(self):
        return BasicCompleteWorker
    
    def fill_input_queue(self,inp_queue,model):
        """ Fill the input queue with the sequences to complete """
        for i,row in enumerate(self.data.a):
            for j,seq in enumerate(row):
                inp_queue.put((seq,model,i,j))
    
    def init_workers(self):
        """ Initialize the workers """
        self.inp_queue = Queue()
        self.out_queue = Queue()
        self.workers = [self.worker_class(i,self.inp_queue,self.out_queue) for i in range(BATCH_SIZE)]
    
    def run_workers(self):
        """ Start the workers """
        for w in self.workers:
            w.start()

    def wait_for_results(self):
        """ Loop until all the results are in the data """
        got_data = np.zeros(self.data.a.shape,dtype=bool)
        nb_tot = np.prod(self.data.a.shape)
        while not(np.all(got_data)):
            seq,i,j = self.out_queue.get()
            self.data.a[i][j] = seq
            got_data[i,j] = True
            print("{}%".format(np.sum(got_data)/nb_tot*100),end='\r')

    def close_workers(self):
        """ Close the workers"""
        for w in self.workers:
            w.terminate()
        del self.workers
        del self.inp_queue
        del self.out_queue

    def _run(self,model=None):
        """ Run the study """
        if model is None:
            model = self.model
        #Loads the hub
        self.init_workers()
        logging.info("Hub launched.")
        #Fill the queue
        self.fill_input_queue(self.inp_queue,model)
        logging.info("Queue filled.")
        #Run the hub
        self.run_workers()
        logging.info("Hub running.")
        #Get the results
        self.wait_for_results()
        #Close the hub
        self.close_workers()

    @property
    def path(self):
        data_name = None
        if not(self.dataset is None):
            data_name = self.dataset.name
        if not(self.data is None):
            data_name = self.data.name
        if data_name is None:
            raise ValueError("Unamed data or dataset.")
        return os.path.join('data',self.name,self.model.name+'-'+data_name)
    
BasicStudy = BasicCompleteStudy  #alias

class BasicReadStudy(BasicCompleteStudy):
    @property
    def worker_class(self):
        return BasicReadWorker