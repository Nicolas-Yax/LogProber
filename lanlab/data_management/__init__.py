import pickle

def load_study(study_path):
    """ Load a study from a path """
    with open(study_path,'rb') as f:
        study = pickle.load(f)
    return study