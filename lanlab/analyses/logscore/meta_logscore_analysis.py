from lanlab.analyses.logscore.logscore_analysis import LogScoreAnalysis
from lanlab.analyses.analysis import Analysis
import matplotlib.pyplot as plt
import numpy as np

class MetaLogScoreAnalysis(Analysis):
    """ MetaLogScoreAnalysis is a LogScoreAnalysis that can be run on multiple studies and plot the evolution across studies."""
    def __init__(self,*args,**kwargs):
        self.data = []
        super().__init__(*args,**kwargs)
    
    def run(self,studies):
        """ Run the analysis on multiple studies """
        for study in studies:
            self.data.append(LogScoreAnalysis(study))
        #Reshape to fit in a single array of sequences of shape (n_studies,n_questions)
        self.data = np.array([d.data for d in self.data],object)

    def plot_comparison(self):
        """ Plot the comparison of the analysis """
        fig = plt.figure()
        for i,data in enumerate(self.data):
            for j,(t,lp,m) in enumerate(data):
                plt.scatter([m.params[0]],[m.params[1]],label=str(i))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('A')
        plt.ylabel('B')

    def plot_evolution_comparison(self):
        """ Plot the evolution of the comparison of the analysis """
        fig = plt.figure()
        #Plot the arrows between the studies to show the evolution
        params0 = np.array([[self.data[si,qi,2].params[0] for qi in range(len(self.data[0]))] for si in range(len(self.data))])
        params1 = np.array([[self.data[si,qi,2].params[1] for qi in range(len(self.data[0]))] for si in range(len(self.data))])
        #Different colors for each question qi
        colors = [plt.cm.jet(i/float(len(self.data[0])-1)) for i in range(len(self.data[0]))]
        for si in range(len(self.data)-1):
            plt.quiver(params0[si],
                       params1[si],
                       params0[si+1]-params0[si],
                       params1[si+1]-params1[si],
                       angles='xy',scale_units='xy',scale=1,color=colors)
        #Plot the points of the study
        plt.scatter(params0.reshape(-1),params1.reshape(-1),c=np.array(colors*len(self.data)))
        #Put colors in the legend with the index of the question
        for i in range(len(self.data[0])):
            plt.scatter([],[],c=colors[i],label=str(i))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.ylim(10**-1,10**2)
        plt.xlim(10**-1,10**2)
        
