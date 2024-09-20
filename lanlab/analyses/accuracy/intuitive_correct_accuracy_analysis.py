import matplotlib.pyplot as plt

#Import the modules
from lanlab.analyses.accuracy.accuracy_analysis import AccuracyAnalysis

class IntuitiveCorrectAccuracyAnalysis(AccuracyAnalysis):

    @property
    def labels_and_colors(self):
        return {"other":('o','orange'),"intuitive":('i','blue'),"correct":('c','green')}

    def run(self,study):
        #Count the number of correct and incorrect answers for each question
        for i,line in enumerate(study.data.a):
            for j,seq in enumerate(line):
                qseg = seq[0] #question segment
                aseg = seq[-1] #answer segment
                correct = False
                for k in range(len(qseg['keywords']['correct'])):
                    if qseg['keywords']['correct'][k] in aseg['text']:
                        correct = True
                        break
                self.data[i][j] = "correct" if correct else "wrong"
                if not(correct):
                    intuitive = False
                    for k in range(len(qseg['keywords']['intuitive'])):
                        if qseg['keywords']['intuitive'][k] in aseg['text']:
                            intuitive = True
                            break
                    self.data[i][j] = "intuitive" if intuitive else "other"
    def plot(self):
        #Plot the output data as a bar chart with the correct answers in green and the wrong answers in red
        bc = plt.bar(range(len(self.data)),(self.data=="correct").mean(axis=1),color='green',label='correct')
        bi = plt.bar(range(len(self.data)),(self.data=="intuitive").mean(axis=1),bottom=(self.data=="correct").mean(axis=1),color='blue',label='intuitive')
        bo = plt.bar(range(len(self.data)),(self.data=="other").mean(axis=1),bottom=(self.data=="correct").mean(axis=1)+(self.data=="intuitive").mean(axis=1),color='orange',label='other')
        plt.xticks(range(len(self.data)))
        plt.xlabel('Question index')
        plt.ylabel('Proportions of answers')
        return bc,bi,bo