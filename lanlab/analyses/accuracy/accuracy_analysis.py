import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter as xl
import os

from lanlab.analyses.analysis import Analysis

class AccuracyAnalysis(Analysis):
    def __init__(self,study,*args,**kwargs):
        #Create a numpy array to store the output data which are strings
        self.data = np.zeros(study.data.shape,dtype='U10')

        super().__init__(study,*args,**kwargs)

    @property
    def labels_and_colors(self):
        return {"wrong":('w','red'),"correct":('c','green')}

    def run(self,study):
        #Count the number of correct and incorrect answers for each question
        for i,line in enumerate(study.data.a):
            for j,seq in enumerate(line):
                qseg = seq[0] #question segment
                aseg = seq[-1] #answer segment
                incorrect = True
                for k in range(len(qseg['keywords']['correct'])):
                    if qseg['keywords']['correct'][k] in aseg['text']:
                        incorrect = False
                        break
                self.data[i][j] = "wrong" if incorrect else "correct"

    def plot(self):
        #Plot the output data as a bar chart with the correct answers in green and the wrong answers in red
        plt.bar(range(len(self.data)),(self.data=="correct").mean(axis=1),color='green',label='correct')
        plt.bar(range(len(self.data)),(self.data=="wrong").mean(axis=1),bottom=(self.data=="correct").mean(axis=1),color='red',label='wrong')
        plt.xticks(range(len(self.data)))
        plt.xlabel('Question index')
        plt.ylabel('Proportions of answers')

    def print_xlsx(self):
        #Print the output data in in rows for each instance of the question with each row being [answer,index,label] and this concatenated for each question
        workbook = xl.Workbook(os.path.join(self.study.path,'accuracy.xlsx'))
        worksheet = workbook.add_worksheet()
        worksheet.write(0,0,'Answer')
        worksheet.write(0,1,'PID')
        worksheet.write(0,2,'Label')
        #Labels is a dictionary with only the initial of the labels and their colors
        labels = {self.labels_and_colors[label][0]:self.labels_and_colors[label][1] for label in self.labels_and_colors}
        for qindex,line in enumerate(self.study.data):
            for aindex,seq in enumerate(line):
                worksheet.write(aindex+1,0+qindex*4,seq[-1]['text']+'\n'+str(seq))
                worksheet.write(aindex+1,1+qindex*4,aindex)
                worksheet.write(aindex+1,2+qindex*4,self.labels_and_colors[self.data[qindex][aindex]][0])
        #Add an automatic coloration to cells depending on the labels
        for label in labels:
            worksheet.conditional_format(1,2,aindex+1,2+qindex*4,{'type':'cell','criteria':'=','value':f'"{label}"','format':workbook.add_format({'bg_color':labels[label]})})
        workbook.close()

                
