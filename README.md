# LogProber

🔬🤖 LogProber is a cost-effective tool to measure contamination of language models on given sequences of text requiring very little information about the language model itself. It is particularly useful when working with question / answer pairs coming from benchmarks or psychology questionnaires.

📖 paper : https://www.arxiv.org/abs/2408.14352

🔬 This repository contains the code for replicating figures in the paper and is given for transparency and replication purposes and may not be fit for production environments in terms of optimization requirements.

🌐 We encourage people that are interested in LogProber to use the colab demo that implements the algorithm in a simple and versatile manner.
colab demo : https://colab.research.google.com/drive/1GDbmEMmCVEOwhYk6-1AothdXeAlnqZ_j?usp=copy

## Step by step installation instructions
This repository uses several libraries but some are optional depending on what you want to plot with LogProber.

- Install the base of LogProber
```
pip install -r requirements.txt
```

## Documentation
This project is built on top of lanlab, a simple library to automate queries to LLMs. The lanlab folder contains the basic materials to make the framework run. The logscores.ipynb notebook contains all the code to replicate the results using lanlab features. If you are interested in better understanding how lanlab works you can refer to the main.ipynb notebook that explains the basics of lanlab.
