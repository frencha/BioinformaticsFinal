# BioinformaticsFinal
This is the code, data, and findings (of which I am the sole author) for the UW Tacoma Bioinformatics winter 2019 final project. Code is an extension of work done during the NCBI February 2019 Hackathon, and the original contributions can be found at https://github.com/NCBI-Hackathons/ConsensusML. 

## Folders
The Clinical_Data and Manifest_Data folders contain data needed for experiments.

The scripts folder contains the 3 Python script files needed for experiments. The folder also contains some sample output from recent experiments in the output log text file and important_genes.csv file. Additionally, the visualizations Jupyter notebook contains some interesting visualizations of the feature selection findings.

## Dependencies
In addition to Python, you will need to have numpy, pandas, sklearn, xgboost, and logitboost installed in order to run the Python scripts included. Information for xgboost can be found at https://xgboost.readthedocs.io/en/latest/python/python_api.html. Information for logitboost can be found at https://logitboost.readthedocs.io/. Both the xgboost and logitboost libraries are built on top of sklearn.
