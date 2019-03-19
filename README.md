# BioinformaticsFinal
This is the code, data, and findings (of which I am the sole author) for the UW Tacoma Bioinformatics Winter 2019 final project. Code is an extension of work done during the NCBI February 2019 Hackathon, and the original contributions can be found at https://github.com/NCBI-Hackathons/ConsensusML. 

## Folders
The Clinical_Data and Manifest_Data folders contain data needed for experiments and scripts.

The scripts folder contains the 3 Python script files needed for experiments. The folder also contains some sample output from recent experiments in the output log text file and important_genes.csv file. Additionally, the visualizations Jupyter notebook contains some interesting visualizations of the feature selection findings.

## Dependencies
In addition to Python, you will need to have numpy, pandas, sklearn, xgboost, and logitboost installed in order to run the Python scripts included. Information for xgboost can be found at https://xgboost.readthedocs.io/en/latest/python/python_api.html. Information for logitboost can be found at https://logitboost.readthedocs.io/. Both the xgboost and logitboost libraries are built on top of sklearn.

## Running the Experiments
To run the experiments, the dependencies will need to be installed and Clinical_Data, Manifest_Data, and scripts folders will need to be downloaded and must be placed into the same folder. The model_walkthrough.py script is the main script. The script can be ran in any Python IDE or can be run on command line or terminal with the command "./model_walkthrough.py".
