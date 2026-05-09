## Reproduction

## Running the code
# Codes:
`config.py` contains the configurations of hyperparameters for different datasets. Note that some of the hyperparameters used are different from the article configuration.
`data_preprocessing.py` contains the helper codes for data preprocessing.
`export_artifacts.py` contains the helper codes to export the needed data.
`main.py` contains the codes to run the training process.
`main_test.py` contains the codes to run the evaluation process.
`network.py` contains the structure of the LSTM network.

# Running the workflow
`python main.py --dataset FD00X` to run the training.
`python main_test.py --dataset FD00X` to run the evaluation.

# Note: Please unzip your data within the `./reproduce` directory, or customize the datapath in `config.py` to read the data.
