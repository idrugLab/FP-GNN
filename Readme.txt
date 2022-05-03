Environment:

The most important python packages are:
python == 3.6.7
pytorch == 1.2.0
torch == 0.4.1
tensorboard == 1.13.1
rdkit == 2019.09.3
scikit-learn == 0.22.2.post1
hyperopt == 0.2.5
numpy == 1.18.2

For using our model more conveniently, we provide the environment file <environment.txt> to install environment directly.


Command:

1. Train
Use train.py
Args:
  --data_path : The path of input CSV file. E.g. input.csv
  --dataset_type : The type of dataset. E.g. classification  or  regression
  --save_path : The path to save output model. E.g. model_save


2. Predict
Use predict.py
Args:
  --predict_path : The path of input CSV file to predict. E.g. input.csv
  --result_path : The path of output CSV file. E.g. output.csv
  --model_path : The path of trained model. E.g. model_save/model.pt


3.Hyperparameters Optimization
Use hyper_opti.py
Args:
  --data_path : The path of input CSV file. E.g. input.csv
  --dataset_type : The type of dataset. E.g. classification  or  regression
  --save_path : The path to save output model. E.g. model_save
  --log_path : The path to record and save the result of hyperparameters optimization. E.g. log


4.Interpretation of fingerprints
Use interpretation_fp.py
Args:
  --predict_path : The path of input CSV file. E.g. input.csv
  --model_path : The path of trained model. E.g. model_save/model.pt
  --result_path : The path of result. E.g. result.txt

5.Interpretation of graph
Use interpretation_graph.py
Args:
  --predict_path : The path of input CSV file. E.g. input.csv
  --model_path : The path of trained model. E.g. model_save/model.pt
  --figure_path : The path to save figures of graph interpretation. E.g. figure



