This repository examines the potential of neural networks in image classification of X-Ray scans to various pneumonia diagnosis (no infection, bacterial infection, and viral infection). In this code, we examine several models and look into various depths of XAI using CAM (Class Activation Maps). 

The models are able to achieve 99% AUC-ROC for classification of infection presence vs. no infection, 0.93 AUC-ROC for full classification, and 88% AUC-ROC for bacterial vs. viral classification.

Refer to branch container for the unorganized code. This code contains files not present in the main branch. These files were used largely for experiments and the unsucessful ones that are not of high importance are not included in the main branch.

Several of the files present in main branch do not need to be run for the program to run correctly.

There are two models and two containers (one for each). The first model is a bagging model of a singular deep neural network structure. This performs classification of the three classes. The second model is also an ensemble combining two binary classifications. This model performs slightly better, it uses the same deep learning model for the disease presence classification (except the output layer is 2 nodes rather than 3) and leverages 3 different transfer learning models for the disease type classification.

To run the entire program run in this order:
1. analysis.py:
    Run this to see the analysis of the data, optional and the remaining code can run without this process.

2. preprocessing.py:
    Run this to preprocess the data and generate a CSV file with image paths and labels. This is done rather than storing all the images in a variable and passing it into a function. This saves on computer efficiency and runtime.

3. ensemble_model.py:
    Run this to create the ensemble of models used in the first ensemble model

4. run_ensemble.py:
    Run this file to run the first ensemble model on any specific images and view Class Activation maps

5. infection_present.py:
    Run this file to get the first binary classification in the second ensemble model

6. threshholding.py:
    Run this file to get the second binary classification in the second ensemble model

7. run_full_ensemble.py:
    Run this file run the entire second ensemble model.


All other files were used in this study however do not necessarily contribute to the final result.
Bayesian_optimization.py:
    This file was used to run the original bayesian optimization measuring the metrics of learning rate, batch-size, and dropout rate together. The optimal learning rates, batch sizes, and dropout rates were eventually used in the final models.

Bayesian_threshhold.py:
    This file was another bayesian optimization using the parameters learning rate, dropout, and threshhold rate for model classifications. This did not end up yielding any valuable results.

camtest.py:
    This file can be used to directly generate Class Activation Maps (CAMs) from model type and model number. This file was used to test the efficacy of the CAM model.

ensemble_validation.py:
    This file was used to figure out how to build the entire ensemble with sequential binary classification. It was also used to generate the metrics for the models accuracy, F-1 score, recall, etc. etc.

evaluate_models.py:
    This file was used to get the metrics of trained models individually without needing to retrain a new model. Identical random_states were always used to ensure no data leakage (I wasted a day thinking I was done due to data leakage)

infection_type.py:
    This file was used to test various binary classification of disease types before threshholding, L2-regularization, and many other techniques were utilized.

Other files which were not included in this repo, but can be found in container branch. They include the preprocessing for RGB images, tests for RGB images, various models with a single output layer, the experiments for models with a singular layer, and other models with slight variations from those existing in the models folder. However the models folder contains pretty much all of the models which were experimented on.
