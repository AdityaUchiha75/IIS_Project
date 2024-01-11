# IIS_Project
Project work done for the course Intelligent Interactive Systems @ Uppsala University for VT2023

To install the required packages run:
`pip install -r requirements.txt` 

To run the demo:
`python scripts/demoScript.py` 


Directory sructure: 
```bash

C:.
│   .gitignore
│   README.md
│   requirements.txt
│
├───furhat                                    #Scripts dealing with the development of subsystem 2
│   │   backgroundWaves.mp3
│   │   excercises.py
│   │   furhatConfig.py
│   │   mainScript.py
│
├───models                                    #Saved models post training
│   │   best_model_0.pt
│   │   best_model_12.pt
│   │   best_model_7.pt
│   │   frontal_face_features.xml
│   │
│   ├───Emo
│   │       best_model_0.pt
│   │       best_model_1.pt
│   │       best_model_11.pt
│   │       best_model_12.pt
│   │       best_model_4.pt
│   │       best_model_7.pt
│   │
│   └───ValAr
│           model1.pt
│
├───scratch
│   │   CNN_model.py
│   │   model_test_emo.ipynb                  #Script to test the classification based pytorch model (subsystem 1)
│   │   model_test_ValAr.ipynb                #Script to test the regression based pytorch model (subsystem 1)
│   │   model_training4affective_states.ipynb #Script to run training and hyperparameter tuning for the regression based pytorch model
│   │   nn_exploration.ipynb                  #Script to run training and hyperparameter tuning for the classification based pytorch model
│   │   parallel_processing_test.ipynb
│   │   parallel_processing_test.py
│   │   preprocess_data.ipynb
│   │   pytorch_model.py
│   │   scikit_exploration.ipynb              #Script to run training and hyperparameter tuning for the classification based scikit-learn models
│   │   valence_arousal_combination.ipynb     
│   │
│   ├───Cnn  
│   │       CNN_model.py
│   │       CNN_model_2.py                    #Scripts to train and tune the end-to-end CNN model
│   │       tes.ipynb
│   │       test_cnn_model.py
│   │       test_cnn_model_2.py
│   │
│   └───demo_backup
│           demoScript_backup.py
│           utils_backup.py
│
└───scripts
    │   demoScript.py                         #Final demo script for deployment
    │   preprocess_data.py                    #Script for creating the dataset csv after preprocessing
    │   py_feat_approach.py
    │   utils.py


```
