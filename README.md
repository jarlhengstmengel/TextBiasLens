# README for python code for Bachelor Thesis of Jarl Hengstmengel

The repository contains the following scripts:
- Preprocessing.py
- Model.py
- ConceptAnalogyGeneration.py
- BiasEavluation.py
- AnalyzingSurvey.py

All scripts contain a section for setting parameters to run.
To run the scripts some packages must be installed first:
- SoMaJo
- Gensim
- Numpy
- scikit-learn

## Preprocessing.py
This script contains code for preprocessing the traing data text files by stripping the line counts and outputs a text file with one sentence per line.
Parameters:
- corpus_path: Path to the raw text data file
- mod_corpus: Path to text file for saving the output

## Model.py
This script contains code for training a fastText model and evaluating it during training.
Parameters:
- learning_data: Path to text file containing the learning data
- evaluation_data: Path to text file containg the evaluation analogies
- model_save_dir: Directory for saving model
- eval_score_dir: Directory for saving evaluation scores 
- trainable_model: directory for loading pre trained model
- vector_size: Dimensionality of embedding
- window: Size of the Context Window
- sg: If set to 0 traing with CBOW Model otherwise Skip-gram is used
- seed: seed for initialising parameters
- negative: Number of negatives to sample for negative sampling
- epochs: Number of Epochs to train

## ConceptAnalogyGeneration.py
This script contains code for generating analogies.
Parameters:
- model_path: Path to binary file containing pretrained model
- save_dir: Directory for saving text file containg analogies
- defining_sets_path:Path to text file containg the defining groups except the dominant group
- comparing_set: Path to text file containg the comparing set, in our case the school levels
- threshold: threshold for similarity
- dom_groups: string array with the dominant groups, in our case "Weiß"

## BiasEvaluation.pi
This Script contains code for evaluating the embedding with PCA and Direct Bias Metric
Parameters:
- model_path: Path to binary file containing pretrained model
- defining_sets_path: Path to text file containg the defining groups except the dominant group
- comparing_set_txt:  Path to text file containg the comparing set, in our case the school levels
- save dir: Path to directory to save the text file containing the results to
- dom_group: String containg the dominat group, in our case "Weiß"
- bias_strictness: strictness threshold for direct bias score

## AnalyzingSurvey.py
This scripts generates diagramms for analyzing the survey results.

## Results used in the thesis
The folder Used_Results_Thesis contains the generated analogies, generated diagrams and analysation output that was used in the thesis itself. 
