This repository contains the scripts with which the results presented in the article titled "Robust hybrid model for breast cancer detection using thermal images and clinical data" are obtained with. The data needed for this purpose is available upon sign up in https://visual.ic.uff.br/dmi/. Recall that the data need to be cleansed as explained in the article before running the scripts.
This repository contains four Jupyter Notebooks and a python file:
  - Segmentation.ipynb: This notebook contains the script to train the different segmentation models tested in the article. The data used for this purpose is obtained from: https://visual.ic.uff.br/en/proeng/marques/
  - Clinical_data_classifiers.ipynb: This notebook contains the experimentation to identify the most suitable machine learning algorithm to classify a set of clinical data.
  - Image_classifiers.ipynb: This notebook contains the experimentation to identify the most suitable deep learning model to classify thermal images from 3 views (front, left, and right).
  - Ensemble_models.ipynb: This notebook contains the experimentation to identify the most suitable method to combine the outputs of the clinical data classifier and the image classifier.
  - Utils.py: This python file contains function and class definitions, used in the previous notebooks.
