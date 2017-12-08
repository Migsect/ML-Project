### Introduction ###

This is the Project for the Machine Learning course for:
- Nolan Reed
- Alex Sammons
- Shane Smith

The main motivation for this project is to determine if we can train a Naïve Bayes Classifier (specifically a Multinomial Naïve Bayes classifier) to be able to correctly predict the creator of a forum post.

### Usage ###

To use this code you must first download the data, process it, and then run the training script (outputs accuracy, doesn't save anything).

To run the code you will need to have Anaconda for Python 3.6 or higher to installed.  This project makes use of matplotlib for graphs as well as numpy for some utilities.  The implementation of the Multinomial Naïve Bayes Classifier is done all within code.

__Downloading the Data:__
```
python download_data.py
```

_Do note that the link to the data may break in the future.  It is currently hosted on one of the team member's Google Drive_

__Processing the Data:__
```
python process_data.py
```

__Training on the Data:__
```
python train_classifier.py [processed data file]
```