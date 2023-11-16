# Predict Language

This is a neural network learning to predict if a word is English or Norwegian. The project is based on [this online book](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.

### Run the project

Make sure to install [python](https://www.python.org/downloads/) (version 3.x).

Then, install the required dependencies with

```bash
pip install -r requirements.txt
```

To try the project, run

```bash
# to run the saved model. This project comes with a pre-trained
python main.py

# to train your own model
python train.py

# for debugging, generate the data  files. You may need to clear the folder for this to work
python data_loader.py
```
