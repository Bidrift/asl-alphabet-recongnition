# asl-alphabet-recongnition
A machine learning model to recognize the alphabet of the american sign language. In this model we use PyTorch and a dataset from Kaggle

# Link for dataset

https://www.kaggle.com/datasets/grassknoted/asl-alphabet/

# Set up
## For preprocessing
First, make sure that you have pip, kaggle installed using the following commands (Linux)

```sh
sudo apt install python3-pip
```

```sh
pip3 install kaggle
```

Extract your kaggle API informatonn from your kaggle profile and put it inside `kaggle.json` in the main directory and windows config directory. This file will not be committed.

Open data_analysis.ipynb and run the section of "Download"

To run setting up data, run the following commands

```sh
pip3 install numpy
pip3 install split-folders
pip3 install tensorflow[and-cuda]
pip3 install opencv-python
pip3 install matplotlib
pip3 install pandas
```

If you import a preprocessed dataset, you should put it in dataset/preprocessed/asl_alphabet/...

For example, 

dataset/preprocessed/asl_alphabet/train/A/A1.jpg

dataset/preprocessed/asl_alphabet/test/A/A1.jpg

dataset/preprocessed/asl_alphabet/val/A/A1.jpg

## For training
Here goes the description for training steps
