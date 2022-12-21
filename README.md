# Deep learning 2 course project

# Car image generator

Deep learning 2 course project which focuses on acquiring knowledge the field of generative models.

## Project Organization

    ├── datasets                <- Folder that contains all raw datasets
    │   ├── cars                <- Dataset from "3D Object Representations for Fine-Grained Categorization"
    ├── docs                    <- Project documentation
    ├── gui                     <- Folder that contains gui elements
    │   ├── icons               <- GUI icons
    │   ├── images              <- GUI images
    │   ├── GUI.py              <- Graphical user interface implementation
    ├── models                  <- Models used in this project
    ├── performance             <- Folder that contains summary of model performance
    ├── states                  <- Folder that contains model states
    ├── util                    <- Folder that contains utility functions
    ├── README.md               <- Top level README.md for developers using this project
    ├── requirements.txt        <- The requirements file for reproducing the environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    ├── train.py                <- Training interface
    ├── generate.py             <- Generating interface
    ├── main.py                 <- Entry point of the project

## Getting started

Create venv:

```bash
python3 -m venv venv
```

Creating venv is required only when running for the first time.

Activate venv:

```bash
source venv/bin/activate
```

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Running locally

+ Interactive GUI experience run `main.py` file.
    + GUI uses trained model which can be configured. The list of trained models ready to be used "out of the box" is
      located in the `states` directory.
+ Training:
    + for training existing models with different hyperparameters run modified `train.py` file.
    + For training new models first create a custom model file in the `models` directory.
+ Generating:
    + In `generate.py` file load trained model and dataset.