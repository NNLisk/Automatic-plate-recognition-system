# Foundations of Artificial Intelligence

## Project overview

## folder structure

```
.
├── data
│   ├── Annotations
│   ├── inference
│   │   ├── raw
│   │   └── sessions
│   │       ├── 1
│   │       ├── 2
│   │       └── 3
│   └── plate_detector_training_data
│       ├── train
│       │   ├── images
│       │   └── labels
│       └── validate
│           ├── images
│           └── labels
├── models
│   └── plate_detector
│       └── weights
├── Notes
│   └── different_thresholds
├── src
│   ├── preprocessing
│   │   └── __pycache__
│   ├── training
│   └── utils
│       └── __pycache__
├── tests
│   └── temp
└── ui
```
+ data is for storing pictures before and after processing
+ models is for storing the model data, currently only the plate detector: weights and graphs
+ Notes: notes
+ src: python scripts for everything
+ ui: interface stuff

+ data storing notation:
    + currently stores everything in two folders:
        + inference/ - for the data inputted by user for its preprocessed derivatives
        + Raw/ - for storing training data

## Collaborators

+ Niko Lausto
+ Jussi Grönroos
+ Jesse Mahkonen
+ Iikka Harjamäki