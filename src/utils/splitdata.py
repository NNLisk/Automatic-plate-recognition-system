## tool for splitting the datasets into train test and val

import splitfolders
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(project_root)
sys.path.insert(0, project_root)

splitfolders.ratio(
    "data/OCR_training_data/CNN_LETTER_DATASET",
    output="data/OCR_training_data/data",
    seed=1337,
    ratio=(.8, .1, .1)
)