import numpy as np
import pandas as pd
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_csv(filename, **arrays):
    df = pd.DataFrame(arrays)
    df.to_csv(filename, index=False)