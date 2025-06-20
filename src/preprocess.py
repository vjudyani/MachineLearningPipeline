import pandas as pd
import numpy as np
import os
import sys
import yaml

## Load parameters from param.yaml

params=yaml.safe_load(open( "params.yaml"))['preprocess']


def preprocess(input_path, output_path):
    data = pd.read_csv(input_path, header=None)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False, header=None)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess(params['input'], params['output'])