import algsel
import argparse
import operator
import sklearn

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on ASLib splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--scenario_name', type=str, default=None)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--model', type=str, default='forest_256')
    return parser.parse_args()

