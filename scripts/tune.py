import os
import parser
import re
import subprocess
from argparse import ArgumentParser
import numpy as np

import joblib
import optuna

SUBMISSION_FOLDER = ""
NUM_MODELS = 1

import random


def evaluate(weights: np.ndarray):
    """
    Evaluate the model
    """
    return random.random()


def optim_function(params):
    """
    Optimize the model
    """
    # create a np vector from params
    w = np.array([params[f"w_{i}"] for i in range(NUM_MODELS)])
    score = evaluate(w)
    return score


def objective(trial):
    params = {}
    for i in range(NUM_MODELS):
        params[f"w_{i}"] = trial.suggest_float(f"w_{i}", 0.01, 1)
    return optim_function(params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--study-name", type=str, default="base")
    args = parser.parse_args()
    NUM_MODELS = args.n
    study = optuna.create_study(direction="maximize", study_name=args.study_name)
    study.optimize(objective, n_trials=100)

    print(study.best_params)
    joblib.dump(study, f"{args.study_name}.pkl")

