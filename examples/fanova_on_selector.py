import algsel
import argparse
import ConfigSpace
import fanova
import numpy as np
import os
import pickle
import sklearn

from examples.run_on_oasc import run_on_scenario

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on ASLib splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--cache_dir', type=str, default=os.path.expanduser("~") + '/experiments/fanova/cache')
    parser.add_argument('--scenario_name', type=str, default='Oberon')
    parser.add_argument('--lower_cutoff', type=float, default=None)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--optimization_runs', type=int, default=150)
    parser.add_argument('--model', type=str, default='forest_256')
    return parser.parse_args()


def get_config_space():
    cs = ConfigSpace.configuration_space.ConfigurationSpace(name='Random Forest Algorithm Selection', seed=42)

    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter('imputer__strategy', ['mean', 'median', 'most_frequent'], default_value="median")
    # criterion = ConfigSpace.hyperparameters.CategoricalHyperparameter("classifier__criterion", ["gini", "entropy"], default_value="gini")
    max_features = ConfigSpace.hyperparameters.UniformFloatHyperparameter("classifier__max_features", 0., 1., default_value=0.5)
    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("classifier__n_estimators", 8, 256, default_value=64, log=True)
    min_samples_split = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("classifier__min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = ConfigSpace.hyperparameters.UniformIntegerHyperparameter("classifier__min_samples_leaf", 1, 20, default_value=1)
    bootstrap = ConfigSpace.hyperparameters.CategoricalHyperparameter("classifier__bootstrap", ["True", "False"], default_value="True")
    single_model = ConfigSpace.hyperparameters.CategoricalHyperparameter("single_model", ["True", "False"], default_value="True")

    cs.add_hyperparameters([imputation, n_estimators, max_features,
                            min_samples_split, min_samples_leaf, bootstrap, single_model])
    return cs


def run_iteration(args, iteration_id, config):
    cache_dir = os.path.join(args.cache_dir, args.scenario_name)
    try:
        os.makedirs(cache_dir)
    except FileExistsError:
        pass
    cache_file = os.path.join(cache_dir, str(iteration_id) + '.pkl')
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as fp:
            print('Loaded iteration %d from cache' %iteration_id)
            return pickle.load(fp)

    pipeline = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.preprocessing.Imputer()),
                                                ('classifier', sklearn.ensemble.RandomForestRegressor())])
    sklearn_params = config.get_dictionary()
    single_model = sklearn_params['single_model']
    del sklearn_params['single_model']
    pipeline.set_params(**sklearn_params)
    meta = algsel.utils.ModelWrapper(pipeline, single_model)
    result = run_on_scenario(args.oasc_scenario_dir, args.scenario_name, meta, args.repeats)

    with open(cache_file, 'wb') as fp:
        pickle.dump(result, fp)

    return result


def _do_fanova(config_space, configurations, performances, lower_cutoff):
    X = list()

    for configuration in configurations:
        current = []

        for param in config_space.get_hyperparameters():
            value = configuration[param.name]

            if isinstance(param, ConfigSpace.hyperparameters.CategoricalHyperparameter):
                value = param.choices.index(value)

            current.append(value)

        X.append(current)

    X = np.array(X)
    y = np.array(performances)

    cutoffs = (-np.inf, np.inf)
    if lower_cutoff is not None:
        p75 = np.percentile(y, lower_cutoff)
        p100 = np.percentile(y, 100.0)
        cutoffs = (p75, p100)

    # start the evaluator
    evaluator = fanova.fanova.fANOVA(X=X, Y=y, config_space=config_space, cutoffs=cutoffs, n_trees=128)

    # obtain the results
    result = {}

    for idx, param in enumerate(config_space.get_hyperparameters()):
        importance = evaluator.quantify_importance([idx])[(idx,)]['total importance']
        result[param.name] = importance
    return result


def run(args):
    config_space = get_config_space()
    configurations = config_space.sample_configuration(args.optimization_runs)
    performances = list()

    for idx, configuration in enumerate(configurations):
        result = run_iteration(args, idx, configuration)

        # first save the results before doing fanova:
        # paralleling and robustness
        performances.append(result[1])

    result = _do_fanova(config_space, configurations, performances, args.lower_cutoff)
    print(result)


if __name__ == '__main__':
    run(parse_args())
