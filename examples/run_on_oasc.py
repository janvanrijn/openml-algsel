import algsel
import argparse
import logging
import numpy as np
import operator
import pandas as pd
import sklearn

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on ASLib splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--scenario_name', type=str, default='Camilla')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--impute', type=str, default='median')
    parser.add_argument('--model', type=str, default='forest_256')
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


def dataframe_to_scores(dataframe):
    tasks = dataframe.instance_id.unique()
    algorithms = dataframe.algorithm.unique()
    task_algorithm_score = {task: dict() for task in tasks}

    for instance_id in tasks:
        frame = dataframe[dataframe['instance_id'] == instance_id]

        for algorithm_id in algorithms:
            score = frame[frame['algorithm'] == algorithm_id]['objective_function'].iloc[0]
            task_algorithm_score[instance_id][algorithm_id] = score

    return task_algorithm_score


def run_on_scenario(oasc_scenario_dir, scenario_name, meta, repeats, verbose):
    train_frame, test_frame, description = algsel.scenario.get_oasc_train_and_test_frame(oasc_scenario_dir, scenario_name)
    maximize = description['maximize'][0]

    test_tasks = set(test_frame['instance_id'].unique())
    avg_oracle_score = algsel.utils.oracle_score(test_frame, test_tasks, maximize)
    avg_best_algorithm = algsel.utils.get_avg_best_algorithm(train_frame, maximize)
    avg_best_score = algsel.utils.average_best_score(test_frame, avg_best_algorithm, test_tasks)
    golden_standard = algsel.scenario.test_frame_to_scores(test_frame)

    task_scores = {task_id: list() for task_id in test_tasks}
    for seed in range(repeats):
        logging.info('Training model on repeat %d' % seed)
        meta.model_template.set_params(classifier__random_state=seed)
        meta.fit(train_frame)

        predictions = meta.predict(test_frame)

        for task_id, pred in predictions.items():
            if maximize:
                predicted_algorithm = max(pred.items(), key=operator.itemgetter(1))[0]
            else:
                predicted_algorithm = min(pred.items(), key=operator.itemgetter(1))[0]

            task_scores[task_id].append(golden_standard[task_id][predicted_algorithm])

    if len(task_scores) != len(test_tasks):
        raise ValueError()
    for task_id, scores in task_scores.items():
        if len(scores) != repeats:
            raise ValueError('Expected %d scores, got %d' %(repeats, len(scores)))
        if np.std(scores) == 0 and verbose:
            print('Instance %d all scores equal' % task_id)

    res = algsel.utils.task_scores_to_avg(task_scores, avg_oracle_score, avg_best_score)
    model_score, gap_score_single, gaps_stdev_single = res
    return model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if args.scenario_name is not None:
        scenarios = [args.scenario_name]
    else:
        scenarios = [
            'Camilla',
            'Oberon',
            # 'Titus'
        ]

    models = {
        'tree': sklearn.tree.DecisionTreeRegressor(),
        'forest_16': sklearn.ensemble.RandomForestRegressor(n_estimators=16),
        'forest_256': sklearn.ensemble.RandomForestRegressor(n_estimators=256)
    }

    pipeline = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.preprocessing.Imputer(strategy=args.impute)),
                                                ('classifier', models[args.model])])

    for scenario_name in scenarios:
        for single_model in [True, False]:
            meta = algsel.models.SklearnModelWrapper(pipeline, single_model)
            logging.info('%s on %s; single model = %s' % (args.model, scenario_name, single_model))
            result = run_on_scenario(args.oasc_scenario_dir, scenario_name, meta, args.repeats, args.verbose)
            model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score = result
            logging.info('Oracle %f' % avg_oracle_score)
            logging.info('Single Best %f' % avg_best_score)
            logging.info('Score %f; GAP %f +/- %f' % (model_score, gap_score_single, gaps_stdev_single))


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
