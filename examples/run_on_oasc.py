import algsel
import argparse
import numpy as np
import operator
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
    return parser.parse_args()


def run_on_scenario(oasc_scenario_dir, scenario_name, meta, repeats):
    train_frame, test_frame, description = algsel.utils.get_train_and_test_frame(oasc_scenario_dir, scenario_name)
    maximize = description['maximize'][0]

    test_tasks = set(test_frame['instance_id'].unique())
    avg_oracle_score = algsel.utils.oracle_score(test_frame, test_tasks, maximize)
    avg_best_algorithm = algsel.utils.get_avg_best_algorithm(train_frame, maximize)
    avg_best_score = algsel.utils.average_best_score(test_frame, avg_best_algorithm, test_tasks)
    golden_standard = algsel.utils.dataframe_to_scores(test_frame)

    task_scores = {task_id: list() for task_id in test_tasks}
    for seed in range(repeats):
        print(algsel.utils.get_current_time(), 'Training model on repeat', seed)
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
        if np.std(scores) == 0:
            print('Taks %d all scores equal' %task_id)


    model_score, gap_score_single, gaps_stdev_single = algsel.utils.task_scores_to_avg(task_scores, avg_oracle_score, avg_best_score)
    return model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score


if __name__ == '__main__':
    args = parse_args()

    if args.scenario_name is not None:
        scenarios = [args.scenario_name]
    else:
        scenarios = ['Camilla', 'Oberon', 'Titus']

    models = {
        'tree': sklearn.tree.DecisionTreeRegressor(),
        'forest_16': sklearn.ensemble.RandomForestRegressor(n_estimators=16),
        'forest_256': sklearn.ensemble.RandomForestRegressor(n_estimators=256)
    }

    pipeline = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.preprocessing.Imputer(strategy=args.impute)),
                                                ('classifier', models[args.model])])

    for scenario_name in scenarios:
        for single_model in [True, False]:
            meta = algsel.utils.ModelWrapper(pipeline, single_model)
            print(algsel.utils.get_current_time(), args.model, 'on', scenario_name, '; single model = ', single_model)
            res = run_on_scenario(args.oasc_scenario_dir, scenario_name, meta, args.repeats)
            model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score = res
            print(algsel.utils.get_current_time(), 'Oracle', avg_oracle_score)
            print(algsel.utils.get_current_time(), 'Single Best', avg_best_score)
            print(algsel.utils.get_current_time(), 'Score', model_score, 'GAP', gap_score_single, '+/-', gaps_stdev_single)
