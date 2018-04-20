import algsel
import argparse
import operator
import sklearn

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on Oberon splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--scenario_name', type=str, default='Oberon')
    parser.add_argument('--single_model', action='store_true', default=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--impute', type=str, default='median')
    return parser.parse_args()


def run(args):
    train_frame, test_frame = algsel.utils.get_train_and_test_frame(args.oasc_scenario_dir, args.scenario_name)

    test_tasks = set(test_frame['instance_id'].unique())
    avg_oracle_score = algsel.utils.oracle_score(test_frame, test_tasks)
    avg_best_algorithm = algsel.utils.get_avg_best_algorithm(train_frame, test_tasks)
    avg_best_score = algsel.utils.average_best_score(test_frame, avg_best_algorithm, test_tasks)
    golden_standard = algsel.utils.dataframe_to_scores(test_frame)

    pipeline = sklearn.pipeline.Pipeline(steps=[('imputer', sklearn.preprocessing.Imputer(strategy=args.impute)),
                                                ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=16))])

    meta = algsel.utils.ModelWrapper(pipeline, args.single_model)
    meta.fit(train_frame)

    predictions = meta.predict(test_frame)

    task_scores = {task_id: list() for task_id in test_tasks}
    for task_id, pred in predictions.items():
        predicted_algorithm = max(pred.items(), key=operator.itemgetter(1))[0]
        task_scores[task_id].append(golden_standard[task_id][predicted_algorithm])

    model_score, gap_score_single, gaps_stdev_single = algsel.utils.task_scores_to_avg(task_scores, avg_oracle_score, avg_best_score)
    print(args.scenario_name)
    print('Score', model_score, 'GAP', gap_score_single, '+/-',gaps_stdev_single)


if __name__ == '__main__':
    run(parse_args())