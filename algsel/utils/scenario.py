import arff

import pandas as pd


def obtain_dataframe_scenario(meta_features_path, evaluations_path, process_instance_id, algorithm_id_idx):
    def column_to_integer(frame, column_name, splitted_idx):
        frame[column_name] = frame.apply(lambda row: row[column_name].split('_')[splitted_idx], axis=1)

    run_cols = {'instance_id', 'repetition', 'algorithm', 'runstatus'}

    # load features
    features_arff = arff.load(open(meta_features_path))
    features_columns = [att[0] for att in features_arff['attributes']]
    features = pd.DataFrame(features_arff['data'], columns=features_columns)
    if process_instance_id:
        column_to_integer(features, 'instance_id', 1)
    features['instance_id'] = features['instance_id'].astype(int)
    features = features.set_index('instance_id')
    features.drop(['repetition'], axis=1, inplace=True)

    # load evaluations
    evaluations_arff = arff.load(open(evaluations_path))
    evaluations_columns = [att[0] for att in evaluations_arff['attributes']]

    # deduce objective function (based on evaluation columns)
    candidates = set(evaluations_columns) - run_cols
    if len(candidates) == 0:
        raise ValueError('No candidate for objective_function')
    elif len(candidates) > 1:
        raise ValueError('Multiple candidate for objective_function')
    objective_function = list(candidates)[0]

    # merge features and evaluations
    relevant_fields = ['instance_id', 'algorithm', objective_function]
    evaluations = pd.DataFrame(evaluations_arff['data'], columns=evaluations_columns)[relevant_fields]
    if process_instance_id:
        column_to_integer(evaluations, 'instance_id', 1)
    evaluations['instance_id'] = evaluations['instance_id'].astype(int)
    column_to_integer(evaluations, 'algorithm', algorithm_id_idx)
    evaluations = evaluations.set_index(['instance_id'])
    evaluations = evaluations.join(features).reset_index()
    evaluations = evaluations.reindex_axis(sorted(evaluations.columns), axis=1)
    evaluations = evaluations.rename(index=str, columns={objective_function: 'objective_function'})

    # sort columns
    evaluations = evaluations[sorted(evaluations.columns.values)]

    # sort rows and return
    return evaluations.sort_values(['instance_id', 'algorithm']).reset_index().drop('index', axis=1)


def get_train_and_test_frame(oasc_scenario_dir, scenario_name):
    meta_arff_train_location = oasc_scenario_dir + 'train/' + scenario_name + '/feature_values.arff'
    runs_arff_train_location = oasc_scenario_dir + 'train/' + scenario_name + '/algorithm_runs.arff'

    meta_arff_test_location = oasc_scenario_dir + 'test/' + scenario_name + '/feature_values.arff'
    runs_arff_test_location = oasc_scenario_dir + 'test/' + scenario_name + '/algorithm_runs.arff'

    train_frame = obtain_dataframe_scenario(meta_arff_train_location, runs_arff_train_location, True, 1)
    test_frame = obtain_dataframe_scenario(meta_arff_test_location, runs_arff_test_location, True, 1)
    return train_frame, test_frame


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