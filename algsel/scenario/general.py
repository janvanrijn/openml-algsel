import arff

import pandas as pd


def obtain_dataframe_scenario(meta_features_path, evaluations_path, feature_status_path, process_instance_id, algorithm_id_idx):
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
    features = features.set_index(['instance_id', 'repetition'])

    # load feature status
    feature_status_arff = arff.load(open(feature_status_path))
    feature_status_columns = [att[0] for att in feature_status_arff['attributes']]
    feature_status = pd.DataFrame(feature_status_arff['data'], columns=feature_status_columns)
    if process_instance_id:
        column_to_integer(feature_status, 'instance_id', 1)
    feature_status['instance_id'] = feature_status['instance_id'].astype(int)
    feature_status = feature_status.set_index(['instance_id', 'repetition'])

    # merge feature with feature status
    features = features.join(feature_status)

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

    # further pre-process evaluations
    relevant_fields = ['instance_id', 'algorithm', 'repetition', objective_function]
    evaluations = pd.DataFrame(evaluations_arff['data'], columns=evaluations_columns)[relevant_fields]
    if process_instance_id:
        column_to_integer(evaluations, 'instance_id', 1)
    evaluations['instance_id'] = evaluations['instance_id'].astype(int)
    column_to_integer(evaluations, 'algorithm', algorithm_id_idx)
    evaluations = evaluations.set_index(['instance_id', 'repetition'])

     # merge
    evaluations = evaluations.join(features).reset_index()
    evaluations = evaluations.reindex(sorted(evaluations.columns), axis=1)
    evaluations = evaluations.rename(index=str, columns={objective_function: 'objective_function'})

    # sort columns
    evaluations = evaluations[sorted(evaluations.columns.values)]

    # sort rows and return
    return evaluations.sort_values(['instance_id', 'algorithm']).reset_index().drop('index', axis=1)