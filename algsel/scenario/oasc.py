import algsel
import yaml


def get_oasc_train_and_test_frame(oasc_scenario_dir, scenario_name):
    meta_arff_train_location = oasc_scenario_dir + 'train/' + scenario_name + '/feature_values.arff'
    runs_arff_train_location = oasc_scenario_dir + 'train/' + scenario_name + '/algorithm_runs.arff'
    status_arff_train_location = oasc_scenario_dir + 'train/' + scenario_name + '/feature_runstatus.arff'

    meta_arff_test_location = oasc_scenario_dir + 'test/' + scenario_name + '/feature_values.arff'
    runs_arff_test_location = oasc_scenario_dir + 'test/' + scenario_name + '/algorithm_runs.arff'
    status_arff_test_location = oasc_scenario_dir + 'test/' + scenario_name + '/feature_runstatus.arff'

    description_location = oasc_scenario_dir + 'test/' + scenario_name + '/description.txt'

    with open(description_location, 'r') as fp:
        description = yaml.load(fp)

    train_frame = algsel.scenario.obtain_dataframe_scenario(meta_arff_train_location, runs_arff_train_location, status_arff_train_location, True, 1)
    test_frame = algsel.scenario.obtain_dataframe_scenario(meta_arff_test_location, runs_arff_test_location, status_arff_test_location, True, 1)
    return train_frame, test_frame, description


def test_frame_to_scores(dataframe):
    tasks = dataframe.instance_id.unique()
    algorithms = dataframe.algorithm.unique()
    task_algorithm_score = {task: dict() for task in tasks}

    for instance_id in tasks:
        frame = dataframe[dataframe['instance_id'] == instance_id]

        for algorithm_id in algorithms:
            score = frame[frame['algorithm'] == algorithm_id]['objective_function'].iloc[0]
            task_algorithm_score[instance_id][algorithm_id] = score

    return task_algorithm_score
