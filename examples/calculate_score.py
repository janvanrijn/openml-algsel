import algsel
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on Oberon splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--scenario_name', type=str, default='Camilla')
    parser.add_argument('--SBS_on_testset', action='store_true', help='Determines how the Single Best Solver is ' +
                                                                      'calculated: as the best on the train set or ' +
                                                                      'the best on the test set. ')
    parser.add_argument('--submissions_dir', type=str, default='../../oasc/submissions/')
    parser.add_argument('--system', type=str, default='ASAP.v2')
    return parser.parse_args()


def calculate_score(oasc_scenario_dir, scenario_name, schedule_file, SBS_on_testset=True):
    train_frame, test_frame, description = algsel.utils.get_train_and_test_frame(oasc_scenario_dir, scenario_name)
    maximize = description['maximize'][0]
    if description['performance_type'][0] != 'solution_quality':
        raise ValueError('Can only calculate quality scenarios')

    test_tasks = set(test_frame['instance_id'].unique())
    avg_oracle_score = algsel.utils.oracle_score(test_frame, test_tasks, maximize)
    if SBS_on_testset:
        avg_best_algorithm = algsel.utils.get_avg_best_algorithm(test_frame, maximize)
    else:
        avg_best_algorithm = algsel.utils.get_avg_best_algorithm(train_frame, maximize)
    avg_best_score = algsel.utils.average_best_score(test_frame, avg_best_algorithm, test_tasks)
    golden_standard = algsel.utils.dataframe_to_scores(test_frame)

    with open(schedule_file, 'r') as fp:
        schedules = json.load(fp)

    task_scores = {}
    for instance, schedule in schedules.items():
        instance_id = int(instance.split('_')[1])
        algorithm_id = schedule[0][0].split('_')[1]
        task_scores[instance_id] = [golden_standard[instance_id][algorithm_id]]

    model_score, gap_score_single, gaps_stdev_single = algsel.utils.task_scores_to_avg(task_scores, avg_oracle_score, avg_best_score)

    return model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score


if __name__ == '__main__':
    args = parse_args()
    schedule_file = args.submissions_dir + '/' + args.system + '/' + args.scenario_name + '.json'
    res = calculate_score(args.oasc_scenario_dir, args.scenario_name, schedule_file, args.SBS_on_testset)
    model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score = res
    print(args.system, 'on', args.scenario_name)
    print('Oracle', avg_oracle_score)
    print('Single Best', avg_best_score)
    print('Score', model_score, 'GAP', gap_score_single, '+/-', gaps_stdev_single)
