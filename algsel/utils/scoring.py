import numpy as np
import random


def gap_fn(score, vbs, sbs):
    return (score - vbs) / (sbs - vbs)


def _bootstrap_model_scores(task_scores, n_samples):
    # for n_samples, returns a random score from each task
    all_results = []
    for i in range(n_samples):
        sum = 0.0
        for _, scores in task_scores.items():
            sum += random.choice(scores)
        all_results.append(sum / len(task_scores))
    return all_results


def oracle_score(dataframe, test_tasks, maximize):
    if maximize:
        task_maxscore = dataframe.groupby(['instance_id'], sort=False)['objective_function'].max().to_dict()
    else:
        task_maxscore = dataframe.groupby(['instance_id'], sort=False)['objective_function'].min().to_dict()

    total_score = 0
    for task_id in test_tasks:
        total_score += task_maxscore[task_id]
    return total_score / len(test_tasks)


def get_avg_best_algorithm(dataframe, test_tasks, maximize):
    setup_ids = set(dataframe['algorithm'])

    avg_best_algorithm_trainset = None
    best_avg_score_trainset = None

    for setup_id in setup_ids:
        total_score = 0
        setups_frame = dataframe[dataframe['algorithm'] == setup_id]
        for _, row in setups_frame.iterrows():
            if test_tasks is None or row['instance_id'] not in test_tasks:
                total_score += row['objective_function']
        avg_score_trainset = total_score / len(setups_frame)
        if best_avg_score_trainset is None or \
                (maximize and avg_score_trainset > best_avg_score_trainset) or \
                (not maximize and avg_score_trainset < best_avg_score_trainset):
            avg_best_algorithm_trainset = setup_id
            best_avg_score_trainset = avg_score_trainset
    return avg_best_algorithm_trainset


def average_best_score(dataframe, avg_best_algorithm, test_tasks):
    def get_scores_best_avg(avg_best_algorithm, dataframe):
        derived = dataframe[(dataframe['algorithm'] == avg_best_algorithm)].set_index('instance_id')
        return derived['objective_function'].to_dict()

    all_scores = get_scores_best_avg(avg_best_algorithm, dataframe)
    total_test_score = 0.0
    for task_id in test_tasks:
        total_test_score += all_scores[task_id]

    return total_test_score / len(test_tasks)


def task_scores_to_avg(task_scores, avg_oracle_score, avg_best_score):
    sum_avg_scores = 0.0
    for task, scores in task_scores.items():
        sum_avg_scores += sum(scores) / len(scores)
    avg_score = sum_avg_scores / len(task_scores)

    gap_score = gap_fn(avg_score, avg_oracle_score, avg_best_score)

    many_scores = _bootstrap_model_scores(task_scores, 1000)
    many_gaps = []
    for score in many_scores:
        gap = gap_fn(score, avg_oracle_score, avg_best_score)
        many_gaps.append(gap)

    return avg_score, gap_score, np.std(many_gaps)