import algsel
import argparse
import aslib_scenario
import json
import logging
import os
import subprocess
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on ASLib splits')
    parser.add_argument('--aslib_scenario_dir', type=str, default=os.path.expanduser('~/projects/aslib_data'))
    parser.add_argument('--scenario_name', type=str, default='OPENML-WEKA-2017')
    parser.add_argument('--n_repetitions', type=int, default=1)
    parser.add_argument('--n_folds', type=int, default=10)

    return parser.parse_args()


def run(args):
    command = '/home/janvanrijn/anaconda3/envs/asap-v2-stable/bin/python /home/janvanrijn/projects/asap-v2-stable/src/run_asap.py v2'

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for scenario_name in os.listdir(args.aslib_scenario_dir):
        if args.scenario_name is not None and scenario_name != args.scenario_name:
            continue
        for repetition in range(1, args.n_repetitions + 1):
            for fold in range(1, args.n_folds + 1):
                for seed in [1, 2, 3]:
                    logging.info('Running scenario=%s repetition=%d fold=%d' % (scenario_name, repetition, fold))

                    # drop the fold files in the proper directory
                    temp_folder = tempfile.mkdtemp('_asap_%s' % scenario_name)
                    os.makedirs(os.path.join(temp_folder, 'output'), exist_ok=True)  # expected by ASAP
                    temp_data_folder = os.path.join(temp_folder, 'data', 'oasc_scenarios')
                    scenario_folder = os.path.join(args.aslib_scenario_dir, scenario_name)
                    algsel.scenario.save_scenario_in_oasc_format(scenario_folder, scenario_name, temp_data_folder, repetition, fold)

                    # prepare and execute cli command
                    total_command = '%s %s %d' % (command, temp_folder, seed)
                    logging.info('Command: %s' % total_command)
                    p = subprocess.Popen(total_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=r'/tmp')
                    retval = p.wait()
                    if retval != 0:
                        logging.error("\n".join(p.stdout.readlines()))
                    result_file_path = os.path.join(temp_folder, 'output', 'asap_v2_oasc', 'reg_weight_5e-03', '%s-test.json' % scenario_name)
                    with open(result_file_path, 'r') as fp:
                        schedules = json.load(fp)

                    # read scenarios
                    test_scenario = aslib_scenario.aslib_scenario.ASlibScenario()
                    test_scenario.read_scenario(dn=os.path.join(temp_data_folder, 'test', scenario_name))
                    train_scenario = aslib_scenario.aslib_scenario.ASlibScenario()
                    train_scenario.read_scenario(dn=os.path.join(temp_data_folder, 'train', scenario_name))

                    # validate
                    validator = algsel.scoring.Validator()

                    if test_scenario.performance_type[0] == "runtime":
                        stats = validator.validate_runtime(schedules=schedules, test_scenario=test_scenario,
                                                           train_scenario=train_scenario)
                    else:
                        stats = validator.validate_quality(schedules=schedules, test_scenario=test_scenario,
                                                           train_scenario=train_scenario)
    pass


if __name__ == '__main__':
    run(parse_args())
