import algsel
import argparse
import logging


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


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    args = parse_args()
    schedule_file = args.submissions_dir + '/' + args.system + '/' + args.scenario_name + '.json'
    res = algsel.utils.calculate_oasc_score(args.oasc_scenario_dir, args.scenario_name, schedule_file, args.SBS_on_testset)
    model_score, gap_score_single, gaps_stdev_single, avg_oracle_score, avg_best_score = res
    logging.info('%s on %s' % (args.system, args.scenario_name))
    logging.info('Oracle %f' % avg_oracle_score)
    logging.info('Single Best %f' % avg_best_score)
    logging.info('Score %f; GAP %f +/- %f' % (model_score, gap_score_single, gaps_stdev_single))
