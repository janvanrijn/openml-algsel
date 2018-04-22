import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from examples.calculate_score import calculate_score


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a sklearn algorithm on Oberon splits')
    parser.add_argument('--oasc_scenario_dir', type=str, default='../../oasc/oasc_scenarios/')
    parser.add_argument('--scenario_name', type=str, default='Camilla')
    parser.add_argument('--results_dir', type=str, default='../../oasc/camilla_analysis/results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    scores = []
    for file in os.listdir(args.results_dir):
        schedule = os.path.join(args.results_dir, file)
        _, gap, _, _, _ = calculate_score(args.oasc_scenario_dir, args.scenario_name, schedule)
        scores.append(gap)

    np.random.seed(0)

    mu = 200
    sigma = 25
    n_bins = 500

    fig, ax = plt.subplots(figsize=(8, 4))

    # plot the cumulative histogram
    n, bins, patches = ax.hist(scores, n_bins, normed=1, histtype='step', cumulative=True)

    ax.axvline(x=1, color='black', ls='dashed')

    ax.set_ylim([0,1])
    ax.set_xlim([min(scores),max(scores)])


    # tidy up the figure
    ax.grid(True)
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('Obtained score')
    ax.set_ylabel('Cumulative Likelihood')

    plt.savefig('asap_camilla.pdf')