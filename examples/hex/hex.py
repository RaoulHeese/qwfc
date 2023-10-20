import argparse
import sys, os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from qiskit import Aer

sys.path.insert(0, '../../src')
from qwfc import Map, MapSlidingWindow, CorrelationRuleSet
from runner import CircuitRunnerIBMQAer
sys.path.insert(0, '../')
from example_utils import coord_rules_fun_generator


def coord_neighbors_fun(coord):
    # clockwise cycle starting from top (north)
    r, s, t = coord  # cube coordinates
    coord_dict = {'n': (r, s + 1, t - 1),
                  'ne': (r + 1, s, t - 1),
                  'se': (r + 1, s - 1, t),
                  's': (r, s - 1, t + 1),
                  'sw': (r - 1, s, t + 1),
                  'nw': (r - 1, s + 1, t),
                  }
    return coord_dict


def draw_map(mapped_coords, colormap, R=1, alpha=.5, lim=2, figsize=(10, 10), labels=None):
    #
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect('equal')
    for idx, (coord, value) in enumerate(mapped_coords.items()):
        x = coord[0] * R
        y = np.sin(np.radians(60)) * (coord[1] - coord[2]) * 2 / 3 * R
        c = colormap[value].lower()
        l = labels[idx] if labels is not None else ''
        hexshape = RegularPolygon((x, y), numVertices=6, radius=2 / 3 * R, orientation=float(np.radians(30)),
                                  facecolor=c,
                                  alpha=alpha, edgecolor='k')
        ax.add_patch(hexshape)
        if len(l) > 0:
            ax.text(x, y, l, ha='center', va='center', size=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.grid(False)
    plt.axis('off')


def coord_path_fun(coord_list, coord_path_seed):
    shuffled_path_coords = [coord for coord in coord_list]
    np.random.RandomState(coord_path_seed).shuffle(shuffled_path_coords)
    return (coord for coord in shuffled_path_coords)


def process_output(parsed_counts):
    colormap = {0: 'blue', 1: 'green'}
    if not os.path.isdir('results'):
        os.mkdir('results')
    for key, (p, coord_map) in parsed_counts.items():
        filename = f'results/{key}.png'
        print(f'{key}: p={p}, file={filename}')
        draw_map(coord_map, colormap, lim=3, figsize=(3, 3))
        plt.savefig(filename)
        plt.close()


def run(map_lim, use_sv=True, shots=1000, coord_path_seed=42):
    # input
    n_values = 2
    adj_order = ['n', 'ne', 'se', 's', 'sw', 'nw']
    rules = {}
    for r in product((None, 0, 1), repeat=6):
        len0 = sum([1 for b in r if b == 0])
        len1 = sum([1 for b in r if b == 1])
        lenN = sum([1 for b in r if b is None])
        assert len0 + len1 + lenN == 6
        if len0 == 6:
            rules[r] = {1: 1}
        elif len1 > 0:
            rules[r] = {0: 1}
        else:
            rules[r] = {0: .5, 1: .5}
        # island rules
    coord_rules_fun = coord_rules_fun_generator(adj_order, rules, lambda: CorrelationRuleSet(n_values))
    if use_sv:
        backend = Aer.get_backend('statevector_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')
    circuit_runner = CircuitRunnerIBMQAer(backend=backend, run_kwarg_dict = dict(shots=shots))
    coord_list = [(r, s, t) for r in range(-map_lim, map_lim + 1) for s in range(-map_lim, map_lim + 1) for t in
                  range(-map_lim, map_lim + 1) if s + r + t == 0]
    check_feasibility = False

    # run
    print(f'hex: run full map generation (circuit_runner={circuit_runner})...')
    m = Map(n_values, coord_list, coord_neighbors_fun, check_feasibility)
    m.run(coord_rules_fun, lambda cl: coord_path_fun(cl, coord_path_seed), circuit_runner=circuit_runner, callback_fun=None)

    # output
    print('finished:')
    process_output(m.pc)


# args
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--map-lim', type=int, default=2, help='map size')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('-s', '--shots', type=int, default=1000, help='number of shots for non-statevector')
parser.add_argument('--coord-path-seed', type=int, default=42, help='random seed')

args = parser.parse_args()

if __name__ == '__main__':
    """
    Hexagonal tiles with island-like rules: green tiles are only allowed to have blue neighbors. The tile traversal is performed in a random order.
    """

    run(args.map_lim, args.use_sv, args.shots, args.coord_path_seed)
