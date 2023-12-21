import argparse
import os
from itertools import product
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from qiskit import Aer
from functools import partial
from tqdm import tqdm
import json
from qwfc.common import DirectionRuleSet
from qwfc.runner import ClassicalRunnerDefault, QuantumRunnerIBMQAer, HybridRunnerDefault
from tests.example_utils import run_wfc, configure_quantum_runner, pattern_weight_fun, fig2img
from qwfc._version import __version__

def draw_image(mapped_coords, map_size):
    # options
    R = 1
    alpha = .5
    lim = map_size*2
    figsize = (10, 10)
    labels = None
    colormap = {0: '#0071BC', 1: '#FFDF42', 2: '#009B55', 3: '#949698'}
    # plot
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_aspect('equal')
    for idx, (coord, value) in enumerate(mapped_coords.items()):
        x = coord[0] * R
        y = np.sin(np.radians(60)) * (coord[1] - coord[2]) * 2 / 3 * R
        c = colormap[value].lower()
        l = labels[idx] if labels is not None else ''
        hexshape = RegularPolygon((x, y), numVertices=6, radius=2 / 3 * R, orientation=float(np.radians(30)),
                                  facecolor=c,
                                  alpha=alpha, edgecolor=None)
        ax.add_patch(hexshape)
        if len(l) > 0:
            ax.text(x, y, l, ha='center', va='center', size=10)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    plt.grid(False)
    plt.axis('off')
    return fig2img(fig)

def process_result(result, map_size, prefix=''):
    timestamp = str(datetime.now().timestamp())
    prefix = f'{prefix[:64]}{timestamp}-'
    #
    pc = result['pc']
    #
    if not os.path.isdir('results'):
        os.mkdir('results')
    #
    filename = f'results/{prefix}data.json'
    with open(filename, 'w') as fh:
        data = dict(pc = {str(key): (float(p), {str(c): int(v) for c,v in mapped_coords.items()}, bool(f) if f is not None else None) for (key, (p, mapped_coords, f)) in pc.items()}, version=__version__)
        json.dump(data, fh)
    #
    for idx, (key, (p, mapped_coords, f)) in enumerate(pc.items()):
        key = str(key)[:100]
        filename = f'results/{prefix}{idx}-{key}.png'
        print(f'{key}: p={p}, f={f}, file={filename}')
        draw_image(mapped_coords, map_size).save(filename)

def run(map_size, n_chunks=1, alpha = 5, backend_name=None, channel=None, instance=None, use_sv=False, shots=1, engine='Q', name=''):

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

    def coord_list_fun():
        return [(r, s, t) for r in range(-map_size, map_size + 1) for s in range(-map_size, map_size + 1) for t in range(-map_size, map_size + 1) if s + r + t == 0]

    def chunk_map_fun(parsed_counts):
        # use most probable
        for _, (_, coord_map, feasibility) in parsed_counts.items():
            if feasibility is None or feasibility:
                return coord_map
        return None

    def chunk_iter_fun(coord_list):
        for coord_list_seg in np.array_split(coord_list, n_chunks):
            yield [tuple(coord) for coord in coord_list_seg], None

    def hwfc_callback_fun(pbar, hwfc, idx, map_chunk, chunk_mapped_coords):
        pbar.update(1)

    def hwfc_qwfc_callback_fun(pbar, qwfc, idx, coord):
        pbar.set_postfix({'idx': idx, 'coord': coord})

    def qwfc_callback_fun(pbar, qwfc, idx, coord):
        hwfc_qwfc_callback_fun(pbar, qwfc, idx, coord)
        pbar.update(1)

    def cwfc_callback_fun(pbar, cwfc, idx, coord, options, new_value):
        pbar.set_postfix({'idx': idx, 'coord': coord, 'new_value': new_value})
        pbar.update(1)

    # values
    n_values = 4
    # 0 blue
    # 1 yellow
    # 2 green
    # 3 gray

    # coordinates
    coord_list = coord_list_fun()

    # rules
    ruleset = DirectionRuleSet(n_values)
    n_keys = ['n', 'ne', 'se', 's', 'sw', 'nw']
    for adj_vals in product(range(n_values), repeat=len(n_keys)):
        pattern = {n_key: adj_val for n_key, adj_val in zip(n_keys, adj_vals)}
        for v in range(n_values):
            if (v == 0 and 2 not in adj_vals and 3 not in adj_vals) \
            or (v == 1 and 3 not in adj_vals) \
            or (v == 2 and 0 not in adj_vals) \
            or (v == 3 and 0 not in adj_vals and 1 not in adj_vals):
                if v == 0:
                    weight = alpha
                else:
                    weight = 1
                context = {'pattern': pattern, 'weight': weight}
                value_const = v
                ruleset.add(value_const, pattern_weight_fun, context)
    print(f'generated ruleset: {len(ruleset)} rules')

    # setup
    if engine == 'Q':
        # QWFC
        runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel, instance=instance,
                                          shots=shots, check_feasibility=False, add_barriers=False)
        total_steps = len(coord_list)
        pbar = tqdm(total=total_steps, desc='qwfc')
        run_kwargs = dict(coord_path_fun=None, coord_fixed=None, callback_fun=partial(qwfc_callback_fun, pbar))
    elif engine == 'C':
        # CWFC
        n_samples = shots
        runner = ClassicalRunnerDefault(n_samples=n_samples)
        total_steps = len(coord_list) * runner.n_samples
        pbar = tqdm(total=total_steps, desc='cwfc')
        run_kwargs = dict(coord_fixed=None, callback_fun=partial(cwfc_callback_fun, pbar))
    elif engine == 'H':
        # HWFC
        quantum_runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel, instance=instance,
                                          shots=shots, check_feasibility=False, add_barriers=False)
        runner = HybridRunnerDefault(quantum_runner=quantum_runner)
        total_steps = n_chunks
        pbar = tqdm(total=total_steps, desc='hwfc')
        run_kwargs = dict(chunk_map_fun=chunk_map_fun, chunk_iter_fun=chunk_iter_fun,
                          qwfc_callback_fun=partial(hwfc_qwfc_callback_fun, pbar),
                          hwfc_callback_fun=partial(hwfc_callback_fun, pbar))
    else:
        raise NotImplementedError

    # run
    pbar.reset()
    result = run_wfc(runner, n_values, coord_list, coord_neighbors_fun, ruleset, **run_kwargs)
    pbar.close()
    process_result(result, map_size, f'{name}-{engine}-s{map_size}-')


# args
parser = argparse.ArgumentParser()
parser.add_argument('--map_size', type=int, default=2, help='map size')
parser.add_argument('--n-chunks', type=int, default=4, help='map chunks (only for H)')
parser.add_argument('--alpha', type=float, default=5, help='blue weight')
parser.add_argument('--backend-name', type=str, default=None, help='IBMQ backend name, None for local simulator (default: None)')
parser.add_argument('--channel', type=str, default=None, help='IBMQ runtime service channel (default: None)')
parser.add_argument('--instance', type=str, default=None, help='IBMQ runtime service instance (default: None)')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1, help='number of shots for non-statevector and CWFC')
parser.add_argument('--engine', type=str, choices=['C', 'Q', 'H'], default='Q', help='WFC engine: C, Q (default) or H ')
parser.add_argument('--name', type=str, default='hex', help='result filename')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Hexagonal tiles with island-like rules: blue - yellow - green - gray. The probability for blue tiles can be controlled with the parameter alpha.
    """
    run(map_size=args.map_size, n_chunks=args.n_chunks, alpha = args.alpha, backend_name=args.backend_name, channel=args.channel, instance=args.instance, use_sv=args.use_sv, shots=args.shots, engine=args.engine, name=args.name)
