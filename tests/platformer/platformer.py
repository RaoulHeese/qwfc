import argparse
import json
import os
from datetime import datetime
from functools import partial

import numpy as np
from tqdm import tqdm

from qwfc._version import __version__
from qwfc.common import DirectionRuleSet
from qwfc.runner import ClassicalRunnerDefault, HybridRunnerDefault
from tests.example_utils import run_wfc, configure_quantum_runner, pattern_weight_fun, compose_image


def draw_image(mapped_coords):
    image_path = 'platformer-tiles.png'  # https://opengameart.org/content/2d-four-seasons-platformer-tileset-16x16
    sprite_map = {0: (0, 1),  # ground
                  1: (0, 0),  # grass
                  2: (9, 0),  # mushroom
                  3: (0, 2),  # ?
                  4: (9, 4),  # air
                  5: (0, 5),  # tree-low
                  6: (0, 4),  # tree-mid
                  7: (0, 3)  # tree-high
                  }
    background_tile = (3, 11)
    sprite_size = 16
    return compose_image(mapped_coords, image_path, sprite_map, sprite_size, background_tile)


def process_result(result, prefix=''):
    timestamp = str(datetime.now().timestamp())
    prefix = f'{prefix[:64]}{timestamp}-'
    #
    pc = result['pc']
    #
    if not os.path.isdir('results'):
        os.mkdir('results')
    #
    for idx, (key, (p, mapped_coords, f)) in enumerate(pc.items()):
        key = str(key)[:100]
        filename = f'results/{prefix}{idx}-{key}.png'
        print(f'{key}: p={p}, f={f}, file={filename}')
        draw_image(mapped_coords).save(filename)
    #
    filename = f'results/{prefix}data.json'
    with open(filename, 'w') as fh:
        data = dict(pc={
            str(key): (float(p), {str(c): int(v) for c, v in mapped_coords.items()}, bool(f) if f is not None else None)
            for (key, (p, mapped_coords, f)) in pc.items()}, version=__version__)
        json.dump(data, fh)


def run(map_x_size, map_y_size, n_chunks=1, alpha=.1, backend_name=None, channel=None, instance=None, use_sv=False,
        shots=1, engine='Q', name=''):
    def coord_neighbors_fun(coord):
        # x: left to right, y: bottom to top
        x, y = coord
        coord_dict = {'w': (x - 1, y),
                      'n': (x, y - 1),
                      'e': (x + 1, y),
                      's': (x, y + 1)}
        return coord_dict

    def coord_list_fun():
        return [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]

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
    n_values = 8
    # 0: ground -> only on ground
    # 1: grass -> only on ground
    # 2: mushroom -> only on ground
    # 3: ? -> only on air
    # 4: air -> only on grass / mushroom / tree-high
    # 5: tree-low -> only on grass
    # 6: tree-mid
    # 7: tree-high

    # coordinates
    coord_list = coord_list_fun()
    print('coord_list', coord_list)

    # rules
    ruleset = DirectionRuleSet(n_values)
    weight_not_on_floor = lambda coord: 0 if coord[1] == map_y_size - 1 else 1
    weight_not_on_floor_alpha = lambda coord: 0 if coord[1] == map_y_size - 1 else alpha

    # ground
    context = {'pattern': {'n': 0, 's': 0}}
    value_const = 0
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 1, 's': 0}}
    value_const = 0
    ruleset.add(value_const, pattern_weight_fun, context)

    # grass
    context = {'pattern': {'n': 4, 's': 0}, 'weight': weight_not_on_floor}
    value_const = 1
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 2, 's': 0}, 'weight': weight_not_on_floor}
    value_const = 1
    ruleset.add(value_const, pattern_weight_fun, context)

    # mushroom
    context = {'pattern': {'n': 4, 's': 1}, 'weight': weight_not_on_floor}
    value_const = 2
    ruleset.add(value_const, pattern_weight_fun, context)

    # ?
    context = {'pattern': {'n': 4, 's': 4}, 'weight': weight_not_on_floor_alpha}
    value_const = 3
    ruleset.add(value_const, pattern_weight_fun, context)

    # air
    context = {'pattern': {'n': 4, 's': 4}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 4, 's': 1}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 4, 's': 2}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 4, 's': 3}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 4, 's': 7}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'n': 3, 's': 4}, 'weight': weight_not_on_floor}
    value_const = 4
    ruleset.add(value_const, pattern_weight_fun, context)

    # tree
    context = {'pattern': {'s': 1, 'n': 6}, 'weight': weight_not_on_floor}
    value_const = 5
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'s': 5, 'n': 5}, 'weight': weight_not_on_floor}
    value_const = 6
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'s': 6, 'n': 4}, 'weight': weight_not_on_floor}
    value_const = 7
    ruleset.add(value_const, pattern_weight_fun, context)

    #
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
        quantum_runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel,
                                                  instance=instance,
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
    process_result(result, f'{name}-{engine}-x{map_x_size}-y{map_y_size}-')


# args
parser = argparse.ArgumentParser()
parser.add_argument('--map_x_size', type=int, default=4, help='map x size')
parser.add_argument('--map_y_size', type=int, default=6, help='map y size')
parser.add_argument('--n_chunks', type=int, default=6, help='map chunks (only for H)')
parser.add_argument('--alpha', type=float, default=.1, help='? weight')
parser.add_argument('--backend-name', type=str, default=None,
                    help='IBMQ backend name, None for local simulator (default: None)')
parser.add_argument('--channel', type=str, default=None, help='IBMQ runtime service channel (default: None)')
parser.add_argument('--instance', type=str, default=None, help='IBMQ runtime service instance (default: None)')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1, help='number of shots for non-statevector and CWFC')
parser.add_argument('--engine', type=str, choices=['C', 'Q', 'H'], default='H', help='WFC engine: C, Q or H (default)')
parser.add_argument('--name', type=str, default='platformer', help='result filename')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Two-dimensional tile arrangement based on game-like tiles. Uses 8 different tiles. Art source: https://opengameart.org/content/2d-four-seasons-platformer-tileset-16x16
    """
    run(map_x_size=args.map_x_size, map_y_size=args.map_y_size, n_chunks=args.n_chunks, alpha=args.alpha,
        backend_name=args.backend_name, channel=args.channel, instance=args.instance, use_sv=args.use_sv,
        shots=args.shots, engine=args.engine, name=args.name)
