import argparse
import json
import os
from datetime import datetime
from functools import partial

from tqdm import tqdm

from qwfc._version import __version__
from qwfc.common import DirectionRuleSet
from qwfc.runner import ClassicalRunnerDefault, HybridRunnerDefault
from tests.example_utils import run_wfc, configure_quantum_runner, pattern_weight_fun, compose_image


def draw_image(mapped_coords):
    image_path = 'lines-tiles.png'
    sprite_map = {0: (0, 0),  # left-right
                  1: (0, 1),  # up-down
                  2: (0, 2),  # up-right
                  3: (0, 3),  # left-down
                  4: (0, 4),  # up-right
                  5: (0, 5),  # down-right
                  6: (0, 6),  # x
                  7: (0, 7)  # empty
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


def run(map_x_size, map_y_size, chunk_x_size=None, chunk_y_size=None, alpha=0, backend_name=None, channel=None,
        instance=None, use_sv=False, shots=1, engine='Q', name=''):
    if chunk_x_size is None:
        chunk_x_size = map_x_size
    if chunk_y_size is None:
        chunk_x_size = map_y_size
    chunk_x_shift = chunk_x_size
    chunk_y_shift = chunk_y_size

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
        assert ((map_x_size - chunk_x_size) // chunk_x_shift) * chunk_x_shift + chunk_x_size == map_x_size
        assert ((map_y_size - chunk_y_size) // chunk_y_shift) * chunk_y_shift + chunk_y_size == map_y_size

        # x: left to right, y: bottom to top
        x_pos = 0
        x_steps = []
        for x in range(map_x_size):
            x_start = x_pos
            x_pos = x_start + chunk_x_shift
            x_steps.append((x_start, x_start + chunk_x_size - 1))
            if x_start + chunk_x_size - 1 == map_x_size - 1:
                break
            elif x_start + chunk_x_size - 1 > map_x_size - 1:
                raise ValueError
        y_pos = 0
        y_steps = []
        for y in range(map_y_size):
            y_start = y_pos
            y_pos = y_start + chunk_y_shift
            y_steps.append((y_start, y_start + chunk_y_size - 1))
            if y_start + chunk_y_size - 1 == map_y_size - 1:
                break
            elif y_start + chunk_y_size - 1 > map_y_size - 1:
                raise ValueError
        y_steps = [(map_y_size - y_step[0] - 1, map_y_size - y_step[1] - 1) for y_step in y_steps]
        #
        for y_start, y_stop in y_steps:
            for x_start, x_stop in x_steps:
                coord_list_seg = [(x, y) for y in range(y_start, y_stop - 1, -1) for x in range(x_start, x_stop + 1) if
                                  (x, y) in coord_list]
                yield coord_list_seg, None

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
    # 0 left-right
    # 1 up-down
    # 2 left-up
    # 3 left-down
    # 4 up-right
    # 5 down-right
    # 6 plus
    # 7 empty

    # coordinates
    coord_list = coord_list_fun()

    # rules
    ruleset = DirectionRuleSet(n_values)
    connectivity = {0: ['e', 'w'],
                    1: ['n', 's'],
                    2: ['n', 'w'],
                    3: ['w', 's'],
                    4: ['n', 'e'],
                    5: ['s', 'e'],
                    6: ['n', 'e', 's', 'w'],
                    7: []}
    weight_fun_empty = lambda coord: ((coord[0] * alpha) if alpha > 0 else 1)
    for e in range(n_values):
        for n in range(n_values):
            for w in range(n_values):
                for s in range(n_values):
                    connections = []
                    if 'w' in connectivity[e]:
                        connections.append('e')
                    if 's' in connectivity[n]:
                        connections.append('n')
                    if 'e' in connectivity[w]:
                        connections.append('w')
                    if 'n' in connectivity[s]:
                        connections.append('s')
                    for v in range(n_values):
                        if sorted(connectivity[v]) == sorted(connections):
                            if v == 7:
                                weight = weight_fun_empty
                            else:
                                weight = 1
                            context = {'pattern': {'e': e, 'n': n, 'w': w, 's': s}, 'weight': weight}
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
        quantum_runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel,
                                                  instance=instance,
                                                  shots=shots, check_feasibility=False, add_barriers=False)
        runner = HybridRunnerDefault(quantum_runner=quantum_runner)
        total_steps = (((map_x_size - chunk_x_size) // chunk_x_shift) + 1) * (
                    ((map_y_size - chunk_y_size) // chunk_y_shift) + 1)
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
parser.add_argument('--map-x-size', type=int, default=4, help='map x size')
parser.add_argument('--map-y-size', type=int, default=4, help='map y size')
parser.add_argument('--chunk-x-size', type=int, default=2, help='map chunk x size (for hybrid)')
parser.add_argument('--chunk-y-size', type=int, default=2, help='map chunk y size (for hybrid)')
parser.add_argument('--alpha', type=float, default=0, help='density > 0, 0 to disable (default)')
parser.add_argument('--backend-name', type=str, default=None,
                    help='IBMQ backend name, None for local simulator (default: None)')
parser.add_argument('--channel', type=str, default=None, help='IBMQ runtime service channel (default: None)')
parser.add_argument('--instance', type=str, default=None, help='IBMQ runtime service instance (default: None)')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1, help='number of shots for non-statevector and CWFC')
parser.add_argument('--engine', type=str, choices=['C', 'Q'], default='H', help='WFC engine: C, Q or H (default)')
parser.add_argument('--name', type=str, default='lines', help='result filename')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Two-dimensional line pattern with specific connection constraints such that the line is never broken. The line density depends on the horizontal position of the tile and can be controlled with the parameter --alpha (0: uniform).
    """
    run(map_x_size=args.map_x_size, map_y_size=args.map_y_size, chunk_x_size=args.chunk_x_size,
        chunk_y_size=args.chunk_y_size, alpha=args.alpha, backend_name=args.backend_name, channel=args.channel,
        instance=args.instance, use_sv=args.use_sv, shots=args.shots, engine=args.engine, name=args.name)
