import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from itertools import product
import json
from qwfc.common import DirectionRuleSet
from qwfc.runner import ClassicalRunnerDefault, HybridRunnerDefault
from tests.example_utils import run_wfc, configure_quantum_runner, pattern_weight_fun, compose_image, fig2img
from qwfc._version import __version__

def draw_image(mapped_coords, map_size, map_height):
    voxelarray = np.zeros((map_size,map_size,map_height))
    facecolors = np.empty((map_size, map_size, map_height), dtype=object)
    for k, v in mapped_coords.items():
        if v == 1:
            voxelarray[k] = 1
            facecolors[k] = '#999999'
        else:
            voxelarray[k] = 0
            facecolors[k] = '#FFFFFF'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=facecolors, edgecolor='k', alpha=1)
    plt.axis('off')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    img = fig2img(fig,bbox_inches='tight',dpi=300)
    plt.close()
    return img

def process_result(result, map_size, map_height, prefix=''):
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
        draw_image(mapped_coords, map_size, map_height).save(filename)
    #
    filename = f'results/{prefix}data.json'
    with open(filename, 'w') as fh:
        data = dict(pc = {str(key): (float(p), {str(c): int(v) for c,v in mapped_coords.items()}, bool(f) if f is not None else None) for (key, (p, mapped_coords, f)) in pc.items()}, version=__version__)
        json.dump(data, fh)


def run(map_size, map_height, n_chunks=1, backend_name=None, channel=None, instance=None, use_sv=False, shots=1, engine='Q', name=''):

    def coord_neighbors_fun(coord):
        coord_dict = {}
        for k in [-1, +1]:
            d=2
            coord_dict[f'z{k:+}'] = tuple([coord[d_]+k if d_==d else coord[d_] for d_ in range(3)])
        return coord_dict

    def coord_list_fun():
        return [(x,y,z) for (x,y) in product(range(map_size), repeat=2) for z in range(map_height)]

    def chunk_map_fun(parsed_counts):
        # use most probable
        for _, (_, coord_map, feasibility) in parsed_counts.items():
            if feasibility is None or feasibility:
                return coord_map
        return None

    def chunk_iter_fun(coord_list):
        for coord_list_seg in np.array_split(coord_list, n_chunks):
            yield [tuple(coord) for coord in coord_list_seg], None

    # values
    n_values = 2
    # 0: black
    # 1: white

    # coordinates
    coord_list = coord_list_fun()
    print('coord_list', coord_list)

    # rules
    ruleset = DirectionRuleSet(n_values)

    context = {'pattern': {'z-1': 1, 'z+1': 1}}
    value_const = 1
    ruleset.add(value_const, pattern_weight_fun, context)

    context = {'pattern': {'z-1': 1, 'z+1': 0}}
    value_const = 1
    ruleset.add(value_const, pattern_weight_fun, context)
    context = {'pattern': {'z-1': 1, 'z+1': 0}}
    value_const = 0
    ruleset.add(value_const, pattern_weight_fun, context)

    context = {'pattern': {'z-1': 0, 'z+1': 0}}
    value_const = 0
    ruleset.add(value_const, pattern_weight_fun, context)

    print(f'generated ruleset: {len(ruleset)} rules')

    # setup
    if engine == 'Q':
        # QWFC
        runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel, instance=instance, shots=shots, check_feasibility=False, add_barriers=False)
        run_kwargs = dict(coord_path_fun=None, coord_fixed=None, callback_fun=None)
    elif engine == 'C':
        # CWFC
        n_samples = shots
        runner = ClassicalRunnerDefault(n_samples=n_samples)
        run_kwargs = dict(coord_fixed=None, callback_fun=None)
    elif engine == 'H':
        # HWFC
        quantum_runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel,
                                                  instance=instance,
                                                  shots=shots, check_feasibility=False, add_barriers=False)
        runner = HybridRunnerDefault(quantum_runner=quantum_runner)
        run_kwargs = dict(chunk_map_fun=chunk_map_fun, chunk_iter_fun=chunk_iter_fun,
                          qwfc_callback_fun=None, hwfc_callback_fun=None)
    else:
        raise NotImplementedError

    # run
    result = run_wfc(runner, n_values, coord_list, coord_neighbors_fun, ruleset, **run_kwargs)
    process_result(result, map_size, map_height, f'{name}-{engine}-s{map_size}-h{map_height}-')

# args
parser = argparse.ArgumentParser()
parser.add_argument('--map-size', type=int, default=3, help='map dimension (x,y)')
parser.add_argument('--map-height', type=int, default=4, help='map height (z)')
parser.add_argument('--n-chunks', type=int, default=2, help='map chunks (only for H)')
parser.add_argument('--backend-name', type=str, default=None, help='IBMQ backend name, None for local simulator (default: None)')
parser.add_argument('--channel', type=str, default=None, help='IBMQ runtime service channel (default: None)')
parser.add_argument('--instance', type=str, default=None, help='IBMQ runtime service instance (default: None)')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1, help='number of shots for non-statevector and CWFC')
parser.add_argument('--engine', type=str, choices=['C', 'Q', 'H'], default='Q', help='WFC engine: C, Q (default) or H')
parser.add_argument('--name', type=str, default='voxel', help='result filename')
args = parser.parse_args()

if __name__ == '__main__':
    """
    3-dimensional voxel graphic similar to a skyline.
    """
    run(map_size=args.map_size, map_height=args.map_height, n_chunks=args.n_chunks, backend_name=args.backend_name, channel=args.channel, instance=args.instance, use_sv=args.use_sv, shots=args.shots, engine=args.engine, name=args.name)
    
