import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from qiskit import Aer

sys.path.insert(0, '../../src')
from qwfc import Map, MapSlidingWindow, CorrelationRuleSet
from runner import CircuitRunnerIBMQAer
sys.path.insert(0, '../')
from example_utils import coord_rules_fun_generator, compose_image


def coord_neighbors_fun(coord):
    # x: left to right, y: bottom to top
    x, y = coord
    coord_dict = {'w': (x - 1, y),
                  'n': (x, y - 1),
                  'e': (x + 1, y),
                  's': (x, y + 1)}
    return coord_dict


def draw_map(mapped_coords, big_flag):
    image_path = 'platformer-tiles.png'
    if not big_flag:
        sprite_map = {0: (1, 0),  # ground
                      1: (9, 4),  # air
                      2: (9, 3),  # mushroom
                      3: (3, 2)  # block
                      }
        background_tile = (0, 11)
    else:
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


def segment_map_fun(parsed_counts):
    # use most probable
    for _, (_, coord_map) in parsed_counts.items():
        return coord_map


def segment_coord_path_fun(coord_list):
    # x: left to right, y: bottom to top
    x_vals = [coord[0] for coord in coord_list]
    y_vals = [coord[1] for coord in coord_list]
    return ((x, y) for y in range(max(y_vals), min(y_vals) - 1, -1) for x in range(min(x_vals), max(x_vals) + 1) if
            (x, y) in coord_list)


def segment_iter_fun_wrapper(coord_list, map_x_size, map_y_size, segment_x_size, segment_y_size, segment_x_shift,
                             segment_y_shift):
    assert ((map_x_size - segment_x_size) // segment_x_shift) * segment_x_shift + segment_x_size == map_x_size
    assert ((map_y_size - segment_y_size) // segment_y_shift) * segment_y_shift + segment_y_size == map_y_size

    # x: left to right, y: bottom to top
    x_pos = 0
    x_steps = []
    for x in range(map_x_size):
        x_start = x_pos
        x_pos = x_start + segment_x_shift
        x_steps.append((x_start, x_start + segment_x_size - 1))
        if x_start + segment_x_size - 1 == map_x_size - 1:
            break
        elif x_start + segment_x_size - 1 > map_x_size - 1:
            raise ValueError
    y_pos = 0
    y_steps = []
    for y in range(map_y_size):
        y_start = y_pos
        y_pos = y_start + segment_y_shift
        y_steps.append((y_start, y_start + segment_y_size - 1))
        if y_start + segment_y_size - 1 == map_y_size - 1:
            break
        elif y_start + segment_y_size - 1 > map_y_size - 1:
            raise ValueError
    y_steps = [(map_y_size - y_step[0] - 1, map_y_size - y_step[1] - 1) for y_step in y_steps]
    #
    for y_start, y_stop in y_steps:
        for x_start, x_stop in x_steps:
            coord_list_seg = [(x, y) for y in range(y_start, y_stop - 1, -1) for x in range(x_start, x_stop + 1) if
                              (x, y) in coord_list]
            yield coord_list_seg, segment_coord_path_fun


def generate_coord_rules_fun(big_flag):
    # setup
    adj_order = ['n', 'e', 's', 'w']
    if not big_flag:
        n_values = 4
        # 0: ground
        # 1: air
        # 2: mushroom
        # 3: block
        rules = {(None, None, None, None): {0: 1},  # default: ground
                 (None, None, 0, None): {0: .5, 1: .3, 2: .2},
                 # second+ floor with lower ground: ground or air or mushroom
                 (None, None, 1, None): {1: .9, 3: .1},  # third+ floor with lower air: air or block
                 (None, None, 2, None): {1: 1},  # third+ floor with lower mushroom: air
                 (None, None, 3, None): {1: 1},  # third+ floor with lower block: air
                 # block has only adj air:
                 (None, None, 1, 0): {1: 1},  # third+ floor with lower air and neighbor ground: air
                 (None, None, 1, 2): {1: 1},  # third+ floor with lower air and neighbor mushroom: air
                 (None, None, 1, 3): {1: 1},  # third+ floor with lower air and neighbor block: air
                 (None, None, 0, 3): {1: 1},  # second+ floor with lower ground and neighbor block: air
                 (None, None, 1, 3): {1: 1},  # second+ floor with lower air and neighbor block: air
                 (None, None, 2, 3): {1: 1},  # second+ floor with lower mushroom and neighbor block: air
                 (None, None, 3, 3): {1: 1},  # second+ floor with lower block and neighbor block: air
                 }
    else:
        n_values = 8
        # 0: ground
        # 1: grass
        # 2: mushroom -> only on ground
        # 3: ?
        # 4: air
        # 5: tree-low -> only on grass
        # 6: tree-mid
        # 7: tree-high
        rules = {  # floor
            (None, None, None, None): {0: .75, 1: .25},
            # mushroom
            (None, None, 0, None): {0: .5, 1: .1, 2: .1, 4: .3},
            # air
            (None, None, 4, None): {4: .9, 3: .1},
            (None, None, 3, None): {4: 1},
            (None, None, 2, None): {4: 1},
            (None, None, 7, None): {4: 1},
            # tree
            (None, None, 1, None): {4: .5, 5: .5},
            (None, None, 5, None): {6: 1},
            (None, None, 6, None): {7: 1},
        }
    return n_values, coord_rules_fun_generator(adj_order, rules, lambda: CorrelationRuleSet(n_values))


def process_output_full(parsed_counts, big_flag):
    if not os.path.isdir('results'):
        os.mkdir('results')
    prefix = 'big-' if big_flag else ''
    img_mean = None
    for key, (p, coord_map) in parsed_counts.items():
        filename = f'results/{prefix}{key}.png'
        print(f'{key}: p={p}, file={filename}')
        img = draw_map(coord_map, big_flag)
        img.save(filename)
        img_array = np.array(img).astype(float) * p
        if img_mean is None:
            img_mean = img_array
        else:
            img_mean += img_array
    img_mean = np.clip(np.round(img_mean), 0, 255).astype(np.uint8)
    Image.fromarray(img_mean).save(f'results/{prefix}mean.png')


def process_output_segmented(mapped_coords, big_flag):
    if not os.path.isdir('results'):
        os.mkdir('results')
    prefix = 'big-' if big_flag else ''
    filename = f'results/{prefix}map.png'
    draw_map(mapped_coords, big_flag).save(filename)
    print(f'saved to {filename}')


def run_full(map_x_size, map_y_size, big_flag, use_sv, shots):
    # input
    if use_sv:
        backend = Aer.get_backend('statevector_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')
    circuit_runner = CircuitRunnerIBMQAer(backend=backend, run_kwarg_dict=dict(shots=shots))
    n_values, coord_rules_fun = generate_coord_rules_fun(big_flag)
    coord_list = [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]
    check_feasibility = False

    # run
    print(f'platformer: run full map generation (circuit_runner={circuit_runner})...')
    m = Map(n_values, coord_list, coord_neighbors_fun, check_feasibility)
    m.run(coord_rules_fun, segment_coord_path_fun, circuit_runner=circuit_runner, callback_fun=None)

    # output
    print('finished:')
    process_output_full(m.pc, big_flag)


def run_segmented(map_x_size, map_y_size, segment_x_size, segment_y_size, segment_x_shift, segment_y_shift, big_flag):
    def callback_fun(msw, idx, map_segment, segment_mapped_coords, pbar):
        pbar.set_postfix({'segment': idx})
        pbar.update(1)

    def segment_callback_fun(m, idx, coord, pbar):
        pbar.set_postfix({'idx': idx}, {'coord': coord})

    # input
    shots = 1
    circuit_runner = CircuitRunnerIBMQAer(backend=Aer.get_backend('qasm_simulator'), run_kwarg_dict=dict(shots=shots))
    n_values, coord_rules_fun = generate_coord_rules_fun(big_flag)
    coord_list = [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]
    feasibility = False

    # run
    print(f'platformer: run segmented map generation (circuit_runner={circuit_runner})...')
    msw = MapSlidingWindow(n_values, coord_list, coord_neighbors_fun, feasibility)
    segment_iter_fun = lambda coord_list: segment_iter_fun_wrapper(coord_list, map_x_size, map_y_size, segment_x_size,
                                                                   segment_y_size, segment_x_shift, segment_y_shift)
    total_steps = (((map_x_size - segment_x_size) // segment_x_shift) + 1) * (
            ((map_y_size - segment_y_size) // segment_y_shift) + 1)
    with tqdm(total=total_steps) as pbar:
        msw.run(segment_map_fun, segment_iter_fun, coord_rules_fun, circuit_runner=circuit_runner,
                segment_callback_fun=lambda m, idx, coord: segment_callback_fun(m, idx, coord, pbar),
                callback_fun=lambda msw, idx, map_segment, segment_mapped_coords: callback_fun(msw, idx, map_segment,
                                                                                               segment_mapped_coords,
                                                                                               pbar))

    # output
    print('finished:')
    process_output_segmented(msw.mapped_coords, big_flag)


def run(map_x_size, map_y_size, segment_x_size, segment_y_size, big_flag, use_sv, shots):
    if segment_x_size == 0 or segment_y_size == 0:
        run_full(map_x_size, map_y_size, big_flag, use_sv, shots)
    else:
        segment_x_shift = segment_x_size
        segment_y_shift = segment_y_size
        run_segmented(map_x_size, map_y_size, segment_x_size, segment_y_size, segment_x_shift, segment_y_shift,
                      big_flag)


# args
parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x-size', type=int, default=8, help='map width')
parser.add_argument('-y', '--y-size', type=int, default=8, help='map height')
parser.add_argument('--segment-x-size', type=int, default=2, help='map segment width (0 to disable segments)')
parser.add_argument('--segment-y-size', type=int, default=2, help='map segment height (0 to disable segments)')
parser.add_argument('--big', dest='big_flag', action='store_true', help='use big tileset')
parser.set_defaults(big_flag=False)
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator (only for full map)')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1000, help='number of shots for non-statevector (only for full map)')
args = parser.parse_args()

if __name__ == '__main__':
    """
    Two-dimensional tile arrangement based on game-like tiles. Uses 4 different types of tiles (or 8 with the argument --big). The tiles are traversed from left to right and bottom to top.
    """

    run(args.x_size, args.y_size, args.segment_x_size, args.segment_y_size, args.big_flag, args.use_sv, args.shots)
