import sys, os
import argparse
from tqdm import tqdm
from qiskit import Aer

sys.path.insert(0, '../../src')
from qwfc import Map, MapSlidingWindow, CorrelationRuleSet
from runner import CircuitRunnerIBMQAer
sys.path.insert(0, '../')
from example_utils import coord_rules_fun_generator, compose_image

n_values = 8


# 0 left-right
# 1 up-down
# 2 left-up
# 3 left-down
# 4 up-right
# 5 down-right
# 6 x
# 7 empty


def coord_neighbors_fun(coord):
    # x: left to right, y: bottom to top
    x, y = coord
    coord_dict = {'w': (x - 1, y),
                  'n': (x, y - 1),
                  'e': (x + 1, y),
                  's': (x, y + 1)}
    return coord_dict


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


def generate_coord_rules_fun(empty_proba_fun):
    adj_order = ['n', 'e', 's', 'w']
    rules = {  # lvl 0
        (None, None, None, None): {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: empty_proba_fun},
        # lvl 1
        (None, None, None, 0): {0: 1, 2: 1, 3: 1},
        (None, None, None, 1): {1: 1, 4: 1, 5: 1, 7: empty_proba_fun},
        (None, None, None, 2): {1: 1, 4: 1, 5: 1, 7: empty_proba_fun},
        (None, None, None, 3): {1: 1, 4: 1, 5: 1, 7: empty_proba_fun},
        (None, None, None, 4): {0: 1, 2: 1, 3: 1},
        (None, None, None, 5): {0: 1, 2: 1, 3: 1},
        (None, None, None, 6): {0: 1, 2: 1, 3: 1},
        (None, None, None, 7): {1: 1, 4: 1, 5: 1, 7: empty_proba_fun},
        #
        (None, None, 0, None): {0: 1, 2: 1, 4: 1, 7: empty_proba_fun},
        (None, None, 1, None): {1: 1, 3: 1, 5: 1, 6: 1},
        (None, None, 2, None): {1: 1, 3: 1, 5: 1, 6: 1},
        (None, None, 3, None): {0: 1, 2: 1, 4: 1, 7: empty_proba_fun},
        (None, None, 4, None): {1: 1, 3: 1, 5: 1, 6: 1},
        (None, None, 5, None): {0: 1, 2: 1, 4: 1, 7: empty_proba_fun},
        (None, None, 6, None): {1: 1, 3: 1, 5: 1, 6: 1},
        (None, None, 7, None): {0: 1, 2: 1, 4: 1, 7: empty_proba_fun},
        # lvl 2
        (None, None, 0, 0): {0: 1, 2: 1},
        (None, None, 1, 0): {3: 1, 6: 1},
        (None, None, 2, 0): {3: 1, 6: 1},
        (None, None, 3, 0): {0: 1, 2: 1},
        (None, None, 4, 0): {3: 1, 6: 1},
        (None, None, 5, 0): {0: 1, 2: 1},
        (None, None, 6, 0): {3: 1, 6: 1},
        (None, None, 7, 0): {0: 1, 2: 1},
        #
        (None, None, 0, 1): {4: 1, 7: empty_proba_fun},
        (None, None, 1, 1): {1: 1, 5: 1},
        (None, None, 2, 1): {1: 1, 5: 1},
        (None, None, 3, 1): {4: 1, 7: empty_proba_fun},
        (None, None, 4, 1): {1: 1, 5: 1},
        (None, None, 5, 1): {4: 1, 7: empty_proba_fun},
        (None, None, 6, 1): {1: 1, 5: 1},
        (None, None, 7, 1): {4: 1, 7: empty_proba_fun},
        #
        (None, None, 0, 2): {4: 1, 7: empty_proba_fun},
        (None, None, 1, 2): {1: 1, 5: 1},
        (None, None, 2, 2): {1: 1, 5: 1},
        (None, None, 3, 2): {4: 1, 7: empty_proba_fun},
        (None, None, 4, 2): {1: 1, 5: 1},
        (None, None, 5, 2): {4: 1, 7: empty_proba_fun},
        (None, None, 6, 2): {1: 1, 5: 1},
        (None, None, 7, 2): {4: 1, 7: empty_proba_fun},
        #
        (None, None, 0, 3): {4: 1, 7: empty_proba_fun},
        (None, None, 1, 3): {1: 1, 5: 1},
        (None, None, 2, 3): {1: 1, 5: 1},
        (None, None, 3, 3): {4: 1, 7: empty_proba_fun},
        (None, None, 4, 3): {1: 1, 5: 1},
        (None, None, 5, 3): {4: 1, 7: empty_proba_fun},
        (None, None, 6, 3): {1: 1, 5: 1},
        (None, None, 7, 3): {4: 1, 7: empty_proba_fun},
        #
        (None, None, 0, 4): {0: 1, 2: 1},
        (None, None, 1, 4): {3: 1, 6: 1},
        (None, None, 2, 4): {3: 1, 6: 1},
        (None, None, 3, 4): {0: 1, 2: 1},
        (None, None, 4, 4): {3: 1, 6: 1},
        (None, None, 5, 4): {0: 1, 2: 1},
        (None, None, 6, 4): {3: 1, 6: 1},
        (None, None, 7, 4): {0: 1, 2: 1},
        #
        (None, None, 0, 5): {0: 1, 2: 1},
        (None, None, 1, 5): {3: 1, 6: 1},
        (None, None, 2, 5): {3: 1, 6: 1},
        (None, None, 3, 5): {0: 1, 2: 1},
        (None, None, 4, 5): {3: 1, 6: 1},
        (None, None, 5, 5): {0: 1, 2: 1},
        (None, None, 6, 5): {3: 1, 6: 1},
        (None, None, 7, 5): {0: 1, 2: 1},
        #
        (None, None, 0, 6): {0: 1, 2: 1},
        (None, None, 1, 6): {3: 1, 6: 1},
        (None, None, 2, 6): {3: 1, 6: 1},
        (None, None, 3, 6): {0: 1, 2: 1},
        (None, None, 4, 6): {3: 1, 6: 1},
        (None, None, 5, 6): {0: 1, 2: 1},
        (None, None, 6, 6): {3: 1, 6: 1},
        (None, None, 7, 6): {0: 1, 2: 1},
        #
        (None, None, 0, 7): {4: 1, 7: empty_proba_fun},
        (None, None, 1, 7): {1: 1, 5: 1},
        (None, None, 2, 7): {1: 1, 5: 1},
        (None, None, 3, 7): {4: 1, 7: empty_proba_fun},
        (None, None, 4, 7): {1: 1, 5: 1},
        (None, None, 5, 7): {4: 1, 7: empty_proba_fun},
        (None, None, 6, 7): {1: 1, 5: 1},
        (None, None, 7, 7): {4: 1, 7: empty_proba_fun},
    }
    return coord_rules_fun_generator(adj_order, rules, lambda: CorrelationRuleSet(n_values))


def process_output(mapped_coords):
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
    if not os.path.isdir('results'):
        os.mkdir('results')
    filename = f'results/lines.png'
    compose_image(mapped_coords, image_path, sprite_map, sprite_size, background_tile).save(filename)
    print(f'saved to {filename}')


def run(map_x_size, map_y_size, segment_x_size, segment_y_size, segment_x_shift, segment_y_shift, alpha):
    def callback_fun(msw, idx, map_segment, segment_mapped_coords, pbar):
        pbar.set_postfix({'segment': idx})
        pbar.update(1)

    def segment_callback_fun(m, idx, coord, pbar):
        pbar.set_postfix({'idx': idx}, {'coord': coord})

    # input
    shots = 1
    circuit_runner = CircuitRunnerIBMQAer(backend=Aer.get_backend('qasm_simulator'), run_kwarg_dict=dict(shots=shots))
    coord_rules_fun = generate_coord_rules_fun((lambda coords: coords[0] * alpha) if alpha > 0 else 1)
    coord_list = [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]
    check_feasibility = False

    # run
    print(f'lines: run segmented map generation (circuit_runner={circuit_runner})...')
    msw = MapSlidingWindow(n_values, coord_list, coord_neighbors_fun, check_feasibility)
    segment_iter_fun = lambda coord_list: segment_iter_fun_wrapper(coord_list, map_x_size, map_y_size, segment_x_size,
                                                                   segment_y_size, segment_x_shift, segment_y_shift)
    total_steps = (((map_x_size - segment_x_size) // segment_x_shift) + 1) * (
            ((map_y_size - segment_y_size) // segment_y_shift) + 1)
    with tqdm(total=total_steps, desc='run') as pbar:
        msw.run(segment_map_fun, segment_iter_fun, coord_rules_fun, circuit_runner=circuit_runner,
                segment_callback_fun=lambda m, idx, coord: segment_callback_fun(m, idx, coord, pbar),
                callback_fun=lambda msw, idx, map_segment, segment_mapped_coords: callback_fun(msw, idx, map_segment,
                                                                                               segment_mapped_coords,
                                                                                               pbar))

    # output
    print('finished:')
    process_output(msw.mapped_coords)


# args
parser = argparse.ArgumentParser()
parser.add_argument('-x', '--x-size', type=int, default=4, help='map width')
parser.add_argument('-y', '--y-size', type=int, default=4, help='map height')
parser.add_argument('--segment-x-size', type=int, default=2, help='map segment width')
parser.add_argument('--segment-y-size', type=int, default=2, help='map segment height')
parser.add_argument('--segment-x-shift', type=int, default=2, help='map segment horizontal shift')
parser.add_argument('--segment-y-shift', type=int, default=2, help='map segment vertical shift')
parser.add_argument('-a', '--alpha', type=float, default=0, help='density reduction factor > 0, 0 to disable')

args = parser.parse_args()

if __name__ == '__main__':
    """
    Two-dimensional line pattern with specific connection constraints such that the line is never broken. The line density depends on the horizontal position of the tile and can be controlled with the parameter --alpha (0: uniform). The tiles are traversed from left to right and bottom to top.
    """

    run(args.x_size, args.y_size, args.segment_x_size, args.segment_y_size, args.segment_x_shift, args.segment_y_shift,
        args.alpha)
