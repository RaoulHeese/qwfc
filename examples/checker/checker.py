import argparse
import sys, os
from qiskit import Aer

sys.path.insert(0, '../../src')
from qwfc import Map, MapSlidingWindow, CorrelationRuleSet
from runner import CircuitRunnerIBMQAer
sys.path.insert(0, '../')
from example_utils import coord_rules_fun_generator, compose_image


def process_output(parsed_counts, qc, prefix=''):
    def draw_map(mapped_coords):
        image_path = 'checker-tiles.png'
        sprite_map = {0: (0, 0),  # white
                      1: (1, 0),  # black
                      }
        background_tile = None
        sprite_size = 32
        mapped_coords_2d = {k if len(k) == 2 else (k[0], 0): v for k, v in mapped_coords.items()}
        return compose_image(mapped_coords_2d, image_path, sprite_map, sprite_size, background_tile)

    if not os.path.isdir('results'):
        os.mkdir('results')
    for key, (p, coord_map) in parsed_counts.items():
        filename = f'results/{prefix}{key}.png'
        print(f'{key}: p={p}, file={filename}')
        draw_map(coord_map).save(filename)
    if qc is not None:
        qc.draw(output='mpl', filename=f'results/{prefix}qc.png')


def run_1d(map_x_size, use_sv, shots, show_qc):
    def coord_neighbors_fun_1d(coord):
        # x: left to right
        x, = coord
        coord_dict = {'w': (x - 1,),
                      'e': (x + 1,)}
        return coord_dict

    def coord_path_fun_1d(coord_list):
        # x: left to right
        x_vals = [coord[0] for coord in coord_list]
        return ((x,) for x in range(min(x_vals), max(x_vals) + 1) if (x,) in coord_list)

    # input
    n_values = 2
    adj_order = ['e', 'w']
    rules = {(None, None): {0: .5, 1: .5},
             (None, 0): {1: 1},
             (None, 1): {0: 1},
             }
    coord_rules_fun_1d = coord_rules_fun_generator(adj_order, rules, lambda: CorrelationRuleSet(n_values))
    if use_sv:
        backend = Aer.get_backend('statevector_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')
    circuit_runner = CircuitRunnerIBMQAer(backend = backend, run_kwarg_dict = dict(shots=shots))
    coord_list_1d = [(x,) for x in range(map_x_size)]
    check_feasibility = False

    # run
    print(f'checker 1d: run full map generation (circuit_runner={circuit_runner})...')
    m = Map(n_values, coord_list_1d, coord_neighbors_fun_1d, check_feasibility)
    m.run(coord_rules_fun_1d, coord_path_fun_1d, circuit_runner, callback_fun=None)

    # output
    process_output(m.pc, m.qc if show_qc else None, '1d-')


def run_2d(map_x_size, map_y_size, use_sv, shots, show_qc):
    def coord_neighbors_fun_2d(coord):
        # x: left to right, y: bottom to top
        x, y = coord
        coord_dict = {'w': (x - 1, y),
                      'n': (x, y - 1),
                      'e': (x + 1, y),
                      's': (x, y + 1)}
        return coord_dict

    def coord_path_fun_2d(coord_list):
        # x: left to right, y: bottom to top
        x_vals = [coord[0] for coord in coord_list]
        y_vals = [coord[1] for coord in coord_list]
        return ((x, y) for y in range(max(y_vals), min(y_vals) - 1, -1) for x in range(min(x_vals), max(x_vals) + 1) if
                (x, y) in coord_list)

    # input
    n_values = 2
    adj_order = ['n', 'e', 's', 'w']
    rules = {(None, None, None, None): {0: .5, 1: .5},
             (None, None, None, 0): {1: 1},
             (None, None, None, 1): {0: 1},
             (None, None, 0, None): {1: 1},
             (None, None, 1, None): {0: 1},
             }
    coord_rules_fun_2d = coord_rules_fun_generator(adj_order, rules, lambda: CorrelationRuleSet(n_values))
    if use_sv:
        backend = Aer.get_backend('statevector_simulator')
    else:
        backend = Aer.get_backend('qasm_simulator')
    circuit_runner = CircuitRunnerIBMQAer(backend=backend, run_kwarg_dict=dict(shots=shots))
    coord_list_2d = [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]
    check_feasibility = False

    # run
    print(f'checker 2d: run full map generation (circuit_runner={circuit_runner})...')
    m = Map(n_values, coord_list_2d, coord_neighbors_fun_2d, check_feasibility)
    m.run(coord_rules_fun_2d, coord_path_fun_2d, circuit_runner, callback_fun=None)

    # output+
    print('finished:')
    process_output(m.pc, m.qc if show_qc else None, '2d-')


def run(dim, max_x_size, map_y_size, use_sv, shots, show_qc):
    if dim == 1:
        run_1d(max_x_size, use_sv, shots, show_qc)
    elif dim == 2:
        run_2d(max_x_size, map_y_size, use_sv, shots, show_qc)
    else:
        raise NotImplementedError


# args
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dim', type=int, choices=[1, 2], default=1, help='map dimension')
parser.add_argument('-x', '--x-size', type=int, default=3, help='map width')
parser.add_argument('-y', '--y-size', type=int, default=3, help='map height')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=1000, help='number of shots for non-statevector')
parser.add_argument('--qc', dest='show_qc', action='store_true', help='show generated quantum circuit')
parser.set_defaults(show_qc=False)
args = parser.parse_args()

if __name__ == '__main__':
    """
    One- or two-dimensional checkerboard with two different kinds of tiles: black tiles and white tiles. Neighboring tiles must be of a different color. The tiles are traversed from left to right and bottom to top.
    """

    run(args.dim, args.x_size, args.y_size, args.use_sv, args.shots, args.show_qc)
