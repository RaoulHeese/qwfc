from qwfc.common import DirectionRuleSet
from qwfc.classical import CWFC
from qwfc.quantum import QWFC
from qwfc.runner import ClassicalRunnerDefault, QuantumRunnerIBMQAer, HybridRunnerDefault
from qwfc.hybrid import HWFC
from qiskit import Aer

n_values = 2

def coord_neighbors_fun(coord):
    # x: left to right, y: bottom to top
    x, y = coord
    coord_dict = {'w': (x - 1, y),
                  'n': (x, y - 1),
                  'e': (x + 1, y),
                  's': (x, y + 1)}
    return coord_dict

map_x_size = 4
map_y_size = 4
coord_list = [(x, y) for y in range(map_y_size - 1, -1, -1) for x in range(map_x_size)]
print('coord_list', list(coord_list))

def draw_map(mapped_coords):
    s1, s2 = '', ''
    for coord in coord_list:
        x, y = coord
        value = mapped_coords[coord]
        ind = list(mapped_coords).index(coord)
        s1 += ' ' if value == 0 else 'X'
        s2 += f'{ind:2d} '
        if x > 0 and x % (map_x_size-1) == 0:
            s1 += '\n'
            s2 += '\n'
    print('-'*map_x_size)
    print(s1[:-1])
    print('-'*map_x_size)
    print(s2[:-1])
    print('-'*map_x_size)

ruleset = DirectionRuleSet(n_values)

def pattern_weight_fun(coord, coord_adj, coord_adj_offmap, mapped_coords, context):
    if context.get('only_initial', False) and len(mapped_coords) > 0:
        return 0
    for n_key, value in context['pattern'].items():
        if value is None: # ignore
            continue
        if value == 'any': # any (defined or undefined)
            if n_key in coord_adj_offmap:
                return 0
            else:
                continue
        elif value == 'none': # nothing or undefined
            if n_key in coord_adj_offmap:
                continue
            else:
                for mapped_coord, mapped_value in mapped_coords.items():
                    if n_key in coord_adj and coord_adj[n_key] == mapped_coord:
                        return 0
        else:
            # forall
            for mapped_coord, mapped_value in mapped_coords.items():
                if n_key in coord_adj and coord_adj[n_key] == mapped_coord and mapped_value != value:
                    return 0
            # exists
            if context.get('must_exist', False):
                if sum([1 for mapped_coord, mapped_value in mapped_coords.items() if n_key in coord_adj and coord_adj[n_key] == mapped_coord and mapped_value != value]) == 0:
                    return 0
    if callable(context['weight']):
        weight = context['weight'](coord)
    else:
        weight = context['weight']
    return weight




#context = {'pattern': {}, 'weight': 1, 'only_initial': True}
#value_fun = lambda coord: 0
#ruleset.add(value_fun, pattern_weight_fun, context)
#context = {'pattern': {}, 'weight': 1, 'only_initial': True}
#value_fun = lambda coord: 1
#ruleset.add(value_fun, pattern_weight_fun, context)

context = {'pattern': {'n': 1, 'e': 1, 's': 1, 'w': 1}, 'weight': 1}
value_const = 0
ruleset.add(value_const, pattern_weight_fun, context)

context = {'pattern': {'n': 0, 'e': 0, 's': 0, 'w': 0}, 'weight': 1}
value_const = 1
ruleset.add(value_const, pattern_weight_fun, context)

# classical

print('\n\nCMAP\n\n')
classical_runner = ClassicalRunnerDefault(1)
cmap = CWFC(n_values, coord_list, coord_neighbors_fun)
pc = cmap.run(ruleset, classical_runner, coord_fixed = None)

print('cmap')
for key, (p, mc, f) in pc.items():
    print(key, ':', p, f)
    draw_map(mc)

# quantum

print('\n\nQMAP\n\n')

def coord_path_fun(coord_list):
    # x: left to right, y: bottom to top
    x_vals = [coord[0] for coord in coord_list]
    y_vals = [coord[1] for coord in coord_list]
    return ((x, y) for y in range(max(y_vals), min(y_vals) - 1, -1) for x in range(min(x_vals), max(x_vals) + 1) if
            (x, y) in coord_list)

check_feasibility = False
quantum_runner = QuantumRunnerIBMQAer(Aer.get_backend('statevector_simulator'),check_feasibility= False, add_barriers= True, add_measurement = False)
qmap = QWFC(n_values, coord_list, coord_neighbors_fun)
qmap.run(ruleset, quantum_runner, coord_path_fun, coord_fixed = None, callback_fun = None)

print('qmap')
for bitstring, (p, mapped_coords, f) in qmap.pc.items():
    print(bitstring, ':', p, f)
    draw_map(mapped_coords)

#print(qmap.qc.draw())

# hybrid

print('\n\nHMAP\n\n')

def chunk_map_fun(parsed_counts):
    # use most probable
    for _, (_, coord_map, feasibility) in parsed_counts.items():
        if feasibility is None or feasibility:
            return coord_map
        return None

def chunk_iter_fun_wrapper(coord_list, map_x_size, map_y_size, segment_x_size, segment_y_size, segment_x_shift,
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
            yield coord_list_seg, coord_path_fun

segment_x_size = 2
segment_y_size = 2
segment_x_shift = 2
segment_y_shift = 2
chunk_iter_fun = lambda coord_list: chunk_iter_fun_wrapper(coord_list, map_x_size, map_y_size, segment_x_size,
                                                                   segment_y_size, segment_x_shift, segment_y_shift)

hybrid_runner = HybridRunnerDefault(quantum_runner)

hmap = HWFC(n_values, coord_list, coord_neighbors_fun)

hmap.run(ruleset, hybrid_runner, chunk_map_fun, chunk_iter_fun, chunk_callback_fun=None, callback_fun=None)
print('hmap')
draw_map(hmap.mapped_coords)
print(hmap.mapped_coords)
