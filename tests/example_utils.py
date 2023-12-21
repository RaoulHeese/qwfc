import io

from PIL import Image
from qiskit import Aer

from qwfc.classical import CWFC
from qwfc.hybrid import HWFC
from qwfc.quantum import QWFC
from qwfc.runner import QuantumRunnerInterface, ClassicalRunnerInterface, HybridRunnerInterface, QuantumRunnerIBMQAer, \
    QuantumRunnerIBMQRuntime


def run_wfc(runner, n_values, coord_list, coord_neighbors_fun, ruleset, **run_kwargs):
    result = dict()
    if isinstance(runner, QuantumRunnerInterface):
        quantum_runner = runner
        qwfc = QWFC(n_values, coord_list, coord_neighbors_fun)
        qwfc.run(ruleset, quantum_runner, run_kwargs.get('coord_path_fun', None), run_kwargs.get('coord_fixed', None),
                 run_kwargs.get('callback_fun', None))
        result.update(dict(pc=qwfc.pc, qc=qwfc.qc))
    elif isinstance(runner, ClassicalRunnerInterface):
        classical_runner = runner
        cwfc = CWFC(n_values, coord_list, coord_neighbors_fun)
        cwfc.run(ruleset, classical_runner, run_kwargs.get('coord_fixed', None), run_kwargs.get('callback_fun', None))
        result.update(dict(pc=cwfc.pc))
    elif isinstance(runner, HybridRunnerInterface):
        hybrid_runner = runner
        hwfc = HWFC(n_values, coord_list, coord_neighbors_fun)
        hwfc.run(ruleset, hybrid_runner, run_kwargs.get('chunk_map_fun', None), run_kwargs.get('chunk_iter_fun', None),
                 run_kwargs.get('qwfc_callback_fun', None), run_kwargs.get('hwfc_callback_fun', None))
        result.update(dict(pc=hwfc.pc))
    else:
        raise NotImplementedError
    return result


def configure_quantum_runner(backend_name=None, use_sv=False, channel=None, instance=None, shots=1024,
                             check_feasibility=False, add_barriers=False):
    if backend_name == None:
        if use_sv:
            backend = Aer.get_backend('statevector_simulator')
            run_kwarg_dict = dict()
            add_measurement = False
        else:
            backend = Aer.get_backend('qasm_simulator')
            run_kwarg_dict = dict(shots=shots)
            add_measurement = True
        quantum_runner = QuantumRunnerIBMQAer(backend=backend, run_kwarg_dict=run_kwarg_dict,
                                              check_feasibility=check_feasibility, add_barriers=add_barriers,
                                              add_measurement=add_measurement)
    else:
        tp_kwarg_dict = dict(optimization_level=3)
        run_kwarg_dict = dict()
        runtime_service_kwarg_dict = dict(channel=channel, instance=instance)
        options_kwarg_dict = dict()  # resilience_level = 1, optimization_level = 3
        add_measurement = True
        quantum_runner = QuantumRunnerIBMQRuntime(backend_name=backend_name, tp_kwarg_dict=tp_kwarg_dict,
                                                  run_kwarg_dict=run_kwarg_dict, shots=shots,
                                                  runtime_service_kwarg_dict=runtime_service_kwarg_dict,
                                                  options_kwarg_dict=options_kwarg_dict,
                                                  check_feasibility=check_feasibility, add_barriers=add_barriers,
                                                  add_measurement=add_measurement)
    return quantum_runner


def pattern_weight_fun(coord, coord_adj, coord_adj_offmap, mapped_coords, context):
    # check special cases
    if context.get('only_for_coord', None) is not None and context['only_for_coord'] != coord:
        return 0

    # check pattern
    coord_adj_all = {}
    coord_adj_all.update(coord_adj)
    coord_adj_all.update(coord_adj_offmap)
    for n_key, value in context.get('pattern', {}).items():
        if value is None:  # ignore
            continue
        if value == 'any':  # any (defined or undefined)
            if n_key not in coord_adj_all:
                return 0
            else:
                continue
        elif value == 'none':  # nothing or undefined
            if n_key not in coord_adj_all or coord_adj_all[n_key] not in mapped_coords.keys():
                continue
            else:
                for mapped_coord, mapped_value in mapped_coords.items():
                    if n_key in coord_adj and coord_adj[n_key] == mapped_coord:
                        return 0
        elif type(value) is int:
            # forall [default]
            for mapped_coord, mapped_value in mapped_coords.items():
                if n_key in coord_adj_all and coord_adj_all[n_key] == mapped_coord and mapped_value != value:
                    return 0
            # exists
            if context.get('must_exist', False):
                if sum([1 for mapped_coord, mapped_value in mapped_coords.items() if
                        n_key in coord_adj_all and coord_adj_all[
                            n_key] == mapped_coord and mapped_value == value]) == 0:
                    return 0
        else:
            raise NotImplementedError

    # return weight
    weight = context.get('weight', 1)
    if callable(weight):
        args = []
        if context.get('weight_callable_coord', True):
            args.append(coord)
        if context.get('weight_callable_coord_adj', False):
            args.append(coord_adj)
        if context.get('weight_callable_coord_adj_offmap', False):
            args.append(coord_adj_offmap)
        if context.get('weight_callable_mapped_coords', False):
            args.append(mapped_coords)
        if context.get('weight_callable_context', False):
            args.append(context)
        weight = weight(*args)
    return weight


def compose_image(mapped_coords, image_path, sprite_map, sprite_size, background_tile=None):
    img = Image.open(image_path)
    #
    w = max(coords[0] + 1 for coords in mapped_coords.keys())
    h = max(coords[1] + 1 for coords in mapped_coords.keys())
    #
    map_img = Image.new('RGBA', (w * sprite_size, h * sprite_size))
    blank_tile = Image.new('RGBA', (sprite_size, sprite_size))
    #
    for x in range(w):
        for y in range(h):
            if (x, y) in mapped_coords:
                value = mapped_coords[(x, y)]
                if value in sprite_map:
                    sprite_x, sprite_y = sprite_map[value]
                    tile_img = img.crop((sprite_x * sprite_size, sprite_y * sprite_size, (sprite_x + 1) * sprite_size,
                                         (sprite_y + 1) * sprite_size))
                    if background_tile is not None:
                        bg_x, bg_y = background_tile
                        bg_img = img.crop((bg_x * sprite_size, bg_y * sprite_size, (bg_x + 1) * sprite_size,
                                           (bg_y + 1) * sprite_size))
                        tile_img = Image.alpha_composite(bg_img, tile_img)
                    map_img.paste(tile_img, (x * sprite_size, y * sprite_size))

                else:
                    tile_img = blank_tile
            else:
                tile_img = blank_tile
            map_img.paste(tile_img, (x * sprite_size, y * sprite_size))
    #
    return map_img


def fig2img(fig, **kwargs):
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    img = Image.open(buf)
    return img
