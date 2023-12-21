import argparse
import os
import numpy as np
from qiskit import Aer
from datetime import datetime
from itertools import product
from functools import partial
from tqdm import tqdm
import re
from qwfc.common import DirectionRuleSet
from qwfc.runner import ClassicalRunnerDefault, QuantumRunnerIBMQAer, HybridRunnerDefault
import json
from tests.example_utils import run_wfc, configure_quantum_runner, pattern_weight_fun
from tests.poetry.image_generator import generate_images
from qwfc._version import __version__

special_chars = ['.', '!', '?']

def load_text(txt_file_path, txt_file_encoding='utf8', skip_chars=0, maximum_connectivity=None, verbose=False):
    # utility dictionaries
    str_text_replace = {
        "_": "",
        "“": "'",
        "”": "'",
        "’": "'",
        "ù": "u",
        "-'": "",
        "CHAPTER I.": "",
        "CHAPTER II.": "",
        "CHAPTER III.": "",
        "CHAPTER IV.": "",
        "CHAPTER V.": "",
        "CHAPTER VI.": "",
        "CHAPTER VII.": "",
        "CHAPTER VIII.": "",
        "CHAPTER IX.": "",
        "CHAPTER X.": "",
        "CHAPTER XI.": "",
        "CHAPTER XII.": "",
    }

    str_word_replace = {
        "—": "",
    }

    word_map = {
    }

    # utility functions
    def read_text(txt_file_path, txt_file_encoding, skip_chars):
        with open(txt_file_path, 'r', encoding=txt_file_encoding) as fh:
            text = fh.read()
        return text[skip_chars:]

    def text_to_words(text):
        for str_from, str_to in str_text_replace.items():
            text = text.replace(str_from, str_to)
        words = re.split(r"([^0-9a-zA-Z'-])", text)
        return words

    def transform_word(word):
        word = word.strip().lower()
        for str_from, str_to in str_word_replace.items():
            word = word.replace(str_from, str_to)
        if len(word) > 1:
            word = ''.join([c for i, c in enumerate(word) if
                            c.isalnum() or (c == "'" and i > 0) or (c == "-" and i > 0 and i < len(word) - 1)])
        if word in word_map:
            word = word_map[word]
        return word

    def word_is_valid(word):
        if len(word) == 0:
            return False
        if len(word) == 1:
            return word in special_chars
        return True

    # perform text transformation
    connectivity = {None: {}}
    previous_word = None
    for word in tqdm(text_to_words(read_text(txt_file_path, txt_file_encoding, skip_chars)), disable=not verbose):
        word = transform_word(word)
        if word_is_valid(word):
            if previous_word is not None:
                if previous_word not in connectivity:
                    connectivity[previous_word] = {}
                if word not in connectivity[previous_word]:
                    connectivity[previous_word][word] = 0
                connectivity[previous_word][word] += 1
            if word not in special_chars:
                if word not in connectivity[None]:
                    connectivity[None][word] = 0
                connectivity[None][word] += 1
            previous_word = word
    if previous_word not in connectivity:
        connectivity[previous_word] = connectivity[None]  # loop with start

    # reduce connectivity
    if maximum_connectivity is not None:
        for word in connectivity.keys():
            connectivity[word] = dict(
                sorted(connectivity[word].items(), key=lambda item: -item[1])[:maximum_connectivity])

    # return connectivity map
    return connectivity


def parse_word_path(coord_fixed, connectivity):
    word_list = []
    word = None
    for value in coord_fixed.values():
        value = value % len(connectivity[word])  # wrap if too long (only relevant for noisy evaluation)
        word = list(connectivity[word].keys())[value]
        word_list.append(word)
    return word_list

def process_output(mapped_coords, connectivity, space=' '):
    words = parse_word_path(mapped_coords, connectivity)
    word_list = []
    for idx, word in enumerate(words):
        if idx > 0 and word not in special_chars:
            word_list.append(space)
        word_list.append(word)
    return word_list


def process_poem(mapped_coords, connectivity, composition_path):
    # text
    print('process final poem')
    word_list = process_output(mapped_coords, connectivity)
    poem = ''.join(word_list)

    # generate additional results
    if composition_path is not None:
        image_list = generate_images(composition_path, path_to_word_list_fun=lambda word_list: process_output(word_list, connectivity))
    else:
       image_list = []
    print('composition_path', composition_path)

    return poem, image_list

def process_result(result, connectivity, composition_path, prefix=''):
    timestamp = str(datetime.now().timestamp())
    prefix = f'{prefix[:64]}{timestamp}-'
    #
    pc = result['pc']
    #
    if not os.path.isdir('results'):
        os.mkdir('results')
    # process final poem
    for idx, (key, (p, mapped_coords, f)) in enumerate(pc.items()):
        key = str(key)[:100]
        filename_txt = f'results/{prefix}{idx}-{key}.txt'
        poem, image_list = process_poem(mapped_coords, connectivity, composition_path)
        print(f'{key}: p={p}, f={f}, file={filename_txt}, poem={poem}')
        with open(filename_txt, 'w') as fh:
            fh.write(poem)
        for jdx, image in enumerate(image_list):
            filename_img = f'results/{prefix}{idx}-{jdx:04d}.png'
            image.save(filename_img)
    #
    filename = f'results/{prefix}data.json'
    with open(filename, 'w') as fh:
        data = dict(pc = {str(key): (float(p), {str(c): int(v) for c,v in mapped_coords.items()}, bool(f) if f is not None else None) for (key, (p, mapped_coords, f)) in pc.items()}, version=__version__)
        json.dump(data, fh)

def run(txt_file_path, txt_file_encoding, txt_file_skip_chars, maximum_connectivity, n_words, n_chunks=1,
        verbose_load_flag=False, generate_images_flag=False, backend_name=None, channel=None, instance=None, use_sv=False, shots=1, engine='Q', name=''):

    # load text
    connectivity = load_text(txt_file_path, txt_file_encoding, txt_file_skip_chars, maximum_connectivity, verbose_load_flag)
    print(f'processed text: {len(connectivity)} tokens')
    word_segment_size = 1

    # images
    composition_path = {} if generate_images_flag else None

    def coord_neighbors_fun(word_segment_size, coord):
        n = coord[0]
        coord_dict = {-(k + 1): (n - (k + 1),) for k in range(word_segment_size)} # -1 : coord, ..., -word_segment_size: coord
        return coord_dict

    def coord_list_fun():
        return [(n,) for n in range(n_words)]

    def chunk_map_fun(composition_path, parsed_counts):
        # use most probable
        for _, (_, coord_map, feasibility) in parsed_counts.items():
            if feasibility is None or feasibility:
                if composition_path is not None:
                    composition_path[len(composition_path)] = {}
                    composition_path[len(composition_path)-1]['chosen'] = coord_map
                return coord_map
        return None

    def chunk_iter_fun(coord_list):
        for coord_list_seg in np.array_split(coord_list, n_chunks):
            yield [tuple(coord) for coord in coord_list_seg], None

    def hwfc_callback_fun(composition_path, pbar, hwfc, idx, map_chunk, chunk_mapped_coords):
        pbar.update(1)
        if composition_path is not None:
            if map_chunk.pc is not None:
                composition_path[len(composition_path) - 1]['options'] = [(p, mc) for (p, mc, _) in map_chunk.pc.values()]
            else:
                composition_path[len(composition_path) - 1]['options'] = []

    def hwfc_qwfc_callback_fun(pbar, qwfc, idx, coord):
        pbar.set_postfix({'idx': idx, 'coord': coord})

    def qwfc_callback_fun(pbar, qwfc, idx, coord):
        hwfc_qwfc_callback_fun(pbar, qwfc, idx, coord)
        pbar.update(1)

    def cwfc_callback_fun(pbar, cwfc, idx, coord, options, new_value):
        pbar.set_postfix({'idx': idx, 'coord': coord, 'new_value': new_value})
        pbar.update(1)

    # values
    n_values = maximum_connectivity # context-dependent word identifiers

    # coordinates
    coord_list = coord_list_fun()

    # rules
    def context_dependent_weight_fun(mapped_coords, context):
        word_list = parse_word_path(mapped_coords, connectivity)
        rule_word = word_list[-1] if len(word_list) > 0 else None
        value = context['contextual_value']
        connectivity_values = list(connectivity[rule_word].values())
        p_dict = {n: (connectivity_values[n] if len(connectivity[rule_word]) > n else 0.) for n in range(n_values)}
        p_sum = sum(p_dict.values())
        return p_dict[value]/p_sum
    #
    ruleset = DirectionRuleSet(n_values)
    n_keys = [-(k+1) for k in range(word_segment_size)]
    for adj_vals in product(range(n_values), repeat=len(n_keys)):
        pattern = {n_key: adj_val for n_key, adj_val in zip(n_keys, adj_vals)}
        for value_const in range(n_values):
            context = {'pattern': pattern,
                       'weight': context_dependent_weight_fun,
                       'weight_callable_coord': False,
                       'weight_callable_mapped_coords': True,
                       'weight_callable_context': True,
                       'contextual_value': value_const}
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
        quantum_runner = configure_quantum_runner(backend_name=backend_name, use_sv=use_sv, channel=channel, instance=instance,
                                          shots=shots, check_feasibility=False, add_barriers=False)
        runner = HybridRunnerDefault(quantum_runner=quantum_runner)
        total_steps = n_chunks
        pbar = tqdm(total=total_steps, desc='hwfc')
        run_kwargs = dict(chunk_map_fun=partial(chunk_map_fun, composition_path),
                          chunk_iter_fun=chunk_iter_fun,
                          qwfc_callback_fun=partial(hwfc_qwfc_callback_fun, pbar),
                          hwfc_callback_fun=partial(hwfc_callback_fun, composition_path, pbar))
    else:
        raise NotImplementedError

    # run
    pbar.reset()
    result = run_wfc(runner, n_values, coord_list, partial(coord_neighbors_fun, word_segment_size), ruleset, **run_kwargs)
    pbar.close()
    process_result(result, connectivity, composition_path, prefix=f'{name}-{engine}-n{n_words}x{word_segment_size}-'),

# args
parser = argparse.ArgumentParser()
parser.add_argument('--txt-file-path', type=str, default='alice.txt', help='text file')
parser.add_argument('--txt-file-encoding', type=str, default='utf8', help='text file encoding')
parser.add_argument('--txt-file-skip-chars', type=int, default=560, help='number of characters to skip at the beginning of the text file')
parser.add_argument('--maximum-connectivity', type=int, default=8, help='maximum word graph connections, corresponds to number of values')
parser.add_argument('--n-words', type=int, default=16, help='number of words in generated poem')
parser.add_argument('--n-chunks', type=int, default=4, help='number of chunks (only for H)')
parser.add_argument('--verbose-load', dest='verbose_load_flag', action='store_true', help='show text file processing progress')
parser.set_defaults(verbose_load_flag=True)
parser.add_argument('--generate-images', dest='generate_images_flag', action='store_true', help='generate and store images of the probabilistic poem generation progess (only for H, shots > 1)')
parser.set_defaults(generate_images_flag=True)
parser.add_argument('--backend-name', type=str, default=None, help='IBMQ backend name, None for local simulator (default: None)')
parser.add_argument('--channel', type=str, default=None, help='IBMQ runtime service channel (default: None)')
parser.add_argument('--instance', type=str, default=None, help='IBMQ runtime service instance (default: None)')
parser.add_argument('--sv', dest='use_sv', action='store_true', help='use statevector simulator')
parser.set_defaults(use_sv=False)
parser.add_argument('--shots', type=int, default=100, help='number of shots for non-statevector and CWFC')
parser.add_argument('--engine', type=str, choices=['C', 'Q', 'H'], default='H', help='WFC engine: C, Q or H (default)')
parser.add_argument('--name', type=str, default='poetry', help='result filename')
args = parser.parse_args()

if __name__ == '__main__':
    """
    One-dimensional tile arrangement, where each tile represents a word (string). The word sequence is created from left to right according to the reading direction. The probability of each word is only based on the previous word (Markovian) and depends on the ratio of occurence within the provided text.
    """
    run(txt_file_path=args.txt_file_path,
    txt_file_encoding=args.txt_file_encoding,
    txt_file_skip_chars=args.txt_file_skip_chars,
    maximum_connectivity=args.maximum_connectivity,
    n_words=args.n_words,
    n_chunks=args.n_chunks,
    verbose_load_flag=args.verbose_load_flag,
    generate_images_flag=args.generate_images_flag,
    backend_name=args.backend_name, channel=args.channel, instance=args.instance,
    use_sv=args.use_sv,
    shots=args.shots,
    engine=args.engine,
    name=args.name)
