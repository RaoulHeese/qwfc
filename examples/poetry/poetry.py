import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from qiskit import Aer
import re
from itertools import product

from image_generator import generate_images

sys.path.insert(0, '../../src')
from qwfc import Map, MapSlidingWindow, CorrelationRuleSet
from runner import CircuitRunnerIBMQAer

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


def coord_rules_fun(coord, visited_coord_adj, coord_adj_offmap, n_values, coord_fixed, connectivity, word_segment_size):
    ruleset = CorrelationRuleSet(n_values)
    word_list = parse_word_path(coord_fixed, connectivity)
    word = word_list[-1] if len(word_list) > 0 else None
    n_unexplored = min(len(visited_coord_adj), word_segment_size)
    #
    for values in product(range(n_values), repeat=n_unexplored):
        rule_word = word
        is_contained = True
        for value in values:
            if value < len(connectivity[rule_word]):
                rule_word = list(connectivity[rule_word].keys())[value]
            else:
                is_contained = False
                break
        if is_contained:
            visited_coord_adj_values = [kv[1] for kv in
                                        sorted([kv for kv in visited_coord_adj.items()], key=lambda kv: kv[0])]
            coord_value_dict = {visited_coord_adj_values[k]: values[k] for k in range(n_unexplored)}
            connectivity_values = list(connectivity[rule_word].values())
            p_dict = {n: (connectivity_values[n] if len(connectivity[rule_word]) > n else 0.) for n in range(n_values)}
            p_sum = sum(p_dict.values())
            p_dict = {idx: value / p_sum for idx, value in p_dict.items()}
            ruleset.add(coord_value_dict, p_dict)
        else:
            pass
    return ruleset


def segment_map_fun(parsed_counts, composition_path):
    # use most probable
    for _, (_, coord_map) in parsed_counts.items():
        if composition_path is not None:
            composition_path[len(composition_path)] = {'chosen': coord_map}
        return coord_map


def coord_neighbors_fun(coord, word_segment_size):
    n = coord[0]
    coord_dict = {-(k + 1): (n - (k + 1),) for k in range(word_segment_size)}
    return coord_dict


def segment_coord_path_fun(coord_list):
    return (coord for coord in coord_list)


def segment_iter_fun(coord_list, word_segment_size):
    for coord_list_seg in [coord_list[i:i + word_segment_size] for i in range(0, len(coord_list), word_segment_size)]:
        yield coord_list_seg, segment_coord_path_fun


def process_output(mapped_coords, connectivity, space=' '):
    words = parse_word_path(mapped_coords, connectivity)
    word_list = []
    for idx, word in enumerate(words):
        if idx > 0 and word not in special_chars:
            word_list.append(space)
        word_list.append(word)
    return word_list


def callback_fun(msw, idx, map_segment, segment_mapped_coords, connectivity, composition_path):
    if composition_path is not None:
        composition_path[len(composition_path) - 1]['options'] = list(map_segment.pc.values())


def segment_callback_fun(m, idx, coord):
    pass


def process_results(mapped_coords, connectivity, composition_path, name):
    if not os.path.isdir('results'):
        os.mkdir('results')

    # process final poem
    print('process final poem')
    word_list = process_output(mapped_coords, connectivity)
    poem = ''.join(word_list)
    print(f'poem: "{poem}"')
    with open(f'results/{name}.txt', 'w') as fh:
        fh.write(poem)
    print(f'poem saved to {name}.txt')

    # generate additional results
    if composition_path is not None:
        image_list = generate_images(composition_path,
                                     path_to_word_list_fun=lambda word_list: process_output(word_list, connectivity))
        for idx, image in enumerate(image_list):
            filename = f'results/{name}-{idx:04d}.png'
            image.save(filename)
        print(f'{len(image_list)} images saved to {name}*.png')


def run(txt_file_path, txt_file_encoding, skip_chars, maximum_connectivity, n_words, word_segment_size, shots,
        verbose_load_flag, generate_images_flag, name):
    # load text
    connectivity = load_text(txt_file_path, txt_file_encoding, skip_chars, maximum_connectivity, verbose_load_flag)

    # input
    circuit_runner = CircuitRunnerIBMQAer(backend=Aer.get_backend('qasm_simulator'), run_kwarg_dict=dict(shots=shots))
    n_values = maximum_connectivity
    coord_list = [(n,) for n in range(n_words)]
    composition_path = {} if generate_images_flag else None
    check_feasibility = False

    # run
    msw = MapSlidingWindow(n_values, coord_list, lambda coord: coord_neighbors_fun(coord, word_segment_size), check_feasibility)
    msw.run(lambda parsed_counts: segment_map_fun(parsed_counts, composition_path),
            lambda coord_list: segment_iter_fun(coord_list, word_segment_size),
            lambda coord, visited_coord_adj, coord_adj_offmap, n_values, coord_fixed: coord_rules_fun(coord,
                                                                                                      visited_coord_adj,
                                                                                                      coord_adj_offmap,
                                                                                                      n_values,
                                                                                                      coord_fixed,
                                                                                                      connectivity,
                                                                                                      word_segment_size),
            circuit_runner=circuit_runner,
            segment_callback_fun=segment_callback_fun,
            callback_fun=lambda msw, idx, map_segment, segment_mapped_coords: callback_fun(msw, idx, map_segment,
                                                                                           segment_mapped_coords,
                                                                                           connectivity,
                                                                                           composition_path))

    # output
    print('finished:')
    process_results(msw.mapped_coords, connectivity, composition_path, name)


# args
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--txt-file-path', type=str, default='alice.txt', help='path of the text file to process')
parser.add_argument('-e', '--txt-file-encoding', type=str, default='utf8', help='encoding of the text file to process')
parser.add_argument('-s', '--skip-chars', type=int, default=560,
                    help='number of characters to skip at the beginning of the text file (to crop the relevant part)')
parser.add_argument('-c', '--maximum-connectivity', type=int, default=8,
                    help='maximum number of adjacent words that are considered for the Markov condition (a higher value leads to more possibilities, choose a power of 2 to exploit all quantum states)')
parser.add_argument('-n', '--num-words', type=int, default=42, help='total number of words to compose')
parser.add_argument('-w', '--word-segment-size', type=int, default=3,
                    help='size of word segments (i.e., map segments)')
parser.add_argument('--name', type=str, default='poem', help='name prefix of the result files')
parser.add_argument('--shots', type=int, default=8,
                    help='number of shots for non-statevector (a lower value induces more randomness)')
parser.add_argument('--verbose-load', dest='verbose_load_flag', action='store_true',
                    help='show text file processing progress')
parser.set_defaults(verbose_load_flag=False)
parser.add_argument('--generate-images', dest='generate_images_flag', action='store_true',
                    help='generate and store images of the probabilistic poem generation progess')
parser.set_defaults(generate_images_flag=False)
args = parser.parse_args()

if __name__ == '__main__':
    """
    One-dimensional tile arrangement, where each tile represents a word (string). The word sequence is created from left to right according to the reading direction. The probability of each word is only based on the previous word (Markov condition) and depends on the ratio of occurence within a sample text that can be provided via -f.
    """

    run(args.txt_file_path, args.txt_file_encoding, args.skip_chars, args.maximum_connectivity, args.num_words,
        args.word_segment_size, args.shots, args.verbose_load_flag, args.generate_images_flag, args.name)
