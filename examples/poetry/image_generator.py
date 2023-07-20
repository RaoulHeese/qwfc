from PIL import Image
from PIL import ImageDraw
import numpy as np
import os


def paint_word(word='', p=1, n=None):
    char_w = 6
    char_h = 12
    if n is None:
        n = len(word)
    w = n * char_w
    h = char_h
    bg = (0, 255, 0, 0)
    image = Image.new('RGBA', (w, h), bg)
    draw = ImageDraw.Draw(image)
    alpha = int(np.clip(np.clip(p, 0, 1) * 255, 0, 255))
    draw.text((0, 0), word, (0, 0, 0, alpha))
    return image


def paint_p_word_list(p_word_list):
    k = len(p_word_list)
    N = max([len(w) for _, w in p_word_list])
    imgs = []
    for n in range(N):
        p_words = []
        for p, word_list in p_word_list:
            if len(word_list) <= n:
                word = ''
            else:
                word = word_list[n]
            p_words.append((p, word))
            word_max_len = max([len(word) for _, word in p_words])
        img = paint_word(n=word_max_len)
        for p, word in p_words:
            jmg = paint_word(word=word, p=p, n=word_max_len)
            img = Image.alpha_composite(img, jmg)
        imgs.append(img)
    #
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    full_img = Image.new('RGBA', (total_width, max_height))
    x_offset = 0
    for img in imgs:
        full_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return full_img


def parse_composition_path(composition_path, path_to_word_list_fun):
    images_dict = {}
    path = {}
    for idx, path_info in composition_path.items():
        p_word_list = []
        for p, coord_map_option in path_info['options']:
            path_option = path.copy()
            path_option.update(coord_map_option)
            word_list = path_to_word_list_fun(path_option)
            p_word_list.append((p, word_list))
        images_dict[idx] = {'options': paint_p_word_list(p_word_list)}
        coord_map = path_info['chosen']
        path.update(coord_map)
        word_list = path_to_word_list_fun(path)
        n = len(path_info['options'])
        images_dict[idx]['chosen'] = paint_p_word_list(
            [(1 / n, word_list) for _ in range(n)])  # [(1., word_list)] leads to other color
    return images_dict


def generate_images(composition_path, path_to_word_list_fun):
    image_list = []
    images_dict = parse_composition_path(composition_path, path_to_word_list_fun)
    for image_dict in images_dict.values():
        image_list.append(image_dict['options'])
    if len(images_dict) > 0:
        image_list.append(list(images_dict.values())[-1]['chosen'])
    return image_list
