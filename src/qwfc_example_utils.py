from itertools import product
from PIL import Image


def coord_rules_fun_generator(adj_order, necessary_rules_dict, correlation_ruleset_generator):
    def coord_rules_fun(coord, visited_coord_adj, coord_adj_offmap, n_values, coord_fixed):
        # extract constraints
        constrained_coord_adj = {adj_key: coord_fixed[coord] for adj_key, coord in coord_adj_offmap.items() if
                                 coord in coord_fixed}
        constraints = [constrained_coord_adj.get(adj_key) for adj_key in adj_order]

        # filter rules by constraints and adjacency
        rules_filtered = {}
        for rule in necessary_rules_dict:
            if all([c is None or r is None or r == c for r, c in zip(rule, constraints)]) and all(
                    [r is None or (adj_key in constrained_coord_adj or adj_key in visited_coord_adj) for r, adj_key in
                     zip(rule, adj_order)]):
                rules_filtered[rule] = necessary_rules_dict[rule]

        # sort by number of Nones:
        rules_filtered = {k: v for k, v in
                          sorted(rules_filtered.items(), key=lambda item: -sum([r is None for r in item[0]]))}

        # check rule dependencies and add clarifications
        rules_merged = {}
        for rule in rules_filtered:
            unclear_rule_idx = []
            for rule_check in rules_filtered:
                if all([r == r_c or r is None for r, r_c in zip(rule, rule_check)]) and rule != rule_check:
                    for idx, (r, r_c) in enumerate(zip(rule, rule_check)):
                        if r is None and r_c is not None and idx not in unclear_rule_idx:
                            unclear_rule_idx.append(idx)
            for rule_check in rules_merged:
                if all([r == r_c or r is None for r, r_c in zip(rule, rule_check)]) and rule != rule_check:
                    for idx, (r, r_c) in enumerate(zip(rule, rule_check)):
                        if r is None and r_c is not None and idx not in unclear_rule_idx:
                            unclear_rule_idx.append(idx)
            if len(unclear_rule_idx) > 0:
                for r_replace in product(range(n_values), repeat=len(unclear_rule_idx)):
                    new_rule = []
                    replace_idx = 0
                    for idx in range(len(rule)):
                        if idx in unclear_rule_idx:
                            new_rule.append(r_replace[replace_idx])
                            replace_idx += 1
                        else:
                            new_rule.append(rule[idx])
                    new_rule = tuple(new_rule)
                    if all([c is None or r is None or r == c for r, c in
                            zip(new_rule, constraints)]) and new_rule not in rules_filtered:
                        rules_merged[new_rule] = rules_filtered[rule]
            else:
                new_rule = rule
                rules_merged[new_rule] = rules_filtered[rule]

        # finalize rules
        rules_clean = {}
        for rule, d in rules_merged.items():
            p_dict = {k: abs(v(coord) if callable(v) else float(v)) for k, v in d.items()}
            norm = sum(p_dict.values())
            if norm > 0:
                p_dict = {k: v / norm for k, v in p_dict.items()}
            else:
                p_dict = {k: 1 / n_values for k, v in p_dict.items()}
            rules_clean[rule] = p_dict

        # traverse finalized rules
        correlation_ruleset = correlation_ruleset_generator()
        for rule, p_dict in rules_clean.items():
            if all([adj_key in visited_coord_adj or adj_key in constrained_coord_adj for r, adj_key in
                    zip(rule, adj_order) if r is not None]):
                coord_value_dict = {}
                for r, adj_key in zip(rule, adj_order):
                    if r is not None and adj_key in visited_coord_adj:
                        coord_value_dict[visited_coord_adj[adj_key]] = r
                correlation_ruleset.add(coord_value_dict, p_dict)
        if len(correlation_ruleset) == 0:
            coord_value_dict = {}
            p_dict = {k: 1 / n_values for k in range(n_values)}
            correlation_ruleset.add(coord_value_dict, p_dict)

        # return resulting correlation ruleset
        return correlation_ruleset

    return coord_rules_fun


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
