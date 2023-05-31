import os

assert os.system('cd checker && python checker.py --dim 1 --sv --qc') == 0
assert os.system('cd checker && python checker.py --dim 2 --sv') == 0

assert os.system('cd lines && python lines.py') == 0

assert os.system('cd platformer && python platformer.py') == 0
assert os.system('cd platformer && python platformer.py --big') == 0
assert os.system('cd platformer && python platformer.py -x 4 -y 4 --segment-x-size 0') == 0

assert os.system('cd hex && python hex.py') == 0
