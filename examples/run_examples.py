import os
import sys


def run_example(sub_dir, python_file, *args):
    assert os.system(f'cd {sub_dir} && {sys.executable} {python_file} {" ".join([str(arg) for arg in args])}') == 0


run_example('checker', 'checker.py', '--dim 1 --sv --qc')
run_example('checker', 'checker.py', '--dim 2 --sv')

run_example('lines', 'lines.py')

run_example('platformer', 'platformer.py')
run_example('platformer', 'platformer.py', '--big')
run_example('platformer', 'platformer.py', '-x 4 -y 4 --segment-x-size 0')

run_example('hex', 'hex.py')

run_example('poetry', 'poetry.py')
