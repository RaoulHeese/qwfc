import os

from tests.checker.checker import run as checker_run
from tests.hex.hex import run as hex_run
from tests.lines.lines import run as lines_run
from tests.platformer.platformer import run as platformer_run
from tests.poetry.poetry import run as poetry_run
from tests.voxel.voxel import run as voxel_run

# checker
os.chdir('checker')
checker_run(map_dim=2, map_size=3, use_sv=True, shots=5, engine='Q')
checker_run(map_dim=2, map_size=3, shots=5, engine='C')
checker_run(map_dim=3, map_size=3, shots=5, engine='Q')
os.chdir('..')

# lines
os.chdir('lines')
lines_run(map_x_size=4, map_y_size=4, alpha=0, engine='C')
lines_run(map_x_size=4, map_y_size=4, alpha=1, engine='C')
lines_run(map_x_size=2, map_y_size=2, alpha=0, use_sv=False, shots=1, engine='Q')
lines_run(map_x_size=4, map_y_size=4, chunk_x_size=2, chunk_y_size=2, alpha=0, use_sv=False, shots=1, engine='H')
os.chdir('..')

# hex
os.chdir('hex')
hex_run(map_size=7, n_chunks=20, shots=1, engine='H')
os.chdir('..')

# platformer
os.chdir('platformer')
platformer_run(map_x_size=4, map_y_size=6, n_chunks=6, alpha=.1, shots=1, engine='H')
os.chdir('..')

# voxel
os.chdir('voxel')
voxel_run(map_size=5, map_height=5, shots=10, engine='Q')
os.chdir('..')

# poetry
os.chdir('poetry')
poetry_run(txt_file_path='alice.txt', txt_file_encoding='utf8', txt_file_skip_chars=560, maximum_connectivity=8,
           n_words=8, n_chunks=1, generate_images_flag=True, shots=100, engine='H')
os.chdir('..')
