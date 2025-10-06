from pathlib import Path

import yaml

map_files = ["training-maps", "test-maps"]

maps = {}
for file_name in map_files:
    with open(Path(__file__).parent / f"{file_name}.yaml", "r") as f:
        maps.update(**yaml.safe_load(f))

# Normalize custom symbols to Pogema-compatible ones
# Pogema expects FREE='.' and OBSTACLE='#'. Some maps use '!','@','$'.
# Here we remap: '!' -> '#', '@' -> '.', '$' -> '.'.
_SYMBOL_MAP = str.maketrans({
    '!': '#',
    '@': '.',
    '$': '.',
})

for _name, _map in list(maps.items()):
    if isinstance(_map, str):
        maps[_name] = _map.translate(_SYMBOL_MAP)

MAPS_REGISTRY = maps
