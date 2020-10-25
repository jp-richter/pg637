position_ids = [i for i in range(36)]
field_length = 6

INVALID_PID = -1

def left(position_id):
    if position_id % field_length > 0:
        return position_id - 1

    return INVALID_PID


def right(position_id):
    if position_id % field_length < field_length:
        return position_id + 1

    return INVALID_PID


def up(position_id):
    if position_id / field_length >= 1:
        return position_id - field_length

    return INVALID_PID 


def down(position_id):
    if position_id / field_length <= (field_length - 1):
        return position_id + field_length

    return INVALID_PID


next_position_functions = {
    'left': left,
    'right': right,
    'up': up,
    'down': down
}


labyrinth = {
    0: ['right', 'down'],
    1: ['left', 'down'],
    2: ['right', 'down'],
    3: ['left', 'right', 'down'],
    4: ['left', 'right'],
    5: ['left'],

    6: ['up', 'down'],
    7: ['right', 'up'],
    8: ['left', 'up', 'down'],
    9: ['right', 'up', 'down'],
    10: ['left', 'right', 'down'],
    11: ['left'],

    12: ['up', 'down'],
    13: ['right', 'down'],
    14: ['left', 'up'],
    15: ['up', 'down'],
    16: ['up', 'down'],
    17: ['down'],

    18: ['right', 'up'],
    19: ['left', 'right', 'up', 'down'],
    20: ['left', 'down'],
    21: ['right', 'up'],
    22: ['left', 'up', 'down'],
    23: ['up', 'down'],

    24: ['right'],
    25: ['left', 'up', 'down'],
    26: ['up', 'down'],
    27: ['right'],
    28: ['left', 'up', 'right'],
    29: ['left', 'up', 'down'],

    30: ['right'],
    31: ['left', 'right', 'up'],
    32: ['left', 'right', 'up'],
    33: ['left', 'right'],
    34: ['left', 'right'],
    35: ['left', 'up']
}

assert not any(next_position_functions[d](wpid) == INVALID_PID for (wpid, directions) in labyrinth.items() for d in directions)

entry_id = 24
exit_id = 2  # 11
trap_id = 17

def get_valid_directions(position_id):
    return labyrinth[position_id]

def move(direction, source_id):
    if source_id == exit_id:
        return source_id, 1

    if source_id == trap_id:
        return source_id, -1

    if direction not in get_valid_directions(source_id):
        return source_id, 0

    target_id = next_position_functions[direction](source_id)

    if target_id == INVALID_PID:
        raise ValueError('Something went horribly wrong!')

    return target_id, 0
