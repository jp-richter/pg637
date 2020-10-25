position_ids = [i for i in range(36)]
field_length = 6

INVALID_PID = -1
FINISHED = -2

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
    7: ['up'],
    8: ['up', 'down'],
    9: ['up', 'down'],
    10: ['right', 'down'],
    11: ['left'],

    12: ['up', 'down'],
    13: ['right', 'down'],
    14: ['left', 'up'],
    15: ['up', 'down'],
    16: ['up', 'down'],
    17: ['down'],

    18: ['right', 'up'],
    19: ['left', 'right', 'up'],
    20: ['left', 'down'],
    21: ['right', 'up'],
    22: ['left', 'up'],
    23: ['up', 'down'],

    24: ['right'],
    25: ['left', 'down'],
    26: ['up', 'down'],
    27: ['right'],
    28: ['left', 'right'],
    29: ['left', 'up', 'down'],

    30: ['right'],
    31: ['left', 'right', 'up'],
    32: ['left', 'right', 'up'],
    33: ['left', 'right'],
    34: ['left', 'right'],
    35: ['left', 'up'],

    FINISHED: []
}

assert not any(next_position_functions[d](wpid) == INVALID_PID for (wpid, directions) in labyrinth.items() for d in directions)

entry_id = 24
exit_id = 11
trap_id = 5

def get_valid_directions(position_id):
    return labyrinth[position_id]

def move(direction, source_id):
    if direction not in get_valid_directions(source_id):
        return source_id, 0

    target_id = next_position_functions[direction](source_id)

    if target_id == INVALID_PID:
        raise ValueError('Something went horribly wrong!')

    success, target_id = (1, FINISHED) if target_id == exit_id else (0, target_id)
    failure, target_id = (-1, FINISHED) if target_id == trap_id else (0, target_id)

    return target_id, (success + failure)
