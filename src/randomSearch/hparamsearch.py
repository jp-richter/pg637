import argparse
import random
from datetime import datetime
import json


def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def generic_random(start, end):
    if type(start) == int:
        return random.randint(start, end)
    else:
        return round(random.uniform(start, end), 5)

def parse(line):
    return line.split(" = ")


def randomize(key_val_tuple):
    key, val = key_val_tuple
    if '{' in val:
        d = json.loads(val)
        exp = generic_random(d["start"], d["end"])
        if type(d["val"]) == int:
            val = int(d["val"]**exp)
        else:
            val = round(d["val"]**exp, 6)

    elif " - " in val:
        start, end = tuple([num(v) for v in val.split(" - ")])
        val = generic_random(start, end)
    return key, val


def to_str(key_val_tuple):
    key, val = key_val_tuple
    return "{} = {};".format(key, val)


def write_to(lines, outfile_path):
    lines = [line + "\n" for line in lines]
    with open(outfile_path, 'w') as fw:
        fw.writelines(lines)


def generate_exp_names(method, n):
    day = datetime.now().day
    month = datetime.now().month
    return [f'{month}{day}-{method.upper()}-{c}' for c in list(char_range('A', 'Z'))[:n]]


def replace_exp_name(lines, exp_name):
    result = []
    for line in lines:
        if 'experimentName' in line:
            line = line.replace()
        result.append(line)
    return result


def main(file_path, method, num_configs, num_seeds):
    with open(file_path) as f:
        lines = f.read().split(";\n")[:-1]
        key_val_tuples = list(map(parse, lines))
        exp_names = generate_exp_names(method, num_configs)

        for exp_name in exp_names:
            randomized = list(map(randomize, key_val_tuples))
            lines = list(map(to_str, randomized))
            for seed_no in range(num_seeds):
                random_seed = random.randint(1, 10000)
                lines_to_write = lines.copy()
                lines_to_write.append(f'randomSeed = {random_seed};')
                lines_to_write.append(f'experimentName = "{exp_name}-{seed_no}";')
                outfile_path = f'{exp_name}-{seed_no}.cfg'
                write_to(lines_to_write, outfile_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("method")
    parser.add_argument("num_configs", type=int)
    parser.add_argument("num_seeds", type=int)
    args = parser.parse_args()
    assert(args.num_configs <= 24)
    main(args.file_path, args.method, args.num_configs, args.num_seeds)
    