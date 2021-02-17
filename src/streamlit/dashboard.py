import streamlit
import altair
import numpy
import json
import pandas
import os
import re
import numbers

#
# Run this script with "streamlit run dashboard.py" from the directory the script is located in. Change the parameters
# below. The script makes a few assumptions about the format of the json logs, which are listed below. If your logs
# don't meet these assumptions, update your log files manually or the respective log entries will be omitted.
#
#     - The directory your logs are located in should adhere to the naming convention '{methodname}-{char}-{mmdd}', f.e.
#       'SAC-A-0221'.
#     - 'Info.json' and 'Logs.json' have to exist in that directory.
#     - 'Info.json' has to have keys: MethodName, ShortDescription, LongDescription, Notes, Runtime, Hyperparameter. The
#       method name has to be equal to the name of the base directory.
#     - 'Logs.json' consists of entries {Log Name}: {'Values': [tuple], 'Framestamps': str, 'Plot Type': str}. The
#       framestamps value can be set to 'True' or 'False'. For plot types see the logger documentation.
#     - Depending on the plot type the tuples in the list entry of 'Values' can only be of certain length. If
#       framestamps is set to 'True', one additional dimension is allowed. If you dont actually want to plot your log
#       you can use plot type Empty.
#

LAYOUT = 'wide'  # change this, options are wide and centered
PATH = './'  # change this, this is the path to the folder containing the experiments
FILL_BROWSER_WIDTH = True  # iff true, the plots will expand to the full length of your browser window


# keys used in the json log, dont change anything below

KEY_METHOD_NAME = 'MethodName'
KEY_SHORT_DESCR = 'ShortDescription'
KEY_LONG_DESSCR = 'LongDescription'
KEY_NOTES = 'Notes'
KEY_RUNTIME = 'Runtime'
KEY_HYPERPARAMETERS = 'Hyperparameter'
KEY_VALUES = 'Values'
KEY_FRAMESTAMPS = 'Framestamps'
KEY_PLOTTYPE = 'Plot Type '


def main():
    streamlit.set_page_config(layout=LAYOUT)  # options are wide and centered
    experiments = load(base_path=PATH)  # change the base path to the actual path
    experiment_chosen = streamlit.sidebar.selectbox('Choose an experiment!', list(experiments.keys()))
    streamlit.title(experiment_chosen)

    for name, data in experiments.items():
        streamlit.sidebar.markdown('''
            **{}**: {}
        '''.format(name, data[KEY_SHORT_DESCR]))

    for name, data in experiments.items():
        if name == experiment_chosen:
            visualize(name, data)


@streamlit.cache
def load(base_path):
    experiments = []  # (log, info)
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]

    for folder in folders:
        basename = os.path.basename(folder)

        if not re.match(r'[a-zA-Z0-9]*-[a-zA-Z]-[0-9][0-9][0-9][0-9]', basename):
            print(f'Error: Folder {folder} does not adhere to the naming convention and will be omitted. Naming '
                  f'convention is [a-zA-Z0-9]*-[a-zA-Z]-[0-9][0-9][0-9][0-9], f.e. SAC-A-0205, where 0205 is the date'
                  f'of format mmdd. As we dont store the date somewhere else, this is quite convenient. Suggestions to '
                  f'change the naming convention are welcome.')
            continue

        if not os.path.exists(os.path.join(folder, 'Info.json')) or not os.path.exists(
                os.path.join(folder, 'Logs.json')):
            print(f'Error: Folder {folder} does not contain Info.json or Logs.json and will be omitted.')
            continue

        with open(os.path.join(folder, 'Info.json'), 'r') as file:
            info = json.load(file)

        with open(os.path.join(folder, 'Logs.json'), 'r') as file:
            logs = json.load(file)

        required_keys = [
            KEY_METHOD_NAME,
            KEY_SHORT_DESCR,
            KEY_LONG_DESSCR,
            KEY_RUNTIME,
            KEY_NOTES,
            KEY_HYPERPARAMETERS
        ]

        if not all((k in info.keys() for k in required_keys)):
            print(f'Error: {folder} does not contain all required data {required_keys} and will be omitted. If you '
                  f'already run your experiment, add the entries manually to the json file.')
            continue

        if not basename == info[KEY_METHOD_NAME]:
            print(f'Error: Folder is named {folder} and method is named {info["MethodName"]}, please stick to the same'
                  f'naming convention. Suggestions to change the naming convention are welcome. The folder will be '
                  f'omitted.')

        allowed_dimensions = {
            'line': 1,
            'histogram': 1,
            'histogram2d': 2,
            'scatter': 2,
            'tube': 2,
            'Empty': 999999999
        }

        cache = []

        for name, log in logs.items():
            if not type(log[KEY_VALUES][0]) == list:
                print(f'Warning: Non-tuple type in value log of {name} in {folder}/Logs.json. The entries will be '
                      f'interpreted as 1-dimensional tuples.')

                try:
                    for i in range(len(log[KEY_VALUES])):
                        log[KEY_VALUES][i] = list(log[KEY_VALUES][i])
                except Exception as e:
                    print(f'Error: Interpreting entries as 1-dimensional tuples failed, the log will be omitted. '
                          f'Stacktrace: {e}')
                    cache.append(name)
                    continue

            if not isinstance(log[KEY_VALUES][0][0], numbers.Number):
                print(f'Warning: Non-number type in value log of {name} in {folder}/Logs.json, found type '
                      f'{type(log[KEY_VALUES][0][0])} instead. Log will be omitted.')
                cache.append(name)
                continue

            dimension_actual = len(log[KEY_VALUES][0])
            dimension_allowed = allowed_dimensions[log[KEY_PLOTTYPE]]

            if log[KEY_FRAMESTAMPS] == 'True':  # assumed to be the first dimension
                dimension_allowed += 1

            if dimension_actual != dimension_allowed: 
                print(f'Warning: The variable {name} in {folder}/Logs.json has dimensions {dimension_actual} and plot '
                      f'type {log[KEY_PLOTTYPE]} with Framestamps={log[KEY_FRAMESTAMPS]}, which allows only entries '
                      f'with dimension {dimension_allowed}. The log for {name} will not be visualized.')

            if dimension_actual != dimension_allowed or log[KEY_PLOTTYPE] == 'Empty':
                cache.append(name)

        for key in cache:
            del logs[key]

        experiments.append((logs, info))

    return {info[KEY_METHOD_NAME]: {
        KEY_SHORT_DESCR: info[KEY_SHORT_DESCR],
        KEY_LONG_DESSCR: info[KEY_LONG_DESSCR],
        KEY_RUNTIME: info[KEY_RUNTIME],
        KEY_NOTES: info[KEY_NOTES],
        KEY_HYPERPARAMETERS: info[KEY_HYPERPARAMETERS],
        'logs': log
    } for (log, info) in experiments}


def visualize(title, data):
    streamlit.markdown('''
    ## Runtime: {}
    ## Description
    {}
    ## Notes
    {}
    '''.format(data[KEY_RUNTIME], data[KEY_LONG_DESSCR], data[KEY_NOTES]))

    with streamlit.beta_expander('Hyperparameters'):
        streamlit.write(data[KEY_HYPERPARAMETERS])

    functions = {
        'line': line,
        'histogram': histogram,
        'histogram2d': histogram2d,
        'scatter': scatter,
        'tube': tube
    }

    with_slider = ['histogram', 'histogram2d']

    for name, log in data['logs'].items():
        fn = functions[log[KEY_PLOTTYPE]]  # see json logger for key

        ident = name
        ident_slider_episodes = name + 'with episode slider'
        ident_slider_frames = name + 'with frame slider'

        logs = log[KEY_VALUES]
        variables = list(zip(*log[KEY_VALUES]))

        if log[KEY_FRAMESTAMPS] == 'True':
            frames, variables = variables[0], variables[1:]

        # normal plots

        with streamlit.beta_expander(ident):
            figure = fn(*variables)
            streamlit.altair_chart(figure, use_container_width=FILL_BROWSER_WIDTH)

        # episode slider plots

        if log[KEY_PLOTTYPE] in with_slider:
            with streamlit.beta_expander(ident_slider_episodes):
                no_partitions = len(logs) // 100
                partitions = numpy.array_split(logs, no_partitions)
                partitions = [list(zip(*p))[1:] for p in partitions]  # [1:] skips the frames in [0]

                slider = streamlit.slider(f'{ident} -- episodes * {len(partitions[0])}', 0, len(partitions) - 1)
                figure = fn(*partitions[slider])
                streamlit.altair_chart(figure, use_container_width=FILL_BROWSER_WIDTH)

        # frame slider plots

        if log[KEY_FRAMESTAMPS] == 'True':
            max_frame = max(frames)
            no_buckets = 10
            size_buckets = (max_frame // no_buckets) + 1
            buckets = [[] for _ in range(no_buckets)]

            for entry in logs:
                bucket = int((entry[0] // size_buckets)) - 1
                buckets[bucket].append(entry[1:])

            buckets = [list(zip(*b)) for b in buckets]

            with streamlit.beta_expander(ident_slider_frames):
                slider = streamlit.slider(f'{name} -- frame buckets of size {size_buckets}', 0, no_buckets - 1)
                figure = fn(*buckets[slider])
                streamlit.altair_chart(figure, use_container_width=FILL_BROWSER_WIDTH)


def line(y, name='y'):
    frame = pandas.DataFrame({
        'episodes': numpy.linspace(0, len(y), len(y)),
        name: numpy.array(y)
    })

    return altair.Chart(frame).mark_line().encode(x='episodes', y=name)


def histogram(x, name='x'):
    frame = pandas.DataFrame({
        name: numpy.array(x),
    })

    return altair.Chart(frame).mark_bar().encode(x=altair.X(name + '', bin=True), y='count()')


def histogram2d(x, y, x_name='x', y_name='y'):
    frame = pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })

    plot = altair.Chart(frame).mark_circle().encode(
        altair.X(x_name, bin=True),
        altair.Y(y_name, bin=True),
        size='count()'
    ).interactive()

    return plot


def scatter(x, y, x_name='x', y_name='y'):
    frame = pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })

    plot = altair.Chart(frame).mark_circle(size=60).encode(
        x=x_name,
        y=y_name,
        color='Group',
        tooltip=['Name', 'Group', x_name, y_name]
    ).interactive()

    return plot


def tube(x, y, x_name='x', y_name='y'):
    x_array = numpy.array(x)
    y_array = numpy.array(y)

    frame = pandas.DataFrame({
        'episodes': numpy.linspace(0, len(x), len(x)),
        x_name: x_array,
        'lower': x_array - y_array,
        'upper': x_array + y_array
    })

    line = altair.Chart(frame).mark_line().encode(
        x='episodes',
        y=x_name
    )

    band = altair.Chart(frame).mark_area(opacity=0.5).encode(
        x='episodes',
        y='lower',
        y2='upper'
    )

    return band + line


main()
