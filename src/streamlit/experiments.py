import streamlit
import numpy
import json
import pandas
import matplotlib.pyplot
import matplotlib.axes._subplots
import matplotlib.figure
import seaborn
import os
import re
import numbers


LAYOUT = 'centered'  # change this, options are wide and centered
PATH = './'  # change this, this is the path to the folder containing the experiments


# keys used in the json log

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
            figure = fn(title + ident, *variables)
            streamlit.pyplot(figure)

        # episode slider plots

        if log[KEY_PLOTTYPE] in with_slider:
            with streamlit.beta_expander(ident_slider_episodes):
                no_partitions = len(logs) // 100
                partitions = numpy.array_split(logs, no_partitions)
                partitions = [list(zip(*p))[1:] for p in partitions]  # [1:] skips the frames in [0]

                slider = streamlit.slider(f'{ident} -- episodes * {len(partitions[0])}', 0, len(partitions) - 1)
                figure = fn(title + ident_slider_episodes + str(slider), *partitions[slider])
                streamlit.pyplot(figure)

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
                figure = fn(title + ident_slider_frames + str(slider), *buckets[slider])
                streamlit.pyplot(figure)


# why is streamlit this stupid?
disable_hashing_on = {
    tuple: (lambda t: hash(t[0])),

    pandas.DataFrame: (lambda _: None),
    numpy.ndarray: (lambda _: None),
    matplotlib.figure.Figure: (lambda _: None),
    matplotlib.axes._subplots.SubplotBase: (lambda _: None),
    seaborn.axisgrid.JointGrid: (lambda _: None),
    matplotlib.axes.Axes: (lambda _: None)
}


@streamlit.cache(allow_output_mutation=True, hash_funcs=disable_hashing_on)
def line(ident, y, name=''):
    frame = pandas.DataFrame({
        'episodes': len(y),
        'name': y
    })
    episodes = len(y)
    x = numpy.linspace(0, episodes, episodes)
    figure, axis = matplotlib.pyplot.subplots()
    axis.plot(x, y, '-')

    return figure


@streamlit.cache(allow_output_mutation=True, hash_funcs=disable_hashing_on)
def histogram(ident, x, name=''):
    # frame = pandas.DataFrame({
    #     name: x
    # })

    # matplotlib.pyplot.figure()
    # plot = seaborn.histplot(frame, x=name)
    # figure = plot.get_figure()
    abc = ident
    figure, axis = matplotlib.pyplot.subplots()
    axis.hist(x)

    return figure


@streamlit.cache(allow_output_mutation=True, hash_funcs=disable_hashing_on)
def histogram2d(ident, x, y, x_name='', y_name=''):
    frame = pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })

    matplotlib.pyplot.figure()
    plot = seaborn.jointplot(data=frame, x=x_name, y=y_name, kind='hex')
    figure = plot.fig

    return figure


@streamlit.cache(allow_output_mutation=True, hash_funcs=disable_hashing_on)
def scatter(ident, x, y, x_name='', y_name=''):
    frame = pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })

    matplotlib.pyplot.figure()
    plot = seaborn.scatterplot(data=frame, x=x_name, y=y_name)
    figure = plot.get_figure()

    return figure


@streamlit.cache(allow_output_mutation=True, hash_funcs=disable_hashing_on)
def tube(ident, x, y):
    episodes = len(x)
    mean = numpy.array(x)
    std = numpy.array(y)
    upper = mean + std
    lower = mean - std

    print('RERUN?')

    x = numpy.linspace(0, episodes, episodes)
    figure, axis = matplotlib.pyplot.subplots()

    print(type(axis))

    axis.plot(x, mean, '-', alpha=1)
    axis.fill_between(x, upper, lower, alpha=0.5)

    return figure


main()
