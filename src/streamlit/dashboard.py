import streamlit
import altair
import numpy
import json
import pandas
import os
import math
import statistics
import numbers

#
# Run this script with "streamlit run dashboard.py" from the directory the script is located in. Change the parameters
# below. Depending on the plot type the tuples in the list entry of 'Values' can only be of certain length. If
# framestamps is set to 'True', one additional dimension is allowed. If you dont actually want to plot your log you
# should use plot type Empty.
#

LAYOUT = 'wide'  # change this, options are wide and centered
PATH = './'  # change this, this is the path to the folder containing the experiments
FILL_BROWSER_WIDTH = True  # iff true, the plots will expand to the full length of your browser window
values_per_figure = 1000  # mutated by slider

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
KEY_LENGTH = 'Length'


def main():
    global values_per_figure

    streamlit.set_page_config(layout=LAYOUT)

    experiment_folders = [os.path.basename(f.path) for f in os.scandir(PATH) if f.is_dir()]
    experiment_chosen = streamlit.sidebar.selectbox('Choose an experiment!', experiment_folders)
    values_per_figure = streamlit.sidebar.number_input('Values Per Plot', 1000, 10000, 1000, 1000)

    streamlit.title(experiment_chosen)

    name, data = load(experiment_chosen)
    smoothed_logs_partitioning = smooth(data)

    # to allow streamlit caching of load() mutating the data dict directly is not possible :( its a
    # partitioning, e.g. instead of log its [log], since this is uniform with the possible partitionings
    # we create when using the frame or episode sliders. we put it inside a list in load because it might
    # be computationally expensive and we want to avoid doing it possibly multiple times in visualize()

    visualize(data, smoothed_logs_partitioning)


@streamlit.cache
def load(folder):
    if not os.path.exists(os.path.join(folder, 'Info.json')) or not os.path.exists(
            os.path.join(folder, 'Logs.json')):
        print(f'Error: Folder {folder} does not contain Info.json or Logs.json and will be omitted.')

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

    for key in required_keys:
        if key not in info.keys():
            info[key] = ''

    # if not all((k in info.keys() for k in required_keys)):
    #     print(f'Error: {folder} does not contain all required data {required_keys} and will be omitted. If you '
    #           f'already run your experiment, add the entries manually to the json file.')
    #     continue
    #
    # if not basename == info[KEY_METHOD_NAME]:
    #     print(f'Error: Folder is named {folder} and method is named {info["MethodName"]}, please stick to the same'
    #           f'naming convention. Suggestions to change the naming convention are welcome. The folder will be '
    #           f'omitted.')

    allowed_dimensions = {
        'line': 1,
        'histogram': 1,
        'histogram2d': 2,
        'scatter': 2,
        'tube': 2,
        'Empty': 999999999
    }

    to_delete = []

    for name, log in logs.items():
        if len(log[KEY_VALUES]) == 0:
            print(f'Warning: Found empty log {name}.')
            to_delete.append(name)
            continue

        if not type(log[KEY_VALUES][0]) == list:
            # print(f'Warning: Non-tuple type in value log of {name} in {folder}/Logs.json. The entries will be '
            #       f'interpreted as 1-dimensional tuples.')

            try:
                for i in range(len(log[KEY_VALUES])):
                    log[KEY_VALUES][i] = [log[KEY_VALUES][i]]
            except Exception as e:
                print(f'Error: Interpreting entries as 1-dimensional tuples failed, the log will be omitted. '
                      f'Message: {e}')
                to_delete.append(name)
                continue

        if not isinstance(log[KEY_VALUES][0][0], numbers.Number):
            print(f'Warning: Non-number type in value log of {name} in {folder}/Logs.json, found type '
                  f'{type(log[KEY_VALUES][0][0])} instead. Log will be omitted.')
            to_delete.append(name)
            continue

        dimension_actual = len(log[KEY_VALUES][0])
        dimension_allowed = allowed_dimensions[log[KEY_PLOTTYPE]]

        if log[KEY_FRAMESTAMPS]:  # assumed to be the first dimension
            dimension_allowed += 1

        if dimension_actual != dimension_allowed:
            print(f'Warning: The variable {name} in {folder}/Logs.json has dimensions {dimension_actual} and plot '
                  f'type {log[KEY_PLOTTYPE]} with Framestamps={log[KEY_FRAMESTAMPS]}, which allows only entries '
                  f'with dimension {dimension_allowed}. The log for {name} will not be visualized.')

        if dimension_actual != dimension_allowed or log[KEY_PLOTTYPE] == 'Empty':
            to_delete.append(name)
            continue

        # logs contain lists of values for each time step, we need lists of time steps for each value
        log[KEY_VALUES] = list(zip(*log[KEY_VALUES]))

        # later on we treat this log as partitioning with a single partition, i.e. we put it in a list. to avoid
        # doing this potentially multiple times, we do it here once
        log[KEY_VALUES] = [log[KEY_VALUES]]

    for key in to_delete:
        del logs[key]

    return info[KEY_METHOD_NAME], {
        KEY_SHORT_DESCR: info[KEY_SHORT_DESCR],
        KEY_LONG_DESSCR: info[KEY_LONG_DESSCR],
        KEY_RUNTIME: info[KEY_RUNTIME],
        KEY_NOTES: info[KEY_NOTES],
        KEY_HYPERPARAMETERS: info[KEY_HYPERPARAMETERS],
        'logs': logs
    }


@streamlit.cache
def smooth(data):
    smoothed_logs_partitioning = {}

    for name, log in data['logs'].items():
        partitioning = log[KEY_VALUES]
        no_datapoints = len(partitioning[0][0])  # some variable of single complete partition
        log[KEY_LENGTH] = no_datapoints

        if not no_datapoints > values_per_figure:
            log[KEY_VALUES] = log[KEY_VALUES]
            continue

        smoothed_logs_partitioning[name] = [[]]  # partitioning with one empty partition
        smoothing_window = no_datapoints // values_per_figure

        for v, variable in enumerate(partitioning[0]):
            log[KEY_VALUES][0].append([])

            for i in range(values_per_figure):
                index = i * smoothing_window
                mean = statistics.mean(variable[index:index + smoothing_window])
                smoothed_logs_partitioning[name][0][v].append(mean)

    return smoothed_logs_partitioning


def visualize(data, smoothed_logs_partitioning):
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

    for name, log in data['logs'].items():
        streamlit.markdown('''## {}'''.format(name))

        fn = functions[log[KEY_PLOTTYPE]]  # see json logger for key
        partitioning = smoothed_logs_partitioning[name][KEY_VALUES]  # partitioning with a single partition

        if log[KEY_FRAMESTAMPS]:
            frames = partitioning[0][0]
            del partitioning[0][0]

        variables = partitioning[0]
        no_episodes = len(variables[0])

        # if fn in ['histogram', 'histogram2d'] and streamlit.checkbox('Show episode slider'):
        #     no_buckets = min(100, no_episodes)
        #     sizeof_buckets = max(no_episodes // no_buckets, 1)
        #     chosen_bucket = streamlit.slider(f'{name}: Choose one of {no_buckets}', 0, no_buckets - 1)
        #
        # else:

        sizeof_buckets = no_episodes
        chosen_bucket = 0
        partitioning = partition(partitioning, variables, no_episodes, sizeof_buckets)

        if not [*partitioning[chosen_bucket]]:
            streamlit.write('No data for this partition, how can this happen?')

        # hier koennte man die frame slider einbauen, der dann nur auf der partition funktioniert

        # max_frame = max(frames)
        # no_buckets = 10
        # size_buckets = (max_frame // no_buckets) + 1
        # buckets = [[] for _ in range(no_buckets)]
        #
        # for entry in logs:
        #     bucket = int((entry[0] // size_buckets)) - 1
        #     buckets[bucket].append(entry[1:])

        # streamlit creates all this once and never again (what is caching for?)

        else:
            figure = fn(*partitioning[chosen_bucket])
            streamlit.altair_chart(figure, use_container_width=FILL_BROWSER_WIDTH)


@streamlit.cache
def partition(partitioning, variables, no_episodes, sizeof_buckets):
    if math.ceil(no_episodes / sizeof_buckets) == len(partitioning):
        return partitioning

    partitioning = []

    for i in range(no_episodes):
        if i % sizeof_buckets == 0:
            partitioning.append([[] for _ in range(len(variables))])

        for j in range(len(variables)):
            partitioning[-1][j].append(variables[j][i])

    return partitioning


@streamlit.cache
def build_line_dataframe(y, name):
    return pandas.DataFrame({
        'episodes': numpy.linspace(0, len(y), len(y)),
        name: numpy.array(y)
    })


def line(y, name='y'):
    frame = build_line_dataframe(y, name)
    return altair.Chart(frame).mark_line().encode(x='episodes', y=name)


@streamlit.cache
def build_histogram_dataframe(x, name):
    return pandas.DataFrame({
        name: numpy.array(x),
    })


def histogram(x, name='x'):
    frame = build_histogram_dataframe(x, name)
    return altair.Chart(frame).mark_bar().encode(x=altair.X(name + '', bin=True), y='count()')


@streamlit.cache
def build_histogram2d_dataframe(x, y, x_name, y_name):
    return pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })


def histogram2d(x, y, x_name='x', y_name='y'):
    frame = build_histogram2d_dataframe(x, y, x_name, y_name)

    # plot = altair.Chart(frame).mark_circle().encode(
    #     altair.X(x_name, bin=True),
    #     altair.Y(y_name, bin=True),
    #     size='count()'
    # ).interactive()

    plot = altair.Chart(frame).mark_rect().encode(
        altair.X(x_name, bin=altair.Bin(maxbins=60)),
        altair.Y(y_name, bin=altair.Bin(maxbins=40)),
        altair.Color('count()', scale=altair.Scale(scheme='greenblue'))
    )

    return plot


@streamlit.cache
def build_scatter_dataframe(x, y, x_name, y_name):
    return pandas.DataFrame({
        x_name: numpy.array(x),
        y_name: numpy.array(y)
    })


def scatter(x, y, x_name='x', y_name='y'):
    frame = build_scatter_dataframe(x, y, x_name, y_name)

    plot = altair.Chart(frame).mark_circle(size=60).encode(
        x=x_name,
        y=y_name,
        color='Group',
        tooltip=['Name', 'Group', x_name, y_name]
    ).interactive()

    return plot


@streamlit.cache
def build_tube_dataframe(x, y, x_name, y_name):
    x_array = numpy.array(x)
    y_array = numpy.array(y)

    return pandas.DataFrame({
        'episodes': numpy.linspace(0, len(x), len(x)),
        x_name: x_array,
        'lower': x_array - y_array,
        'upper': x_array + y_array
    })


def tube(x, y, x_name='x', y_name='y'):
    frame = build_tube_dataframe(x, y, x_name, y_name)

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
