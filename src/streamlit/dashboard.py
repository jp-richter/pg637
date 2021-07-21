import streamlit
import altair
import numpy
import json
import pandas
import os
import statistics
import numbers
import math
import timeit


LAYOUT = 'wide'  # options are wide and centered, has effect on plot sitze
PATH = './'  # this is the path to the folder containing the experiments
FILL_BROWSER_WIDTH = True  # iff true, the plots will expand to the full length of your browser window
NO_EPISODE_BUCKETS = 10
NO_VALUES_PER_VARIABLE = 1000  # compression factor, number of values per variable per plot
QUANTILE_UPPER = 0.95
QUANTILE_LOWER = 0.5


# Run this script with "streamlit run dashboard.py" from the directory the script is located in. Change the parameters
# above. Depending on the plot type the tuples in the list entry of 'Values' can only be of certain length. If
# framestamps is set to 'True', one additional dimension is allowed. If you dont actually want to plot your log you
# should use plot type Empty.

# This script constructs a dict with all necessary data from the logs and runs it through a processing pipeline with
# steps for sanitizing and smoothing the dict keys and time series. The resulting dictionary can be seen below. Keys
# are always present, regardless if the entry contains data for consistency reasons.
#
# {
#         KEY_METHOD_NAME: str,
#         KEY_SHORT_DESCR: str,
#         KEY_LONG_DESSCR: str,
#         KEY_RUNTIME: str,
#         KEY_NOTES: str,
#         KEY_HYPERPARAMETERS: dict,
#
#         KEY_LOGS_RAW: {
#             KEY_FRAMESTAMPS: bool,
#             KEY_VALUES: list,
#             KEY_FRAMESTAMP_VALUES: list,
#             KEY_X_AXIS: str,
#             KEY_Y_AXIS: str,
#             KEY_LENGTH: int,
#             KEY_COMPRESSION: int
#         }
#
#         KEY_LOGS_PROCESSED: {
#             KEY_FRAMESTAMPS: bool,
#             KEY_VALUES: list,
#             KEY_X_AXIS: str,
#             KEY_Y_AXIS: str,
#             KEY_LENGTH: int
#             KEY_COMPRESSION: int,
#             KEY_QUANTILE_UPPER: list,
#             KEY_QUANTILE_LOWER: list
#         }
# }

# dont change anything below

# info keys
KEY_METHOD_NAME = 'MethodName'
KEY_SHORT_DESCR = 'ShortDescription'
KEY_LONG_DESSCR = 'LongDescription'
KEY_NOTES = 'Notes'
KEY_RUNTIME = 'Runtime'
KEY_HYPERPARAMETERS = 'Hyperparameter'
KEY_LOGS_RAW = 'Logs Raw'
KEY_LOGS_PROCESSED = 'Logs Preprocessed'

# log keys
KEY_VALUES = 'Values'
KEY_FRAMESTAMPS = 'Framestamps'
KEY_FRAMESTAMP_VALUES = 'Framestamp Values'
KEY_PLOTTYPE = 'Plot Type '
KEY_LENGTH = 'Length'
KEY_COMPRESSION = 'Compression'
KEY_QUANTILE_UPPER = 'Upper Quantile Values'
KEY_QUANTILE_LOWER = 'Lower Quantile Values'
KEY_X_AXIS = 'X Axis Name'
KEY_Y_AXIS = 'Y Axis Name'
KEY_DIMENSIONS = 'Dim'
KEY_UNIQUE_FRAMES = 'Unique Frames'

HELP_MESSAGE = '''
    * Logs with to many data points will be compressed to 1000 values per variable. Compression is done by taking the
    mean for line plots, class modus of fixed 100 classes for histograms. The upper and lower line in line plots mark 
    the quantiles for p=5 and p=95 of raw values (if I ever add this feature).
    * Please be aware, that the compression of histograms DISTORTS THE DATA A LOT! I am still working on a way, to 
    prevent this. If you have any good idea, feel free to make suggestions.
    * You can still download high resolution images with the button next to the plots. You can also download the 
    smoothed plots directly at the three dots at each plot.
    * To save plots you need to install some stuff, see https://github.com/altair-viz/altair_saver/ for .svg
    files. The plots will be saved to the current directory. Until I refactored it to use a new thread it will
    block streamlit for the runtime though.
    * You can change the width of the plots with FILL_BROWSER_WIDTH in the script. This has an effect on the 
    plot size. For presentable plots consider FILL_BROWSER_WIDTH = False. You might have to restart! You can
    also chose the LAYOUT as 'wide' or 'centered'.
    * Note that you can always view the plots fullscreen with the arrow next to them. This is the size of your
    browser window. This way you have complete control over plot sizes.
    * Consider chosing plot type EMPTY for unneeded plots, since it speeds up the loading times.
    * If you get any errors when the folder contains a preprocessed log from older versions try deleting the
    preprocessed log, since this script won't trigger the preprocessing step if this file is present.
'''


def main():
    streamlit.set_page_config(layout=LAYOUT)

    experiment_folders = [os.path.basename(f.path) for f in os.scandir(PATH) if f.is_dir()]
    experiment_chosen = streamlit.sidebar.selectbox('Choose an experiment!', experiment_folders)

    with streamlit.sidebar.beta_expander('Click here for some info and tipps!'):
        streamlit.markdown(HELP_MESSAGE)

    streamlit.title(experiment_chosen)

    data = load('00405-7-PPOCoDeg-MultipleAnglesAndTwoActions-2')  # see at the top of the script for doc
    visualize(data)


@streamlit.cache
def load(folder):
    data, is_preprocessed = preprocess_load(folder)

    if not is_preprocessed:
        print(f'PRE-PROCESSING {folder}..')

        preprocess_check_validity(data)
        preprocess_sanitize_keys(data)
        preprocess_translate_logs(data)
        preprocess_extract_framestamps(data)
        preprocess_remove_framestamp_outlier(data)
        preprocess_smooth_logs(data)
        preprocess_save(data, folder)

    return data


def preprocess_load(folder):
    if os.path.exists(os.path.join(folder, 'Preprocessed.json')):
        print(f'FOUND PRE-PROCESSED LOG FILE FOR {folder}, SKIPPING PRE-PROCESSING STEP')
        with open(os.path.join(folder, 'Preprocessed.json'), 'r') as file:
            data = json.load(file)

        return data, True

    if not os.path.exists(os.path.join(folder, 'Info.json')) or not os.path.exists(
            os.path.join(folder, 'Logs.json')):
        print(f'Error: Folder {folder} does not contain Info.json or Logs.json and will be omitted.')

    with open(os.path.join(folder, 'Info.json'), 'r') as file:
        info = json.load(file)

    with open(os.path.join(folder, 'Logs.json'), 'r') as file:
        logs = json.load(file)

    info[KEY_LOGS_RAW] = logs

    return info, False


def preprocess_check_validity(data):
    to_delete = []

    break_conditions = [
        break_on_empty_log,
        break_on_non_tuple_type,
        break_on_non_number_input,
        break_on_wrong_dimensions
    ]

    for name, log in data[KEY_LOGS_RAW].items():
        for condition in break_conditions:
            if condition(name, log):
                to_delete.append(name)
                break

    for key in to_delete:
        del data[KEY_LOGS_RAW][key]


def break_on_empty_log(name, log):
    if len(log[KEY_VALUES]) == 0:
        print(f'Warning: Found empty log {name}.')
        return True

    return False


def break_on_non_tuple_type(name, log):
    if not type(log[KEY_VALUES][0]) == list:
        # print(f'Warning: Non-tuple type in value log of {name} in {folder}/Logs.json. The entries will be '
        #       f'interpreted as 1-dimensional tuples.')

        try:
            for i in range(len(log[KEY_VALUES])):
                log[KEY_VALUES][i] = [log[KEY_VALUES][i]]
        except Exception as e:
            print(f'Error: Interpreting entries as 1-dimensional tuples failed, the log will be omitted. '
                  f'Message: {e}')
            return True

    return False


def break_on_non_number_input(name, log):
    if not isinstance(log[KEY_VALUES][0][0], numbers.Number):
        print(f'Warning: Non-number type in value log of {name}, found type '
              f'{type(log[KEY_VALUES][0][0])} instead. Log will be omitted.')
        return True

    return False


allowed_dimensions = {
    'line': 1,
    'histogram': 1,
    'histogram2d': 2,
    'scatter': 2,
    'tube': 2,
    'Empty': 999999999
}


def break_on_wrong_dimensions(name, log):
    dimension_allowed = allowed_dimensions[log[KEY_PLOTTYPE]]
    actual_dimension = len(log[KEY_VALUES][0])

    if log[KEY_FRAMESTAMPS]:
        dimension_allowed += 1

    if actual_dimension != dimension_allowed:
        print(f'Warning: The variable {name} has dimensions {actual_dimension} and plot '
              f'type {log[KEY_PLOTTYPE]} with Framestamps={log[KEY_FRAMESTAMPS]}, which allows only entries '
              f'with dimension {dimension_allowed}. The log for {name} will not be visualized.')

    if actual_dimension != dimension_allowed or log[KEY_PLOTTYPE] == 'Empty':
        return True

    return False


def preprocess_sanitize_keys(data):
    required_info_keys = [
        KEY_METHOD_NAME,
        KEY_SHORT_DESCR,
        KEY_LONG_DESSCR,
        KEY_RUNTIME,
        KEY_NOTES,
        KEY_HYPERPARAMETERS
    ]

    for key in required_info_keys:
        if key not in data.keys():
            data[key] = ''

    data[KEY_LOGS_PROCESSED] = dict()

    for log in data[KEY_LOGS_RAW].values():
        log[KEY_LENGTH] = len(log[KEY_VALUES])
        log[KEY_DIMENSIONS] = len(log[KEY_VALUES][0])
        log[KEY_FRAMESTAMP_VALUES] = []
        log[KEY_COMPRESSION] = 1

        if log[KEY_FRAMESTAMPS]:
            log[KEY_DIMENSIONS] -= 1

        if 'Names' in log.keys():
            log[KEY_X_AXIS] = log['Names'][0]

            if len(log['Names']) > 1:
                log[KEY_Y_AXIS] = log['Names'][1]

        if KEY_X_AXIS not in log.keys():
            log[KEY_X_AXIS] = 'x'

        if KEY_Y_AXIS not in log.keys():
            log[KEY_Y_AXIS] = 'y'


def preprocess_translate_logs(data):
    for log in data[KEY_LOGS_RAW].values():
        log[KEY_VALUES] = list(zip(*log[KEY_VALUES]))


def preprocess_extract_framestamps(data):
    for log in data[KEY_LOGS_RAW].values():
        if log[KEY_FRAMESTAMPS]:
            log[KEY_FRAMESTAMP_VALUES] = log[KEY_VALUES][0]
            log[KEY_VALUES] = log[KEY_VALUES][1:]


def preprocess_remove_framestamp_outlier(data):
    for name, log in data[KEY_LOGS_RAW].items():
        if not log[KEY_FRAMESTAMPS]:
            continue

        unique_frames = list(set(log[KEY_FRAMESTAMP_VALUES]))
        unique_frame_count = [0 for _ in unique_frames]

        for frame in log[KEY_FRAMESTAMP_VALUES]:
            unique_frame_count[unique_frames.index(frame)] += 1

        outlier = []

        for count, unique_frame in zip(unique_frames, unique_frame_count):
            if count < max(unique_frame_count):
                outlier.append(unique_frame)

        to_remove = []

        for i in range(len(log[KEY_VALUES])):
            if log[KEY_FRAMESTAMP_VALUES] in outlier:
                to_remove.append(i)

        if to_remove:
            print(f'Found frame outliers in {name}: {to_remove}')

        for index in to_remove:
            del log[KEY_VALUES][index]
            del log[KEY_FRAMESTAMP_VALUES][index]


def preprocess_smooth_logs(data):
    for name, log in data[KEY_LOGS_RAW].items():
        if log[KEY_LENGTH] < NO_VALUES_PER_VARIABLE:
            data[KEY_LOGS_PROCESSED][name] = log
            continue

        sliding_window = log[KEY_LENGTH] // NO_VALUES_PER_VARIABLE

        copy = {
            KEY_VALUES: [[] for _ in range(len(log[KEY_VALUES]))],
            KEY_QUANTILE_UPPER: [[] for _ in range(len(log[KEY_VALUES]))],
            KEY_QUANTILE_LOWER: [[] for _ in range(len(log[KEY_VALUES]))],
            KEY_FRAMESTAMPS: log[KEY_FRAMESTAMPS],
            KEY_FRAMESTAMP_VALUES: list(log[KEY_FRAMESTAMP_VALUES]),
            KEY_PLOTTYPE: log[KEY_PLOTTYPE],
            KEY_X_AXIS: log[KEY_X_AXIS],
            KEY_Y_AXIS: log[KEY_Y_AXIS],
            KEY_COMPRESSION: sliding_window
        }

        if log[KEY_FRAMESTAMPS]:
            unique_frames = set(log[KEY_FRAMESTAMP_VALUES])
            copy[KEY_UNIQUE_FRAMES] = list(unique_frames)
            splitter = len(unique_frames)
        else:
            splitter = 1  # equals no split

        for v, variable in enumerate(log[KEY_VALUES]):
            for i in range(NO_VALUES_PER_VARIABLE):
                index = i * sliding_window

                window_for_frame = variable[index:][::splitter]
                window_for_frame = window_for_frame[:min(sliding_window, len(window_for_frame))]

                mean = statistics.mean(window_for_frame)
                copy[KEY_VALUES][v].append(mean)

                if log[KEY_FRAMESTAMPS]:
                    copy[KEY_FRAMESTAMP_VALUES].append(log[KEY_FRAMESTAMP_VALUES][i])

                upper, lower = numpy.quantile(
                    variable[index:index + sliding_window],
                    [QUANTILE_UPPER, QUANTILE_LOWER])
                copy[KEY_QUANTILE_UPPER][v].append(upper)
                copy[KEY_QUANTILE_LOWER][v].append(lower)

        copy[KEY_LENGTH] = len(copy[KEY_VALUES][0])
        data[KEY_LOGS_PROCESSED][name] = copy


def preprocess_save(data, folder):
    with open(os.path.join(folder, 'Preprocessed.json'), 'w') as file:
        json.dump(data, file)


def visualize(data):
    streamlit.markdown('''## Runtime: {}'''.format(data[KEY_RUNTIME]))

    with streamlit.beta_expander('Description'):
        streamlit.write(data[KEY_LONG_DESSCR])

    with streamlit.beta_expander('Notes'):
        streamlit.write(data[KEY_NOTES])

    with streamlit.beta_expander('Hyperparameters'):
        streamlit.write(data[KEY_HYPERPARAMETERS])

    for idx, (name, log) in enumerate(data[KEY_LOGS_PROCESSED].items()):
        streamlit.markdown('''## {}'''.format(name))

        slider_episodes = False
        slider_frames = False

        c1, c2, c3, c4 = streamlit.beta_columns(4)

        if c1.button(f'Download High Resolution ID{idx}'):
            download_high_res(name, data[KEY_LOGS_RAW][name])

        if c2.checkbox(f'Episode Slider ID{idx}'):  # if plot type in ['histogram', 'histogram2d']
            slider_episodes = True

        if c3.checkbox(f'Frame Slider ID{idx}'):
            slider_frames = True
            slider_episodes = False

        c4.markdown('''Compression Factor: x{}'''.format(log[KEY_COMPRESSION]))

        figure = compute_figure(name, log, slider_episodes, slider_frames)

        if figure:
            streamlit.altair_chart(figure, use_container_width=FILL_BROWSER_WIDTH)
        else:
            streamlit.write('No data for this partition, how can this happen?')


def compute_figure(name, log, slider_episodes, slider_frames):
    functions = {
        'line': line,
        'histogram': histogram,
        'histogram2d': histogram2d,
        'scatter': scatter,
        'tube': tube
    }

    fn = functions[log[KEY_PLOTTYPE]]  # see json logger for key

    if slider_episodes:
        buckets_size = max(log[KEY_LENGTH] // NO_EPISODE_BUCKETS, 1)
        bucket_chosen = streamlit.slider(f'{name}: Choose one of {NO_EPISODE_BUCKETS}', 0, NO_EPISODE_BUCKETS - 1)

    else:
        buckets_size = log[KEY_LENGTH]
        bucket_chosen = 0

    partitioning = partition(log[KEY_VALUES], log[KEY_LENGTH], buckets_size)

    if not [*partitioning[bucket_chosen]]:
        streamlit.write('This bucket seems to be empty..')
        return None

    if slider_frames:
        if slider_episodes:
            streamlit.write('Please disable episode slider!')
            return None

        if not log[KEY_FRAMESTAMPS]:
            streamlit.write('No Framestamps found for this log..')
            return None

        log[KEY_UNIQUE_FRAMES].sort()
        frame_chosen = streamlit.selectbox(f'{name}: Choose a frame', log[KEY_UNIQUE_FRAMES])

        result = []
        for i in range(len(partitioning[bucket_chosen][0])):
            if log[KEY_FRAMESTAMP_VALUES][i] == frame_chosen:
                result.append(partitioning[bucket_chosen][0][i])

        partitioning[bucket_chosen][0] = result
        # TODO test this

    return fn(*partitioning[bucket_chosen], x_name=log[KEY_X_AXIS], y_name=log[KEY_Y_AXIS])


@streamlit.cache
def partition(variables, no_values_per_variable, sizeof_buckets):
    if no_values_per_variable == sizeof_buckets:
        return [variables]

    partitioning = []

    for i in range(no_values_per_variable):
        if i % sizeof_buckets == 0:
            partitioning.append([[] for _ in range(len(variables))])

        for j in range(len(variables)):
            partitioning[-1][j].append(variables[j][i])

    return partitioning


def download_high_res(name, raw_log):
    figure = compute_figure(name, raw_log, False, False)
    figure.save(f'{name}.svg', scale_factor=1.0)


@streamlit.cache
def build_line_dataframe(y, x_name, y_name):
    return pandas.DataFrame({
        x_name: numpy.linspace(0, len(y), len(y)),
        y_name: numpy.array(y)
    })


def line(y, x_name='x', y_name='y'):
    frame = build_line_dataframe(y, x_name, y_name)
    return altair.Chart(frame).mark_line().encode(x=x_name, y=y_name)


@streamlit.cache
def build_histogram_dataframe(x, name):
    return pandas.DataFrame({
        name: numpy.array(x),
    })


def histogram(x, x_name='x', y_name='y'):
    frame = build_histogram_dataframe(x, x_name)
    return altair.Chart(frame).mark_bar().encode(x=altair.X(x_name + '', bin=True), y='count()')


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
    tube_array = numpy.array(y)

    return pandas.DataFrame({
        x_name: numpy.linspace(0, len(x), len(x)),
        y_name: x_array,
        'lower': x_array - tube_array,
        'upper': x_array + tube_array
    })


def tube(x, y, x_name='x', y_name='y'):
    frame = build_tube_dataframe(x, y, x_name, y_name)

    line = altair.Chart(frame).mark_line().encode(
        x=x_name,
        y=y_name
    )

    band = altair.Chart(frame).mark_area(opacity=0.5).encode(
        x=x_name,
        y='lower',
        y2='upper'
    )

    return band + line


main()
