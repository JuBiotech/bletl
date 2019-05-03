"""Parsing functions for the BioLector Pro"""
import collections
import datetime
import io
import logging
import pandas

from .. import core
from .. import utils


logger = logging.getLogger('blpro')


class BioLectorProParser(core.BLDParser):
    def parse(self, filepath):
        metadata, data = parse_metadata_data(filepath)

        bld = core.BLData(
            model=core.BioLectorModel.BLPro,
            environment=extract_environment(data),
            filtersets=extract_filtersets(metadata),
            references=extract_references(data),
            measurements=extract_measurements(data),
            comments=extract_comments(data),
        )

        bld.metadata = metadata
        bld.fluidics = extract_fluidics(data)
        bld.valves, bld.module = extract_valves_module(data)
        bld.diagnostics = extract_diagnostics(data)

        for key, fts in transform_into_filtertimeseries(bld.metadata, bld.measurements, bld.filtersets):
            bld[key] = fts

        return bld


def parse_metadata_data(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    metadata = collections.defaultdict(dict)
    datalines = collections.defaultdict(list)
    section = None
    data_start = None
    data_end = None

    for l, line in enumerate(lines):
        if line != '\n':
            if line.startswith('='):
                # the first section after the data table...
                if data_start is not None:
                    data_end = l
                # any section header encountered
                section = line.strip().strip('=').strip()
                if not data_start and section == 'data':
                    data_start = l + 1
            elif line.startswith('['):
                # register the value
                key, value = line.split(']')
                key = key.strip('[')
                metadata[section][key] = value.strip()
    datalines = lines[data_start:data_end]
    # append a ; to the header line because the P lines contain a trailing ;
    datalines[0] = 'Type;Cycle;Well;Filterset;Time;Amp_1;Amp_2;AmpRef_1;AmpRef_2;Phase;Cal;' \
        'Temp_up;Temp_down;Temp_water;O2;CO2;Humidity;Shaker;Service;User_Comment;Sys_Comment;' \
        'Reservoir;MF_Volume;Temp_Ch4;T_144;T_180;T_181_1;T_192;P_Ch1;P_Ch2;P_Ch3;T_Hum;T_CO2;' \
        'X-Pos;Y-Pos;T_LED;Ref_Int;Ref_Phase;Ref_Gain;Ch1-MP;Ch2-MF;Ch3-FA;Ch4-OP;Ch5-FB;IGNORE\n'

    # parse the data as a DataFrame
    dfraw = pandas.read_csv(io.StringIO(''.join(datalines)), sep=';', low_memory=False,
                            converters={
                                'Filterset': str
                            })

    return metadata, dfraw[list(dfraw.columns)[:-1]]


def standardize(df):
    if 'time' in df.columns:
        df['time'] = df['time'] / 3600
    return df


def extract_filtersets(metadata):

    # filterset-related metadata is spread over: channels, measurement channels, process
    channels = metadata.pop('channels')
    measurement_channels = metadata.pop('measurement channels')
    process = metadata['process']

    # dictionary that will become the DataFrame
    filtersets = {
        fnum : {}
        for fnum in range(1, int(channels['no_filterset']) + 1)
    }

    # grab data from measurement_channels
    for k, v in measurement_channels.items():
        num = int(k[0:2])
        key = k[3:]
        filtersets[num][key] = v

    # grab data from metadata['process']
    for fnum, fset in filtersets.items():
        for k in ['reference_value', 'reference_gain', 'gain']:
            pk = '{:02d}_{}_{}'.format(fnum, k, fset['name'])
            if pk in process:
                filtersets[fnum][k] = process.pop(pk)

    # convert to DataFrame and align column names & types
    df = pandas.DataFrame(filtersets).T
    ocol_ncol_type = [
        ('no', 'filter_number', int),
        ('name', 'filter_name', str),
        ('filter_id', 'filter_id', str),
        ('filter_type', 'filter_type', str),
        (None, 'excitation', float),
        (None, 'emission', float),
        #(None, 'layout', str),
        ('gain', 'gain', float),
        ('gain_1', 'gain_1', float),
        ('gain_2', 'gain_2', float),
        (None, 'phase_statistic_sigma', float),
        (None, 'signal_quality_tolerance', float),
        ('reference_gain', 'reference_gain', float),
        ('reference_value', 'reference_value', float),
        ('calibration', 'calibration', str),
        (None, 'emission2', float),
    ]
    return utils.__to_typed_cols__(df, ocol_ncol_type)


def extract_comments(dfraw):
    ocol_ncol_type = [
        ('Time', 'time', float),
        ('User_Comment', 'user_comment', str),
        ('Sys_Comment', 'sys_comment', str),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'C'], ocol_ncol_type)
    return standardize(df)


def extract_parameters(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Time', 'time', float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'P'], ocol_ncol_type)
    return standardize(df)


def extract_references(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Filterset', 'filterset', int),
        ('Time', 'time', float),
        ('Amp_1', 'amp_1', float),
        ('Amp_2', 'amp_2', float),
        ('Phase', 'phase', float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'R'], ocol_ncol_type)
    return standardize(df).set_index(['cycle', 'filterset'])


def extract_measurements(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Well', 'well', int),
        ('Filterset', 'filterset', str),
        ('Time', 'time', float),
        ('Amp_1', 'amp_1', float),
        ('Amp_2', 'amp_2', float),
        ('AmpRef_1', 'amp_ref_1', float),
        ('AmpRef_2', 'amp_ref_2', float),
        ('Phase', 'phase', float),
        ('Cal', 'cal', float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'M'], ocol_ncol_type)
    df = df.set_index(['filterset', 'cycle', 'well'])
    return standardize(df)


def extract_environment(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Time', 'time', float),
        (None, 'temp_setpoint', float),
        ('Temp_up', 'temp_up', float),
        ('Temp_down', 'temp_down', float),
        ('Temp_water', 'temp_water', float),
        ('O2', 'o2', float),
        ('CO2', 'co2', float),
        ('Humidity', 'humidity', float),
        (None, 'shaker_setpoint', float),
        ('Shaker', 'shaker_actual', float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'R'], ocol_ncol_type)
    # TODO: write initial setpoints (temp & shaker) into df
    # TODO: parse setpoint changes (temp & shaker) from comments
    # TODO: clean up -9999.0 values in co2 column
    return standardize(df)


def extract_fluidics(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Well', 'well', int),
        ('Time', 'time', float),
        ('Reservoir', 'reservoir', float),
        ('MF_Volume', 'mf_volume', float),
        ('Temp_Ch4', 'volume', float),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'F'], ocol_ncol_type)
    df = df.sort_values(['well', 'cycle']).set_index(['well'])
    return standardize(df)


def extract_valves_module(dfraw):
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Well', 'valve', str),
        ('Filterset', 'well', str),
        ('Time', 'volume_1', str),
        ('Amp_1', 'volume_2', str),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['Type'] == 'N'], ocol_ncol_type)

    # table of valve actions
    df_valves = df[df['valve'] != 'Module 1 (BASE)'].copy()
    df_valves.columns = ['cycle', 'valve', 'well', 'acid', 'base']
    df_valves['valve'] = df_valves['valve'].str.replace('Valve ', '').astype(int)
    df_valves['well'] = df_valves['well'].str.replace('Well', '').astype(int)
    df_valves['acid'] = df_valves['acid'].str.replace('Sollvolumen \(Acid\) ', '').astype(float)
    df_valves['base'] = df_valves['base'].str.replace('Sollvolumen \(Base\) ', '').astype(float)
    df_valves = standardize(df_valves).set_index(['well', 'valve', 'cycle'])

    # TODO: unknown column purpose
    df_module = df[df['valve'] == 'Module 1 (BASE)'].copy()
    df_module.columns = ['cycle', 'module', 'valve', 'well', 'volume']
    df_module['valve'] = df_module['valve'].str.replace('Valve ', '').astype(int)
    df_module['well'] = df_module['well'].str.replace('Well ', '').astype(int)
    df_module['volume'] = df_module['volume'].str.replace('Volume ', '').astype(float)
    df_module = standardize(df_module).set_index(['well', 'valve', 'cycle'])

    return df_valves, df_module


def extract_diagnostics(dfraw):
    dff = dfraw[dfraw['Type'] != 'N']
    ocol_ncol_type = [
        ('Cycle', 'cycle', int),
        ('Time', 'time', float),
        ('Temp_Ch4', 'temp_ch4', float),
        ('T_144', 't_144', float),
        ('T_180', 't_180', float),
        ('T_181_1', 't_181_1', float),
        ('T_192', 't_192', float),
        ('P_Ch1', 'p_ch1', float),
        ('P_Ch2', 'p_ch2', float),
        ('P_Ch3', 'p_ch3', float),
        ('T_Hum', 't_hum', float),
        ('T_CO2', 't_co2', float),
        ('X-Pos', 'x-pos', float),
        ('Y-Pos', 'y-pos', float),
        ('T_LED', 't_led', float),
        ('Ref_Int', 'ref_int', float),
        ('Ref_Phase', 'ref_phase', float),
        ('Ref_Gain', 'ref_gain', float),
        ('Ch1-MP', 'ch1_mp', float),
        ('Ch2-MF', 'ch2_mf', float),
        ('Ch3-FA', 'ch3_fa', float),
        ('Ch4-OP', 'ch4_op', float),
        ('Ch5-FB', 'ch5_fb', float),
    ]
    df = utils.__to_typed_cols__(dff, ocol_ncol_type)
    return standardize(df)


def transform_into_filtertimeseries(metadata:dict, measurements:pandas.DataFrame, filtersets:pandas.DataFrame):
    no_to_id = {
        int(k.split('_')[0]) : v
        for k, v in metadata['fermentation'].items()
        if k.endswith('_well')
    }
    for fs in filtersets.itertuples():
        filter_number = f'{fs.filter_number:02d}'
        key = None
        times = None
        values = None
        if fs.filter_type == 'Intensity' and ('Biomass' in fs.filter_name or 'BS' in fs.filter_name):
            key = f'BS{int(fs.gain_1)}'
            times = measurements.xs(filter_number, level='filterset')['time'].unstack()
            values = measurements.xs(filter_number, level='filterset')['amp_ref_1'].unstack()       
        elif fs.filter_type in {'pH', 'DO'}:
            key = fs.filter_type
            times = measurements.xs(filter_number, level='filterset')['time'].unstack()
            values = measurements.xs(filter_number, level='filterset')['cal'].unstack()
        else:
            if fs.filter_type == 'Intensity':
                logger.warn(
                    f'Skipping Intensity channel because no processing method could be chosen from its name {fs.filter_name}.' + 
                    ' Biomass filters should contain "Biomass" or "BS" in their name.'
                )
            else:
                logger.warn(f'Skipped {fs.filter_type} channel with name "{fs.filter_name}" because no processing routine is implemented.')
            continue

        # transform into nicely formatted DataFrames for FilterTimeSeries
        times.columns = [no_to_id[c] for c in times.columns]
        times = times.reindex(sorted(times.columns), axis=1)
        times.columns.name = 'well'
        values.columns = [no_to_id[c] for c in values.columns]
        values = values.reindex(sorted(values.columns), axis=1)
        values.columns.name = 'well'
        fts = core.FilterTimeSeries(times, values)
        yield (key, fts)
