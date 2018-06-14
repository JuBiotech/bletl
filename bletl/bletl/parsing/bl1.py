"""Parsing functions for the BioLector 1"""
import datetime
import io
import numpy
import pathlib
import pandas


from bletl import core
from bletl import utils


class BioLector1Parser(core.BLDParser):
    def parse(self, filepath):
        headerlines, data = split_header_data(filepath)
        
        metadata = extract_metadata(headerlines)
        process_parameters = extract_process_parameters(headerlines)
        filtersets = extract_filtersets(headerlines)                
        comments = extract_comments(data)
        references = extract_references(data)
        measurements = extract_measurements(data)
        environment = extract_environment(data)

        data = core.BLData()
        data.metadata = metadata
        data.filtersets = filtersets
        data.process_parameters = process_parameters
        data.comments = comments
        data.references = references
        data.measurements = measurements
        return data


def read_header_loglines(dir_incremental):
    fp_header = pathlib.Path(dir_incremental, 'header.csv')
    
    with open(fp_header, encoding='latin-1') as f:
        headerlines = f.readlines()
        
    loglines = []
    for tmpfile in dir_incremental.iterdir():
        if tmpfile.suffix == '.tmp':
            with open(tmpfile, 'r', encoding='latin-1') as f:
                loglines += f.readlines()
    return headerlines, loglines


def split_header_data(fp):
    with open(fp, 'r', encoding='latin-1') as f:
        lines = f.readlines()

    headerlines = []
    datalines = []

    header_end = None

    for l, line in enumerate(lines):
        if not header_end:
            headerlines.append(line)
            if line.startswith('READING;WELLNUM'):
                header_end = l
    datalines = lines[header_end:]
    datalines[0] = datalines[0].strip() + ';IGNORE\n'
    # append a ; to the header line because some lines contain a trailing ;
    datalines[0] = datalines[0].strip() + ';IGNORE\n'

    # parse the data as a DataFrame
    dfraw = pandas.read_csv(io.StringIO(''.join(datalines)), sep=';', low_memory=False)

    return headerlines, dfraw


def extract_metadata(headerlines):
    L4 = headerlines[4].split(';')
    L6 = headerlines[6].split(';')

    metadata = {
        'filename': headerlines[0].split(';')[1].strip(),
        'protocol': headerlines[1].split(';')[1].strip(),
        'file_version': headerlines[2].split(';')[1].strip(),
        'date_start': datetime.datetime.strptime(' '.join(headerlines[3].split(';')[1:3]).strip(), '%Y-%m-%d %H:%M:%S'),
        'date_end': datetime.datetime.strptime(' '.join(L4[1:3]), '%Y-%m-%d %H:%M:%S'),
        'last_reading': L4[5],
        'timestamp': L4[7].strip(),
        'device': headerlines[5][7:].strip(),
        'user': L6[1],
        'comment': L6[3].strip() if L6[3] != 'no comment available\n' else None,
        'platetype': headerlines[7].split(';')[1],
        'lot': headerlines[7].split(';')[3].strip(),
        'mtp_rows': int(headerlines[8][9:-1]),
        'mtp_columns': int(headerlines[9][12:-1]),
        'filtersets': int(headerlines[10].split(';')[1]),
        'reference_mode': headerlines[10].split(';')[3],
        'multi_pmt': headerlines[10].split(';')[5],
    }
    if metadata['date_end'] == metadata['date_start']:
        metadata['date_end'] = None

    return metadata


def extract_filtersets(headerlines):
    filterlines = []
    filter_start = False
    for l, line in enumerate(headerlines):
        if line.startswith('FILTERSET;'):
            filter_start = True
            filterlines.append(line)
        elif filter_start and not line.startswith(';;;'):
            filterlines.append(line)
        elif filter_start:
            break
        
    df_filtersets = pandas.read_csv(io.StringIO(''.join(filterlines)), sep=';', usecols=range(11))
    df_filtersets.columns = ['filtername', 'excitation', 'emission', 'layout', 'filternumber', 'gain', 'phase_statistic_sigma', 'signal_quality_tolerance', 'reference_value', 'emission2', 'gain2']
    cols_numeric = ['excitation', 'emission', 'gain', 'phase_statistic_sigma', 'signal_quality_tolerance', 'reference_value', 'emission2', 'gain2']
    df_filtersets[cols_numeric] = df_filtersets[cols_numeric].apply(pandas.to_numeric)
    return df_filtersets


def extract_process_parameters(headerlines):
    fs_start = None
    for l, line in enumerate(headerlines):
        if line.startswith('FILTERSET;'):
            fs_start = l
            break

    proccess_parameters = {
        'temperature': float(headerlines[fs_start + 1].split(';')[13].strip()),
        'humidity': float(headerlines[fs_start + 2].split(';')[13].strip()),
        'O2': float(headerlines[fs_start + 3].split(';')[13].strip()),
        'CO2': float(headerlines[fs_start + 4].split(';')[13].strip()),
        'shaking': float(headerlines[fs_start + 5].split(';')[13].strip()),
        'cycle_time': float(headerlines[fs_start + 6].split(';')[13].strip()),
        'exp_time': float(headerlines[fs_start + 7].split(';')[13].strip()),
    }
    return proccess_parameters


def extract_comments(dfraw):
    ocol_ncol_type = [
        ('TIME [h]', 'time', float),
        ('COMMENTS', 'user_comment', str),
    ]
    df = utils.__to_typed_cols__(dfraw[dfraw['READING'] == 'K'], ocol_ncol_type)
    # TODO: automatically separate comments into user/sys
    df['sys_comment'] = None
    df.index = range(len(df))
    # TODO: convert time
    return df


def extract_references(dfraw):

    dfref = dfraw[dfraw['READING'] == 'R'].copy()
    lookup = list(dfraw['READING'])
    cycles = -numpy.ones((len(dfref),), dtype=int)
    for i, (r, row) in enumerate(dfref.iterrows()):
        cycles[i] = int(lookup[r + 1][1:])
    dfref['cycle'] = cycles

    ocol_ncol_type = [
        ('cycle', 'cycle', int),
        ('TIME [h]', 'time', float),
        ('FILTERSET', 'filterset', int),
        ('AMPLITUDE', 'amp_1', float),
        ('amp_2', 'amp_2', float),
        ('PHASE', 'phase', float),
    ]
    dfref['amp_2'] = numpy.nan
    df = utils.__to_typed_cols__(dfref, ocol_ncol_type)
    # TODO: convert time
    return df.set_index(['cycle', 'filterset'])


def extract_measurements(dfraw):
    dfmes = dfraw[dfraw['READING'].str.startswith('C')].copy()
    dfmes['cycle'] = dfmes['READING'].str.replace('C', '').astype(int)
    dfmes['amp_2'] = numpy.nan
    dfmes['amp_ref_1'] = numpy.nan
    dfmes['amp_ref_2'] = numpy.nan
    dfmes['cal'] = numpy.nan
    ocol_ncol_type = [
        ('cycle', 'cycle', int),
        ('WELLNUM', 'well', str),
        ('FILTERSET', 'filterset', int),
        ('TIME [h]', 'time', float),
        ('AMPLITUDE', 'amp_1', float),
        ('amp_2', 'amp_2', float),
        ('amp_ref_1', 'amp_ref_1', float),
        ('amp_ref_2', 'amp_ref_2', float),
        ('PHASE', 'phase', float),
        ('cal', 'cal', float),
    ]
    df = utils.__to_typed_cols__(dfmes, ocol_ncol_type)
    df = df.set_index(['filterset', 'cycle', 'well'])

    # TODO: convert time
    # TODO: convert well ids
    return df


def extract_environment(dfraw):
    #lines_env = ''.join([l for l in loglines if not l.startswith('K')])
    #df_env = pandas.read_csv(io.StringIO(lines_env), sep=';', usecols=[5,8,9,10,11])
    #df_env.columns = ['time', 'temperature', 'humidity', 'O2', 'CO2']
    return dfraw[dfraw['READING'] != 'K']
