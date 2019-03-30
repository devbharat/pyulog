#! /usr/bin/env python

"""
Convert a ULog file into h5py file
"""

from __future__ import print_function

import argparse
import os
import h5py

from .core import ULog

#pylint: disable=too-many-locals, invalid-name, consider-using-enumerate

def main():
    """Command line interface"""

    parser = argparse.ArgumentParser(description='Convert ULog to h5py')
    parser.add_argument('filename', metavar='file.ulg', help='ULog input file')

    parser.add_argument(
        '-m', '--messages', dest='messages',
        help=("Only consider given messages. Must be a comma-separated list of"
              " names, like 'sensor_combined,vehicle_gps_position'"))

    parser.add_argument('-o', '--output', dest='output', action='store',
                        help='Output directory (default is same as input file)',
                        metavar='DIR')

    args = parser.parse_args()

    if args.output and not os.path.isdir(args.output):
        print('Creating output directory {:}'.format(args.output))
        os.mkdir(args.output)

    convert_ulog2h5py(args.filename, args.messages, args.output)


def convert_ulog2csv(ulog_file_name, messages=None, output=None):
    """
    Coverts and ULog file to a h5py file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path

    :return: None
    """

    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter)
    data = ulog.data_list

    output_file_prefix = ulog_file_name
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]
        output_file_name = output_file_prefix + '.hdf5'

    # write to different output path?
    if output:
        base_name = os.path.basename(output_file_name)
        output_file_name = os.path.join(output, base_name)

    g = h5py.File(output_file_name,'w')
    for d in data:
        data_keys = [f.field_name for f in d.field_data]
        data_keys.remove('timestamp')
        data_keys.insert(0, 'timestamp')  # we want timestamp at first position
        for key in data_keys:
            g.create_dataset(d.name + '/' + str(d.multi_id) + '/'+ key, data=d.data[key], compression="lzf")
    g.close()
    del g
