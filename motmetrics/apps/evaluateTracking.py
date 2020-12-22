# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using MOTChallenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import io
import logging
import os
import sys
from tempfile import NamedTemporaryFile
import time

import motmetrics as mm
import pandas as pd

METRICS = [
    # mm.metrics.motchallenge_metrics
    'idf1',
    'idp',
    'idr',
    'recall',
    'precision',
    'num_unique_objects',
    'num_unique_predictions',
    'mostly_tracked',
    'partially_tracked',
    'mostly_lost',
    'num_false_positives',
    'num_misses',
    'num_switches',
    'num_fragmentations',
    'mota',
    'motp',
    'num_transfer',
    'num_ascend',
    'num_migrate',
    # For debug purposes:
    'idtp',
    'idfn',
    'idfp',
    'num_frames',
    'num_detections',
    'num_objects',
    'num_predictions',
    'num_overlap',
]


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Seqmap for test data
    [name]
    <SEQUENCE_1>
    <SEQUENCE_2>
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('seqmap', type=str, help='Text file containing all sequences name')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    parser.add_argument('--output_file', type=str, help='(optional) Write result of evalution to file.')
    parser.add_argument('--debug_dir', type=str, help='(optional) Write debug to this dir.')
    return parser.parse_args()


def compare_dataframes(gts, ts, vsflag='', iou=0.5):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    preprocs = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Evaluating %s...', k)
            if vsflag != '':
                fd = io.open(vsflag + '/' + k + '.log', 'w')
            else:
                fd = ''
            acc, _, _, ts_preproc = mm.utils.CLEAR_MOT_M(
                    gts[k][0], tsacc, gts[k][1], 'iou', distth=iou, vflag=fd)
            if fd != '':
                fd.close()
            accs.append(acc)
            names.append(k)
            preprocs.append(ts_preproc)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, names, preprocs


def parseSequences(seqmap):
    """Loads list of sequences from file."""
    assert os.path.isfile(seqmap), 'Seqmap %s not found.' % seqmap
    fd = io.open(seqmap)
    res = []
    for row in fd.readlines():
        row = row.strip()
        if row == '' or row == 'name' or row[0] == '#':
            continue
        res.append(row)
    fd.close()
    return res


def generateSkippedGT(gtfile, skip, fmt):
    """Generates temporary ground-truth file with some frames skipped."""
    del fmt  # unused
    tf = NamedTemporaryFile(delete=False, mode='w')
    with io.open(gtfile) as fd:
        lines = fd.readlines()
        for line in lines:
            arr = line.strip().split(',')
            fr = int(arr[0])
            if fr % (skip + 1) != 1:
                continue
            pos = line.find(',')
            newline = str(fr // (skip + 1) + 1) + line[pos:]
            tf.write(newline)
    tf.close()
    tempfile = tf.name
    return tempfile


def main():
    # pylint: disable=missing-function-docstring
    # pylint: disable=too-many-locals
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    seqs = parseSequences(args.seqmap)
    gtfiles = [os.path.join(args.groundtruths, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(args.tests, '%s.txt' % i) for i in seqs]

    for gtfile in gtfiles:
        if not os.path.isfile(gtfile):
            logging.error('gt File %s not found.', gtfile)
            sys.exit(1)
    for tsfile in tsfiles:
        if not os.path.isfile(tsfile):
            logging.error('res File %s not found.', tsfile)
            sys.exit(1)

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    for seq in seqs:
        logging.info('\t%s', seq)
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    if args.skip > 0 and 'mot' in args.fmt:
        for i, gtfile in enumerate(gtfiles):
            gtfiles[i] = generateSkippedGT(gtfile, args.skip, fmt=args.fmt)

    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f, fmt=args.fmt), os.path.join(args.groundtruths, seqs[i], 'seqinfo.ini'))) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f, fmt=args.fmt)) for i, f in enumerate(tsfiles)])

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    mh = mm.metrics.create()
    st = time.time()
    accs, names, preprocs = compare_dataframes(gt, ts, args.log, 1. - args.iou)
    logging.info('adding frames: %.3f seconds.', time.time() - st)

    # Write preprocessed sequences to debug dir.
    if args.debug_dir:
        for name, ts_preproc in zip(names, preprocs):
            clean = ts_preproc.copy()
            clean[['X', 'Y']] += 1  # Restore removed in load_motchallenge.
            xywh = ['X', 'Y', 'Width', 'Height']
            clean[xywh] = clean[xywh].round(3) + 0  # Add zero to convert -0 to +0.
            try:
                with open(os.path.join(args.debug_dir, f'clean-{name}.csv'), 'w') as f:
                    clean.to_csv(f, header=False, float_format='%g')
            except IOError as ex:
                logging.warning('io error: %s', ex)

    logging.info('Running metrics')

    # summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)
    results_per_seq = OrderedDict()
    for name, acc in zip(names, accs):
        results_dict = mh.compute(
                acc, name=name,
                metrics=(METRICS + ['id_global_assignment', 'obj_frequencies', 'pred_frequencies']),
                return_dataframe=False)
        if args.debug_dir:
            with open(os.path.join(args.debug_dir, f'id-{name}.csv'), 'w') as f:
                _write_id_assignment(f, results_dict)
        results_per_seq[name] = results_dict
    partials = [
            pd.DataFrame(OrderedDict([(k, results_dict[k]) for k in METRICS]), index=[seq_name])
            for seq_name, results_dict in results_per_seq.items()
    ]
    overall = mh.compute_overall(results_per_seq.values(), metrics=METRICS, name='OVERALL')
    partials.append(overall)
    summary = pd.concat(partials)

    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')

    if args.output_file:
        summary.index.name = 'sequence'
        _ensure_parent_dir_exists(args.output_file)
        with open(args.output_file, 'w') as f:
            summary.to_csv(f)


def _ensure_parent_dir_exists(fname):
    parent_dir = os.path.dirname(fname)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _write_id_assignment(f, results_dict):
    freq_gt = results_dict['obj_frequencies']
    freq_pr = results_dict['pred_frequencies']
    fn_matrix = results_dict['id_global_assignment']['fnmatrix']
    fp_matrix = results_dict['id_global_assignment']['fpmatrix']
    matrix_oids = results_dict['id_global_assignment']['oids']
    matrix_hids = results_dict['id_global_assignment']['hids']
    # Optimal solution.
    opt_rids = results_dict['id_global_assignment']['rids']
    opt_cids = results_dict['id_global_assignment']['cids']
    opt_ij = set(zip(opt_rids, opt_cids))

    # Iterate through all track pairs with non-zero overlap.
    pairs = []
    for i, oid in enumerate(matrix_oids):
        for j, hid in enumerate(matrix_hids):
            tp_from_fn = freq_gt[oid] - fn_matrix[i, j]
            tp_from_fp = freq_pr[hid] - fp_matrix[i, j]
            if not tp_from_fn == tp_from_fp:
                raise ValueError('FN and FP do not agree',
                                 (fn_matrix[i, j], freq_gt[oid]),
                                 (fp_matrix[i, j], freq_pr[hid]))
            tp = tp_from_fn
            if tp == 0:
                continue
            pairs.append({
                    'gt_id': oid,
                    'pr_id': hid,
                    'gt_len': freq_gt[oid],
                    'pr_len': freq_pr[hid],
                    'tp': tp,
                    'opt': 1 if (i, j) in opt_ij else 0,
            })

    if not pairs:
        # Leave file empty.
        return
    columns = ['gt_id', 'pr_id', 'gt_len', 'pr_len', 'tp', 'opt']
    table = pd.DataFrame.from_records(pairs, columns=columns)
    table = table.set_index(['gt_id', 'pr_id']).sort_index()
    table.to_csv(f, float_format='%g', header=False)


if __name__ == '__main__':
    main()
