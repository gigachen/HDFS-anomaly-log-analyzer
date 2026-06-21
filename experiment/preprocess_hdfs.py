#!/usr/bin/env python3
"""
preprocess_hdfs.py
Preprocess raw HDFS logs into the same five artifacts the notebook reads.

Inputs:
  --log_file    raw HDFS.log (from LogHub: github.com/logpai/loghub)
  --labels      anomaly_label.csv (BlockId, Label[Normal|Anomaly])
  --templates   HDFS.log_templates.csv (EventId, EventTemplate using [*])
  --output_dir  output folder, must not be 'preprocessed' (default: preprocessed_real)

Outputs (in --output_dir):
  Event_occurrence_matrix.csv  block x E1..E29 frequency features
  Event_traces.csv             block-level ordered event sequences
  HDFS.npz                     x_data / y_data for the LSTM
  HDFS.log_templates.csv       copied through
  anomaly_label.csv            copied through

Usage:
  python preprocess_hdfs.py --log_file raw/HDFS.log \\
      --labels   preprocessed/anomaly_label.csv \\
      --templates preprocessed/HDFS.log_templates.csv \\
      --output_dir preprocessed_real
"""
import argparse
import os
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd


# HDFS log line: <Date> <Time> <Pid> <Level> <Component>: <Content>
LOG_LINE_RE = re.compile(
    r'^(\d+)\s+(\d+)\s+(\d+)\s+(\w+)\s+([\w$.]+):\s*(.*)$'
)
BLOCK_RE = re.compile(r'blk_-?\d+')


def template_to_regex(template: str) -> re.Pattern:
    """Convert a template containing [*] wildcards into a compiled regex.

    Example:  '[*]Receiving block[*]src:[*]dest:[*]'
        ->    '^.*?Receiving block.*?src:.*?dest:.*?$'
    """
    parts = template.split('[*]')
    escaped = [re.escape(p) for p in parts]
    return re.compile('^' + '.*?'.join(escaped) + '$')


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--log_file', required=True, help='raw HDFS.log')
    p.add_argument('--labels', default='preprocessed/anomaly_label.csv')
    p.add_argument('--templates', default='preprocessed/HDFS.log_templates.csv')
    p.add_argument('--output_dir', default='preprocessed_real')
    p.add_argument('--max_lines', type=int, default=None,
                   help='Cap on lines processed (for a quick smoke test)')
    return p.parse_args()


def main():
    args = parse_args()

    if os.path.abspath(args.output_dir) == os.path.abspath('preprocessed'):
        sys.exit("Refusing to write into 'preprocessed/'. Choose another folder.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load templates and compile matchers
    print(f'[1/5] Loading templates: {args.templates}')
    df_templates = pd.read_csv(args.templates)
    matchers = [
        (row['EventId'], template_to_regex(row['EventTemplate']))
        for _, row in df_templates.iterrows()
    ]
    event_ids = [eid for eid, _ in matchers]
    print(f'      {len(matchers)} templates loaded ({event_ids[0]}..{event_ids[-1]})')

    # Load labels
    print(f'[2/5] Loading labels: {args.labels}')
    df_labels = pd.read_csv(args.labels)
    label_map = dict(zip(df_labels['BlockId'], df_labels['Label']))
    print(f'      {len(label_map):,} block labels')

    # Stream parse
    print(f'[3/5] Parsing log: {args.log_file}')
    block_events = defaultdict(list)
    total = matched = malformed = unmatched = 0
    unmatched_samples = []
    t0 = time.time()

    with open(args.log_file, 'r', errors='replace') as f:
        for line in f:
            if args.max_lines and total >= args.max_lines:
                break
            total += 1
            if total % 500_000 == 0:
                rate = total / (time.time() - t0)
                print(f'      {total:>12,} lines  ({rate:>8,.0f}/s)')

            m = LOG_LINE_RE.match(line.rstrip())
            if not m:
                malformed += 1
                continue
            content = m.group(6)

            event_id = None
            for eid, pat in matchers:
                if pat.match(content):
                    event_id = eid
                    break
            if event_id is None:
                unmatched += 1
                if len(unmatched_samples) < 3:
                    unmatched_samples.append(content[:120])
                continue
            matched += 1

            # dedupe block IDs per line: one log line = one event per distinct block
            for blk in set(BLOCK_RE.findall(content)):
                block_events[blk].append(event_id)

    elapsed = time.time() - t0
    print(f'      done in {elapsed:.1f}s')
    print(f'      lines: {total:,}  matched: {matched:,}  '
          f'malformed: {malformed:,}  unmatched: {unmatched:,}')
    print(f'      unique blocks seen: {len(block_events):,}')
    if unmatched_samples:
        print('      sample unmatched lines:')
        for s in unmatched_samples:
            print(f'        | {s}')

    # Keep blocks that have a known label
    blocks = [b for b in block_events if b in label_map]
    print(f'      blocks intersecting label set: {len(blocks):,}')

    # Build Event_traces.csv
    print('[4/5] Writing Event_traces.csv')
    trace_rows = []
    for b in blocks:
        evs = block_events[b]
        trace_rows.append({
            'BlockId': b,
            'Label': 'Fail' if label_map[b] == 'Anomaly' else 'Success',
            'Type': '',
            'Features': '[' + ','.join(evs) + ']',
            'TimeInterval': '',
            'Latency': '',
        })
    pd.DataFrame(trace_rows).to_csv(
        os.path.join(args.output_dir, 'Event_traces.csv'), index=False
    )

    # Build Event_occurrence_matrix.csv
    print('      writing Event_occurrence_matrix.csv')
    occ_rows = []
    for b in blocks:
        counts = dict.fromkeys(event_ids, 0)
        for e in block_events[b]:
            counts[e] += 1
        row = {
            'BlockId': b,
            'Label': 'Fail' if label_map[b] == 'Anomaly' else 'Success',
            'Type': '',
            **counts,
        }
        occ_rows.append(row)
    pd.DataFrame(occ_rows)[['BlockId', 'Label', 'Type'] + event_ids].to_csv(
        os.path.join(args.output_dir, 'Event_occurrence_matrix.csv'), index=False
    )

    # Build HDFS.npz
    print('      writing HDFS.npz')
    x_data = np.array([block_events[b] for b in blocks], dtype=object)
    y_data = np.array(
        [1 if label_map[b] == 'Anomaly' else 0 for b in blocks], dtype=np.int64
    )
    np.savez(os.path.join(args.output_dir, 'HDFS.npz'),
             x_data=x_data, y_data=y_data)

    # Copy templates and labels through so the output folder is self-contained
    print('[5/5] Copying templates and labels')
    shutil.copyfile(args.templates,
                    os.path.join(args.output_dir, 'HDFS.log_templates.csv'))
    shutil.copyfile(args.labels,
                    os.path.join(args.output_dir, 'anomaly_label.csv'))

    n_anom = int(y_data.sum())
    print(f'\nDone. Output in {args.output_dir}/')
    print(f'  blocks: {len(blocks):,}  anomaly: {n_anom:,} '
          f'({n_anom/len(blocks)*100:.2f}%)')
    print('Point the notebook at this folder by setting:')
    print(f"  DATA_DIR = '{args.output_dir}/'")


if __name__ == '__main__':
    main()
