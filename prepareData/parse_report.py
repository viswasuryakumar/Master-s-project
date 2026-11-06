"""Parse-quality report: how many raw log lines parse, and what the rest look like."""
import argparse
import json
import os
from collections import Counter

if __package__:
    from .helper import generate_logformat_regex
else:
    from helper import generate_logformat_regex

SAMPLE_LIMIT = 5

LOG_FORMATS = {
    'bgl': '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>',
    'thunderbird': '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>',
}


def guess_log_format(log_name):
    """Pick a log format from the file name, mirroring sliding_window.py."""
    name = log_name.lower()
    if 'bgl' in name:
        return LOG_FORMATS['bgl']
    if any(key in name for key in ('thunderbird', 'spirit', 'liberty')):
        return LOG_FORMATS['thunderbird']
    raise ValueError('missing valid log format for ' + log_name)


def parse_quality_report(log_file, log_format, start_line=0, end_line=None):
    """Scan log_file with log_format and return a dict of parse-quality counts.

    Counts every line in [start_line, end_line): blank lines, lines that do not
    match the format, parsed lines with empty Content, and duplicate Content.
    Up to SAMPLE_LIMIT unparseable lines are kept as examples.
    """
    headers, regex = generate_logformat_regex(log_format)
    report = {'log_file': log_file, 'log_format': log_format, 'start_line': start_line, 'end_line': end_line,
              'lines_scanned': 0, 'blank_lines': 0, 'parsed': 0, 'unparseable': 0,
              'empty_content': 0, 'duplicate_content': 0, 'unparseable_samples': []}
    seen = Counter()
    with open(log_file, 'r', encoding='latin-1') as fin:
        for line_pos, line in enumerate(fin):
            if line_pos < start_line:
                continue
            if end_line is not None and line_pos >= end_line:
                break
            report['lines_scanned'] += 1
            text = line.strip()
            if not text:
                report['blank_lines'] += 1
                continue
            match = regex.search(text)
            if match is None:
                report['unparseable'] += 1
                if len(report['unparseable_samples']) < SAMPLE_LIMIT:
                    report['unparseable_samples'].append({'line': line_pos, 'text': text[:200]})
                continue
            report['parsed'] += 1
            content = match.group('Content').strip() if 'Content' in headers else ''
            if not content:
                report['empty_content'] += 1
            seen[content] += 1
    report['duplicate_content'] = sum(count - 1 for count in seen.values() if count > 1)
    non_blank = report['lines_scanned'] - report['blank_lines']
    report['parse_rate'] = report['parsed'] / non_blank if non_blank else 0.0
    return report


def write_report(report, output_path):
    with open(output_path, 'w') as fout:
        json.dump(report, fout, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write a parse-quality report for a raw log file.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--log-name', required=True)
    parser.add_argument('--log-format', help='defaults to a format guessed from --log-name')
    parser.add_argument('--start-line', type=int, default=0)
    parser.add_argument('--end-line', type=int, default=None)
    args = parser.parse_args()
    result = parse_quality_report(os.path.join(args.data_dir, args.log_name),
                                  args.log_format or guess_log_format(args.log_name),
                                  args.start_line, args.end_line)
    out_path = os.path.join(args.data_dir, args.log_name + '_parse_report.json')
    write_report(result, out_path)
    print('parsed %d of %d non-blank lines (%.2f%%); report: %s'
          % (result['parsed'], result['lines_scanned'] - result['blank_lines'], 100 * result['parse_rate'], out_path))
