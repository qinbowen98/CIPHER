#!/usr/bin/env python3
"""Clean figure notebooks without executing their code or reading input data."""
import argparse
import ast
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[1]
HELPER = 'Analysis helper. Inputs and outputs follow the function implementation.'
PLACEHOLDERS = ('Restricted-data analysis not included', 'raw_data_code_removed', 'Legacy figure analysis. See the repository documentation')


def remove_helper_docstrings(source):
    lines = source.splitlines(keepends=True)
    offsets = []
    total = 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    edits = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return source
    for token in tokens:
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if value == HELPER:
                start = offsets[token.start[0] - 1] + token.start[1]
                end = offsets[token.end[0] - 1] + token.end[1]
                edits.append((start, end))
    result = source
    for start, end in reversed(edits):
        result = result[:start] + result[end:]
    # A docstring-only function must remain syntactically valid.
    if edits:
        try:
            ast.parse(source)
        except SyntaxError:
            return result
        try:
            ast.parse(result)
        except SyntaxError:
            return source
    return result


def empty_source(source, cell_type):
    if not source.strip():
        return True
    if cell_type == 'code':
        meaningful = [line for line in source.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if not meaningful:
            return True
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False  # valid IPython shell/magic cells are not empty
        return not tree.body or all(isinstance(node, ast.Pass) or (isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)) and isinstance(getattr(node.value, 'value', getattr(node.value, 's', None)), str)) for node in tree.body)
    return not source.strip('# \t\r\n-*=')


def clean_payload(payload):
    kept = []
    counts = {'restricted': 0, 'empty': 0, 'manual_go': 0, 'endpoint_replacements': 0}
    for cell in payload.get('cells', []):
        source = ''.join(cell.get('source', []))
        if any(marker in source for marker in PLACEHOLDERS) or cell.get('metadata', {}).get('public_release') == 'raw_data_code_removed':
            counts['restricted'] += 1
            continue
        if 'GO_selected_barplot.pdf' in source and 'Low-Density Lipoprotein Particle Stimulus' in source and '1.2766' in source:
            counts['manual_go'] += 1
            continue
        counts['endpoint_replacements'] += source.count('ICVDr')
        source = source.replace('ICVDr', 'ICVD')
        if cell.get('cell_type') == 'code':
            source = remove_helper_docstrings(source)
        if empty_source(source, cell.get('cell_type')):
            counts['empty'] += 1
            continue
        cell['source'] = source.splitlines(keepends=True)
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        kept.append(cell)
    payload['cells'] = kept
    return payload, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'notebooks')
    args = parser.parse_args()
    totals = {'restricted': 0, 'empty': 0, 'manual_go': 0, 'endpoint_replacements': 0}
    for path in sorted(args.directory.rglob('*.ipynb')):
        payload, counts = clean_payload(json.loads(path.read_text()))
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=1) + '\n')
        for key, value in counts.items():
            totals[key] += value
        print(path.name + ': ' + str(len(payload['cells'])) + ' retained cells')
    print(json.dumps(totals, sort_keys=True))


if __name__ == '__main__':
    main()
