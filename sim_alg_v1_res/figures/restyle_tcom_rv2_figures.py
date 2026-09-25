#!/usr/bin/env python3
"""Enlarge original vector-PDF text and bar groups without rerunning experiments.

Some historical raw inputs are unavailable. The immutable source PDFs preserve
all plotted values; this operation changes only text transforms and horizontal
bar geometry. Run from any directory with --output pointing at the paper repo.
Requires pypdf and PyMuPDF. Never use generated outputs as new inputs.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, ByteStringObject, ArrayObject

FILES = {
    '3': 'plot_test_dual_optimization_starlink_constellation_varying_beta_d_o_and_p_o_horizontal',
    '4': 'plot_test_dual_optimization_vs_grid_rand',
    '5a': 'plot_test_tcom_shell_time_baselines',
    '5b': 'plot_test_tcom_two_shell',
    '6': 'plot_test_tcom_shell_time_load_scaling',
    '7': 'plot_test_computing_time_measurement',
    '8': 'plot_test_dual_optimization_vs_grid_rand_varying_availability_and_for',
    '9': 'plot_test_dual_optimization_different_constellation',
    '10': 'plot_test_tcom_atp_transition',
}
FONT_SCALE = 10 / 9
BAR_SCALES = {'4': 1.25, '8': 1.4}
LABEL_ABBREVIATIONS = {
    'Average Number of LCTs per Sat.': 'Average LCTs per Sat.',
    'Field of Regard Size (Degree)': 'Field of Regard Size (deg)',
}


def instruction(values, op):
    return ([FloatObject(v) for v in values], op)


def pdf_bytes(reader, ops):
    stream = ContentStream(None, reader)
    stream.operations = ops
    writer = PdfWriter()
    page = writer.add_page(reader.pages[0])
    page.replace_contents(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def text_groups(ops):
    groups, stack = [], []
    matrix = fitz.Matrix(1, 0, 0, 1, 0, 0)
    for i, (args, op) in enumerate(ops):
        if op == b'q':
            stack.append(fitz.Matrix(matrix))
        elif op == b'Q':
            matrix = stack.pop()
        elif op == b'cm':
            matrix = fitz.Matrix(*map(float, args)) * matrix
        elif op == b'BT':
            start, transform = i, fitz.Matrix(matrix)
        elif op == b'ET':
            groups.append((start, i, transform))
    return groups


def restyle(source, target, number):
    bar_scale = BAR_SCALES.get(number, 1)
    reader = PdfReader(source)
    ops = ContentStream(reader.pages[0].get_contents(), reader).operations
    groups = text_groups(ops)
    height = float(reader.pages[0].mediabox.height)
    replacements = {}
    text_edits = {}
    for start, end, matrix in groups:
        # Isolate one text object, preserving its graphics-state transforms.
        isolated = []
        inside = False
        for i, (args, op) in enumerate(ops):
            if op == b'BT':
                inside = True
            if not inside or start <= i <= end:
                isolated.append((args, op))
            if op == b'ET':
                inside = False
        doc = fitz.open(stream=pdf_bytes(reader, isolated), filetype='pdf')
        traces = doc[0].get_texttrace()
        assert traces, (source, start)
        box = fitz.Rect(traces[0]['bbox'])
        for trace in traces[1:]:
            box |= fitz.Rect(trace['bbox'])
        # Scale around the full label centre, keeping kerning and math runs.
        centre = fitz.Point((box.x0 + box.x1) / 2, height - (box.y0 + box.y1) / 2)
        local = centre * ~matrix
        label = ''.join(chr(c[0]) for t in traces for c in t['chars'])
        if number == '8' and label in LABEL_ABBREVIATIONS:
            replacement = LABEL_ABBREVIATIONS[label]
            width = fitz.get_text_length(replacement, fontname='tiro', fontsize=9)
            # Same words/units, abbreviated to fit the narrow panel at 10 pt.
            text_edits[start] = [ops[start+1],
                instruction([(box.width-width)/2, 0], b'Td'),
                ([ArrayObject([ByteStringObject(replacement.encode('utf-16-be'))])], b'TJ')]

        replacements[start] = [([], b'q'), instruction(
            [FONT_SCALE, 0, 0, FONT_SCALE,
             (1-FONT_SCALE)*local.x, (1-FONT_SCALE)*local.y], b'cm')]
        replacements[end] = [([], b'Q')]
        doc.close()

    bars = []
    if number in ('4', '8'):
        for i in range(len(ops)-5):
            if [op for _, op in ops[i:i+6]] != [b'm', b'l', b'l', b'l', b'h', b'f']:
                continue
            xy = [list(map(float, args)) for args, _ in ops[i:i+4]]
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            if len(set(xs)) == 2 and len(set(ys)) == 2 and 0 < max(xs)-min(xs) < 5:
                bars.append((sum(xs)/4, i, xy))
        assert len(bars) == (42 if number == '4' else 84), len(bars)
        bars.sort()
        for offset in range(0, len(bars), 6):
            group = bars[offset:offset+6]
            centre = sum(b[0] for b in group)/6
            for _, i, xy in group:
                for j, (x, y) in enumerate(xy):
                    ops[i+j] = instruction([centre + bar_scale*(x-centre), y], ops[i+j][1])

    result = []
    edited_end = None
    group_ends = {start: end for start, end, _ in groups}
    for i, item in enumerate(ops):
        if edited_end is not None and i < edited_end:
            continue
        if edited_end == i:
            edited_end = None
        if item[1] == b'BT':
            result.extend(replacements[i])
        result.append(item)
        if i in text_edits:
            result.extend(text_edits[i])
            edited_end = group_ends[i]
        if item[1] == b'ET':
            result.extend(replacements[i])
    target.write_bytes(pdf_bytes(reader, result))
    # Text strings, page size, and vector path Y coordinates must be unchanged.
    before, after = fitz.open(source), fitz.open(target)
    btrace, atrace = before[0].get_texttrace(), after[0].get_texttrace()
    before_text = ''.join(chr(c[0]) for t in btrace for c in t['chars'])
    after_text = ''.join(chr(c[0]) for t in atrace for c in t['chars'])
    if number == '8':
        for old, new in LABEL_ABBREVIATIONS.items():
            before_text = before_text.replace(old, new)
    assert before_text == after_text
    assert before[0].rect == after[0].rect
    assert len(btrace) == len(atrace)
    for a, b in zip(btrace, atrace):
        assert abs(b['size']/a['size']-FONT_SCALE) < .001
    before_paths, after_paths = before[0].get_drawings(), after[0].get_drawings()
    assert len(before_paths) == len(after_paths)
    for a, b in zip(before_paths, after_paths):
        assert abs(a['rect'].y0-b['rect'].y0) < .001 and abs(a['rect'].y1-b['rect'].y1) < .001
        assert len(a['items']) == len(b['items'])
        is_bar = bool(bars and a['type'] == 'f' and a['fill'] not in ((1.,1.,1.), (0.,0.,0.)) and 0 < a['rect'].width < 5)
        if is_bar:
            assert abs(b['rect'].width/a['rect'].width-bar_scale) < .001
        else:
            assert abs(a['rect'].x0-b['rect'].x0) < .001 and abs(a['rect'].x1-b['rect'].x1) < .001
            for ai, bi in zip(a['items'], b['items']):
                assert ai[0] == bi[0]
                for av, bv in zip(ai[1:], bi[1:]):
                    if isinstance(av, (fitz.Point, fitz.Rect)):
                        assert all(abs(x-y) < .001 for x,y in zip(av,bv))
                    else:
                        assert av == bv
    return {'figure': number, 'file': target.name,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'font_scale': FONT_SCALE, 'text_objects': len(groups),
            'bar_scale': bar_scale if bars else 1, 'bar_count': len(bars),
            'text_preserved_except_documented_abbreviations': True,
            'label_abbreviations': LABEL_ABBREVIATIONS if number == '8' else {},
            'vertical_data_geometry_preserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_dir = Path(__file__).parent/'tcom_rv2_sources'
    args.output.mkdir(parents=True, exist_ok=True)
    records = [restyle(source_dir/(name+'.pdf'), args.output/(name+'.pdf'), number)
               for number, name in FILES.items() if number not in {'8', '10'}]
    # Fig. 8 now displays a subset of cases and is generated from recovered tables.
    import runpy, shutil
    runpy.run_path(str(Path(__file__).parent / 'plot_test_dual_optimization_vs_grid_rand_varying_availability_and_for_augmented.py'))
    fig8 = Path(__file__).parent / (FILES['8']+'.pdf')
    if fig8.resolve() != (args.output / fig8.name).resolve():
        shutil.copy2(fig8, args.output / fig8.name)
    # Fig. 10 is now regenerated by its ATP sweep plotter with 10-point fonts.
    # Preserve that current result instead of restoring the archived four-point sweep.
    fig10 = Path(__file__).parent / (FILES['10']+'.pdf')
    if fig10.resolve() != (args.output / fig10.name).resolve():
        shutil.copy2(fig10, args.output / fig10.name)
    (args.output/'figure_readability_validation.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
