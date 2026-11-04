"""Sujungti pateiktas modelių suvestines ir sukurti bendras vizualizacijas."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'bendri_rezultatai'
MODELS = ('EfficientNet-B0', 'ResNet18', 'VGG16', 'MobileNetV2')
DATASETS = ('CIFAR-10', 'Fashion-MNIST')
COLORS = ('#237c67', '#3274b6', '#a36c32', '#8b60a8')
METRICS = ('Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)')
METRIC_LABELS = ('Tikslumas', 'Macro precision', 'Macro recall', 'Macro F1')
FIELDS = (
    'Modelis', 'Duomenų rinkinys', *METRICS, 'Testo nuostolis', 'Epochų sk.',
    'Mokymo laikas (min)', 'Inferencija paketu (ms/vaizdui)',
    'Inferencija, 1 vaizdas (ms)', 'Parametrai (mln.)', 'Modelio dydis (MB)',
    'Fine-tuning', 'Batch size', 'Learning rate (head)', 'Learning rate (fine)',
    'Šaltinio failas',
)
NUMERIC = set(FIELDS[2:13]) | {'Batch size', 'Learning rate (head)', 'Learning rate (fine)'}
FIGURES = (
    ('01_kokybes_metrikos', 'Klasifikavimo kokybė',
     'Visų modelių tikslumas, macro precision, recall ir F1 abiem duomenų rinkiniams.'),
    ('02_metriku_matricos', 'Bendros metrikų matricos',
     'Eilutėse — modeliai, stulpeliuose — metrikos. Reikšmės procentais.'),
    ('03_mokymas_ir_nuostolis', 'Mokymo trukmė ir testo nuostolis',
     'Pateiktų paleidimų matavimai. Aparatinė įranga ir bibliotekos nesuvienodintos.'),
    ('04_inferencijos_trukme', 'Inferencijos trukmė',
     'Paketo ir vieno vaizdo matavimai rodomi atskirai bendroje figūroje; ašys logaritminės.'),
    ('05_modeliu_apimtis', 'Modelių apimtis',
     'Parametrų skaičius ir pateiktų modelių failų dydžiai. Failų formatai gali skirtis.'),
    ('06_kokybe_ir_sanaudos', 'Kokybė ir sąnaudos',
     'Tikslumas lyginamas su mokymo trukme ir inferencijos paketu trukme.'),
)


def read_inputs():
    paths = [ROOT / 'rezultatai' / 'ResNet18_suvestine.csv']
    paths += sorted((ROOT / 'kursioku').glob('*.csv'))
    paths += sorted((ROOT / 'kursioku').glob('*.txt'))
    if len(paths) < 4:
        raise ValueError('Trūksta šaltinių: reikalinga ResNet18 ir trijų kitų modelių suvestinė.')
    records = []
    for path in paths:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'Modelis', 'Duomenų rinkinys', *METRICS}.issubset(reader.fieldnames or ()):
                raise ValueError(f'Neatpažinta suvestinė: {path}')
            for row in reader:
                row['Duomenų rinkinys'] = {
                    'FashionMNIST': 'Fashion-MNIST',
                    'Fashion-MNIST': 'Fashion-MNIST',
                    'CIFAR-10': 'CIFAR-10',
                }.get(row['Duomenų rinkinys'].strip(), row['Duomenų rinkinys'].strip())
                row['Modelis'] = row['Modelis'].strip()
                row['Šaltinio failas'] = path.relative_to(ROOT).as_posix()
                for field in NUMERIC:
                    if row.get(field, '') != '':
                        row[field] = float(row[field])
                records.append(row)
    return validate(records)


def validate(records):
    lookup = {}
    required = FIELDS[2:13]
    for row in records:
        key = (row['Modelis'], row['Duomenų rinkinys'])
        if key in lookup:
            raise ValueError(f'Pasikartojantys rezultatai: {key}')
        for field in required:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Netinkama reikšmė: {key}, {field}')
            row[field] = value
        if any(row[field] > 1 for field in METRICS):
            raise ValueError(f'Metrikos turi būti tarp 0 ir 1: {key}')
        lookup[key] = row
    expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
    if set(lookup) != expected:
        raise ValueError(f'Netikėta modelių ir duomenų rinkinių aprėptis: {set(lookup) ^ expected}')
    return [lookup[model, dataset] for dataset in DATASETS for model in MODELS]


def save_summary(records):
    with (OUTPUT / 'bendra_suvestine.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT / 'bendra_suvestine.json').write_text(
        json.dumps([{key: row.get(key) for key in FIELDS} for row in records],
                   ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )


def setup():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelcolor': '#344054', 'text.color': '#182230',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
    })


def finish(fig, stem, title, note=None, *, report_pdf):
    fig.suptitle(title, fontsize=17, fontweight='bold', y=1.01)
    if note:
        fig.text(0.01, -0.025, note, fontsize=9, color='#667085')
    fig.savefig(OUTPUT / f'{stem}.png', dpi=220, bbox_inches='tight')
    fig.savefig(OUTPUT / f'{stem}.pdf', bbox_inches='tight')
    report_pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def grouped(ax, lookup, field, title, *, scale=1, unit='', log=False):
    centers = np.arange(len(DATASETS))
    width = 0.18
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        values = [lookup[model, dataset][field] * scale for dataset in DATASETS]
        bars = ax.bar(centers + (i - 1.5) * width, values, width, color=color, label=model)
        ax.bar_label(bars, labels=[f'{v:.2f}{unit}' for v in values], padding=4, fontsize=8)
    ax.set_xticks(centers, DATASETS)
    ax.set_title(title, loc='left', pad=14)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.15)
    if log:
        ax.set_yscale('log')
        positive = [lookup[m, d][field] * scale for m in MODELS for d in DATASETS]
        ax.set_ylim(min(positive) / 1.7, max(positive) * 2.2)
    elif scale == 100:
        ax.set_ylim(0, 108)
        ax.set_ylabel('Procentai (%)')
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.14)


def legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=4, frameon=False)


def draw(records, report_pdf):
    lookup = {(r['Modelis'], r['Duomenų rinkinys']): r for r in records}
    setup()
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, field, label in zip(axs.flat, METRICS, METRIC_LABELS):
        grouped(ax, lookup, field, label, scale=100)
    fig.subplots_adjust(top=0.88, hspace=0.4, wspace=0.22)
    legend(fig, axs[0, 0])
    finish(fig, FIGURES[0][0], FIGURES[0][1], report_pdf=report_pdf)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.3))
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.14, right=0.88, wspace=0.42)
    for ax, dataset in zip(axs, DATASETS):
        values = np.array([[lookup[model, dataset][field] * 100
                            for field in METRICS] for model in MODELS])
        im = ax.imshow(values, cmap='YlGnBu', vmin=90, vmax=100, aspect='auto')
        ax.set_xticks(range(4), ('Accuracy', 'Precision', 'Recall', 'F1'))
        ax.set_yticks(range(4), MODELS)
        ax.set_title(dataset, loc='left', pad=12)
        for (r, c), value in np.ndenumerate(values):
            ax.text(c, r, f'{value:.2f}', ha='center', va='center',
                    color='white' if value >= 96 else '#102a43', fontweight='bold')
    color_axis = fig.add_axes((0.92, 0.17, 0.015, 0.59))
    fig.colorbar(im, cax=color_axis, label='Procentai (%)')
    finish(fig, FIGURES[1][0], FIGURES[1][1], 'Macro precision, recall ir F1. Vienoda abiejų matricų spalvų skalė: 90–100 %.', report_pdf=report_pdf)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3))
    grouped(axs[0], lookup, 'Mokymo laikas (min)', 'Mokymo trukmė')
    axs[0].set_ylabel('Minutės')
    grouped(axs[1], lookup, 'Testo nuostolis', 'Testo nuostolis')
    fig.subplots_adjust(top=0.81, wspace=0.22)
    legend(fig, axs[0])
    finish(fig, FIGURES[2][0], FIGURES[2][1], 'Laikai priklauso nuo GPU, bibliotekos ir mokymo sąlygų; vienodos sąlygos nepatvirtintos.', report_pdf=report_pdf)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3))
    grouped(axs[0], lookup, 'Inferencija paketu (ms/vaizdui)', 'Inferencija paketu', log=True)
    grouped(axs[1], lookup, 'Inferencija, 1 vaizdas (ms)', 'Vieno vaizdo inferencija', log=True)
    for ax in axs:
        ax.set_ylabel('ms / vaizdui · logaritminė skalė')
    fig.subplots_adjust(top=0.81, wspace=0.24)
    legend(fig, axs[0])
    finish(fig, FIGURES[3][0], FIGURES[3][1], 'Pateikti skirtingų paleidimų matavimai; greičio skirtumai neatskiria aparatinės įrangos ir bibliotekos poveikio.', report_pdf=report_pdf)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.3))
    grouped(axs[0], lookup, 'Parametrai (mln.)', 'Parametrų skaičius')
    axs[0].set_ylabel('Milijonai parametrų')
    grouped(axs[1], lookup, 'Modelio dydis (MB)', 'Išsaugoto modelio dydis')
    axs[1].set_ylabel('MB (kaip pateikta šaltinyje)')
    fig.subplots_adjust(top=0.81, wspace=0.22)
    legend(fig, axs[0])
    finish(fig, FIGURES[4][0], FIGURES[4][1], 'Modelių failų dydžiai priklauso nuo išsaugojimo formato ir saugomos informacijos.', report_pdf=report_pdf)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for col, dataset in enumerate(DATASETS):
        for row, field, xlabel in (
            (0, 'Mokymo laikas (min)', 'Mokymo trukmė (min)'),
            (1, 'Inferencija paketu (ms/vaizdui)', 'Inferencija paketu (ms / vaizdui)'),
        ):
            ax = axs[row, col]
            for model, color in zip(MODELS, COLORS):
                result = lookup[model, dataset]
                x, y = result[field], result['Accuracy'] * 100
                ax.scatter(x, y, color=color, s=85, edgecolor='white', linewidth=1.1, zorder=3)
                ax.annotate(model, (x, y), xytext=(7, 7), textcoords='offset points', fontsize=9)
            ax.set_title(dataset, loc='left')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Tikslumas (%)')
            ax.set_ylim(92, 98.4)
            ax.set_xlim(0, max(lookup[m, dataset][field] for m in MODELS) * 1.28)
            ax.grid(alpha=0.15)
    fig.subplots_adjust(top=0.91, hspace=0.43, wspace=0.23)
    finish(fig, FIGURES[5][0], FIGURES[5][1], 'Grafikai aprašo pateiktus paleidimus. Skirtingos vykdymo sąlygos riboja tiesioginį modelių greičio palyginimą.', report_pdf=report_pdf)


def report(records):
    columns = ('Modelis', 'Duomenų rinkinys', 'Accuracy', 'F1 (macro)', 'Mokymo laikas (min)')
    rows = []
    for row in records:
        values = [row['Modelis'], row['Duomenų rinkinys'],
                  f"{row['Accuracy'] * 100:.2f} %", f"{row['F1 (macro)'] * 100:.2f} %",
                  f"{row['Mokymo laikas (min)']:.2f}"]
        rows.append('<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>')
    cards = ''.join(
        f'<section><h2>{html.escape(title)}</h2><p>{html.escape(description)}</p>'
        f'<a href="{stem}.pdf">PDF</a><img src="{stem}.png" alt="{html.escape(title)}"></section>'
        for stem, title, description in FIGURES
    )
    document = f'''<!doctype html><html lang="lt"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bendras modelių palyginimas</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182230;background:#f4f6f8;margin:0}}
main{{max-width:1200px;margin:auto;padding:32px}} h1{{font-size:34px;line-height:1.2}}
section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}
img{{width:100%;height:auto;margin-top:16px}} table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;border-bottom:1px solid #e4e7ec;padding:10px}}
a{{color:#245f9e}} .note{{color:#475467}} h2{{margin-top:0}}</style>
<main><h1>Keturi modeliai · du duomenų rinkiniai</h1>
<p>EfficientNet-B0, ResNet18, VGG16 ir MobileNetV2 rezultatai vienoje ataskaitoje.</p>
<p><a href="bendras_palyginimas.pdf">Visi grafikai viename PDF</a></p>
<section><h2>Bendra suvestinė</h2><p><a href="bendra_suvestine.csv">CSV</a> ·
<a href="bendra_suvestine.json">JSON</a></p><table><thead><tr>
{''.join(f'<th>{html.escape(c)}</th>' for c in columns)}</tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="note"><h2>Palyginimo ribos</h2>
<p>Pateiktuose kursiokų failuose yra tik galutinės suvestinės. Nėra jų epochų istorijos,
klasių ataskaitų, tikrųjų ir prognozuotų žymių ar skaitinių klaidų matricų.
Todėl bendri mokymo kreivių, F1 pagal klases ir klaidų matricų grafikai nesudaryti.
Rodomos metrikų matricos yra modelių ir metrikų lentelės.</p>
<p>GPU, bibliotekos ir laiko matavimo sąlygos nesuvienodintos. VGG16 šaltinyje nurodytas
paketo dydis 16, EfficientNet-B0 — 64; MobileNetV2 suvestinėje paketo dydžio nėra.
Failų dydžiai taip pat priklauso nuo saugojimo formato.</p></section>{cards}</main></html>'''
    (OUTPUT / 'ataskaita.html').write_text(document, encoding='utf-8')
    (OUTPUT / 'README.md').write_text(
        '# Bendri eksperimentų rezultatai\n\n'
        'Keturi modeliai, du duomenų rinkiniai, aštuoni eksperimentai.\n\n'
        'Atidarykite `ataskaita.html` naršyklėje. Grafikai pateikti PNG ir PDF formatais.\n'
        'Visi šeši grafikai viename dokumente: `bendras_palyginimas.pdf`.\n'
        'Skaitiniai duomenys: `bendra_suvestine.csv` ir `bendra_suvestine.json`.\n\n'
        'Kursiokų šaltinių aplankas `kursioku/` neįtraukiamas į Git. '
        'Suvienodintos jų rezultatų eilutės įtrauktos į bendrą suvestinę.\n\n'
        'Klaidų matricoms ir bendroms mokymo kreivėms trūksta kitų modelių '
        'skaitinių matricų / prognozių ir epochų istorijos. Laikai priklauso nuo '
        'paleidimo sąlygų, modelio failo dydis — nuo išsaugojimo formato.\n', encoding='utf-8',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--is-suvestines', action='store_true',
                        help='Atkurti grafikus iš bendri_rezultatai/bendra_suvestine.json be privačių šaltinių.')
    args = parser.parse_args()
    if args.is_suvestines:
        records = validate(json.loads((OUTPUT / 'bendra_suvestine.json').read_text(encoding='utf-8')))
    else:
        records = read_inputs()
    OUTPUT.mkdir(exist_ok=True)
    save_summary(records)
    with PdfPages(OUTPUT / 'bendras_palyginimas.pdf') as report_pdf:
        draw(records, report_pdf)
    report(records)
    print(f'Sukurta: {OUTPUT / "ataskaita.html"}')
    print(f'{len(records)} eksperimentai, {len(FIGURES)} bendros figūros (PNG ir PDF).')


if __name__ == '__main__':
    main()
