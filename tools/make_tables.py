"""Build paper tables straight from per-episode CSVs. No number is typed by hand.

Each row of the table is one CSV (one controller / scenario) with one row per
evaluation episode. Rows are paired with the reference row on ``episode_seed``
(common random numbers), so the test is paired.

Reported per metric: mean +- sample std (ddof=1), 95% t-interval of the mean,
and against the reference: mean paired difference, paired t-test p-value and
Wilcoxon signed-rank p-value, Holm-corrected across the rows of the table.

    python tools/make_tables.py --name table1_grid8 --ref "Webster=outputs/results/baselines/grid8_webster.csv" \
        "Fixed-time=outputs/results/baselines/grid8_fixed_time.csv" \
        "Local supervisor=outputs/results/supervisor/eval_trips.csv"
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_METRICS = ['mean_time_in_system_all', 'mean_travel_time', 'mean_time_loss', 'mean_waiting_time',
                   'throughput_veh_h', 'mean_queue']
OUT_DIR = 'outputs/tables'


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summarise(rows, ref, metrics):
    ref_label, ref_df = ref
    out = []
    for label, df in [ref] + rows:
        rec = {'row': label, 'n': len(df)}
        for m in metrics:
            x = df[m].to_numpy(dtype=float)
            mean, sd = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
            half = stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x)) if len(x) > 1 else np.nan
            rec[f'{m}_mean'], rec[f'{m}_std'] = mean, sd
            rec[f'{m}_ci_lo'], rec[f'{m}_ci_hi'] = mean - half, mean + half
            if label != ref_label:
                pair = df[['episode_seed', m]].merge(ref_df[['episode_seed', m]], on='episode_seed',
                                                     suffixes=('', '_ref'))
                d = pair[m] - pair[f'{m}_ref']
                rec[f'{m}_diff'] = d.mean()
                rec[f'{m}_n_paired'] = len(d)
                rec[f'{m}_p_t'] = stats.ttest_rel(pair[m], pair[f'{m}_ref']).pvalue if len(d) > 1 else np.nan
                rec[f'{m}_p_wilcoxon'] = (stats.wilcoxon(d).pvalue
                                          if len(d) > 1 and np.any(d != 0) else np.nan)
        out.append(rec)
    res = pd.DataFrame(out)
    for m in metrics:
        col = f'{m}_p_wilcoxon'
        if col in res:
            mask = res[col].notna()
            res.loc[mask, f'{m}_p_holm'] = holm(res.loc[mask, col])
    return res


def to_markdown(res, metrics, ref_label):
    head = '| Row | n | ' + ' | '.join(metrics) + ' |'
    lines = [head, '|' + '---|' * (len(metrics) + 2)]
    for _, r in res.iterrows():
        cells = []
        for m in metrics:
            c = f"{r[f'{m}_mean']:.2f} ± {r[f'{m}_std']:.2f} [{r[f'{m}_ci_lo']:.2f}, {r[f'{m}_ci_hi']:.2f}]"
            if r['row'] != ref_label and not pd.isna(r.get(f'{m}_p_holm', np.nan)):
                c += f" (Δ {r[f'{m}_diff']:+.2f}, p={r[f'{m}_p_holm']:.3g})"
            cells.append(c)
        lines.append(f"| {r['row']} | {int(r['n'])} | " + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append(f'mean ± std (ddof=1) [95% t-CI]; Δ and Holm-corrected Wilcoxon p vs. "{ref_label}", '
                 'paired on episode seed.')
    return '\n'.join(lines)


def to_latex(res, metrics, ref_label):
    cols = 'l' + 'c' * len(metrics)
    lines = [f'\\begin{{tabular}}{{{cols}}}', '\\toprule',
             'Row & ' + ' & '.join(m.replace('_', ' ') for m in metrics) + ' \\\\', '\\midrule']
    for _, r in res.iterrows():
        cells = []
        for m in metrics:
            c = f"{r[f'{m}_mean']:.2f} $\\pm$ {r[f'{m}_std']:.2f}"
            p = r.get(f'{m}_p_holm', np.nan)
            if r['row'] != ref_label and not pd.isna(p) and p < 0.05:
                c += '$^{*}$'
            cells.append(c)
        lines.append(f"{r['row']} & " + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}',
              f'% * Holm-corrected paired Wilcoxon p < 0.05 vs. {ref_label}; n per row in the CSV.']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--ref', required=True, help='label=csv of the reference row')
    p.add_argument('rows', nargs='*', help='label=csv')
    p.add_argument('--metrics', default=','.join(DEFAULT_METRICS))
    args = p.parse_args()

    def load(spec):
        label, path = spec.split('=', 1)
        return label, pd.read_csv(path)

    metrics = args.metrics.split(',')
    ref = load(args.ref)
    res = summarise([load(r) for r in args.rows], ref, metrics)
    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(f'{OUT_DIR}/{args.name}.csv', index=False)
    md = to_markdown(res, metrics, ref[0])
    open(f'{OUT_DIR}/{args.name}.md', 'w', encoding='utf-8').write(md + '\n')
    open(f'{OUT_DIR}/{args.name}.tex', 'w', encoding='utf-8').write(to_latex(res, metrics, ref[0]) + '\n')
    print(md.encode("ascii", "replace").decode())


if __name__ == '__main__':
    main()
