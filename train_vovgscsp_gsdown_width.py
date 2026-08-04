#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml
import train_yolo26_module_sweep as sweep

ROOT = Path(os.environ.get('YOLO26_EXP_ROOT', '/root/autodl-tmp/neu-det-yolo26'))
PROJECT = Path(os.environ.get('YOLO26_WIDTH_PROJECT', ROOT / 'runs_vovgscsp_width_e250'))
GEN_DIR = Path(os.environ.get('YOLO26_WIDTH_GEN_DIR', ROOT / 'generated_models_vovgscsp_width_e250'))
REPORT_JSON = PROJECT / 'width_report.json'
REPORT_CSV = PROJECT / 'width_results.csv'
STATUS_JSON = PROJECT / 'status.json'
BASE_YAML = ROOT / 'generated_models_module_stage3_e250' / 'y26n_s3_vovgscsp_gsdown_e250.yaml'


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load_report() -> dict:
    if REPORT_JSON.exists():
        return json.loads(REPORT_JSON.read_text(encoding='utf-8'))
    return {'schema_version': 1, 'created_at': now(), 'experiments': {}}


def make_doc(e: float) -> dict:
    data = yaml.safe_load(BASE_YAML.read_text(encoding='utf-8'))
    for block in data.get('head', []):
        if len(block) >= 4 and block[2] == 'VoVGSCSP':
            c2 = block[3][0]
            block[3] = [c2, True, 1, float(e)]
    return data


def set_outputs() -> None:
    sweep.PROJECT = PROJECT
    sweep.GEN_DIR = GEN_DIR
    sweep.REPORT_JSON = REPORT_JSON
    sweep.REPORT_CSV = REPORT_CSV
    sweep.STATUS_JSON = STATUS_JSON


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--e', type=float, default=0.75)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    name = f"y26n_vovgscsp_gsdown_e{str(args.e).replace('.', '')}_e{args.epochs}"
    spec = {
        'doc': make_doc(args.e),
        'note': f'vovgscsp_gsdown: widen VoVGSCSP hidden ratio e={args.e} (default baseline e=0.5)',
    }

    PROJECT.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    set_outputs()
    sweep.ensure_modules()

    report = load_report()
    report['baseline_reference'] = str(ROOT / 'runs_module_stage3_e250' / 'combo_results.csv')
    report['base_yaml'] = str(BASE_YAML)
    report.setdefault('experiments', {})

    if report['experiments'].get(name, {}).get('status') == 'done' and not args.force:
        print(f'SKIP_DONE {name}')
        sweep.save_status('done', None, [])
        return 0

    sweep.save_status('running', name, [], {'stage': 'vovgscsp_width', 'e': args.e})
    try:
        result = sweep.run_one(name, spec, args.epochs, args.batch, args.imgsz, args.seed, args.force)
    except Exception as exc:
        report['experiments'][name] = {
            'status': 'failed', 'note': spec['note'], 'error': repr(exc), 'failed_at': now()
        }
        sweep.save_report(report)
        sweep.save_status('failed', name, [], {'stage': 'vovgscsp_width', 'e': args.e, 'error': repr(exc)})
        raise

    result['width_e'] = args.e
    report['experiments'][name] = result
    sweep.save_report(report)
    sweep.save_status('done', None, [], {'stage': 'vovgscsp_width', 'e': args.e})
    print('WIDTH_RESULT', name,
          f"val_mAP50={result['val']['map50']:.5f}",
          f"val_mAP50-95={result['val']['map50_95']:.5f}",
          f"test_mAP50={result['test']['map50']:.5f}",
          f"test_mAP50-95={result['test']['map50_95']:.5f}",
          f"params={result.get('params')}",
          f"gflops={result.get('gflops')}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
