"""Reproduce the numerical results of Shared Temporal Models and Cross-Link Graphs."""
import argparse
import importlib.metadata
import platform
import shutil
from pathlib import Path

import numpy as np
import torch

from data import ROOT, load_configs, prepare
from evaluation import arrays_from_csv, summarize, verify_reference
from models import (DEFAULTS, NEURAL_MODELS, GRAPH_ABLATIONS,
                    make_neural, make_statistical, seed_everything)


def train(gateway, output):
    data = prepare(gateway)
    configs = load_configs(gateway)
    if set(configs) != set(DEFAULTS):
        raise ValueError('The frozen configuration must specify all nine base models')
    Xtr, ytr = data['train']
    Xv, yv = data['val']
    Xt, yt = data['test']
    pipeline, names = data['pipeline'], data['target_names']
    truth = pipeline.inverse_transform_targets(yt)
    times = data['origin_timestamps']['test']
    predictions = {}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / 'configs' / f'{gateway}.csv', output / 'configuration.csv')
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'torch']}
    (output / 'environment.txt').write_text(
        f'Python: {platform.python_version()}\nDevice: CPU\nThreads: 6\n'
        'Seed: 42\nHistory: 24\nHorizon: 1\nEpoch cap: 64\nPatience: 10\n'
        + '\n'.join(f'{key}: {value}' for key, value in versions.items()) + '\n')
    for name in list(configs) + list(GRAPH_ABLATIONS):
        seed_everything(42)
        cfg = configs['PhysicalAdaptive_STGNN' if name in GRAPH_ABLATIONS else name]
        neural = name in NEURAL_MODELS or name in GRAPH_ABLATIONS
        model = (make_neural(name, cfg, data, 1, 'cpu') if neural else
                 make_statistical(name, cfg, data, 1))
        if neural:
            model.fit(Xtr, ytr, Xv, yv, epochs=64, batch_size=cfg['batch_size'],
                      patience=10, target_scale=data['target_scale'])
            prediction = model.predict(Xt)
        else:
            model.fit(Xtr, ytr, training_segments=data['training_segments'])
            prediction = model.predict(Xt, 1)
        predictions[name] = pipeline.inverse_transform_targets(prediction)
        # One archive per gateway, atomically refreshed after every completed model.
        temporary = output / 'predictions.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, y_true=truth, target_names=np.asarray(names),
                                target_timestamps=times,
                                forecast_origin_timestamps=times - np.timedelta64(1, 'h'),
                                rssi_unit=np.asarray('dBm'), **predictions)
        temporary.replace(output / 'predictions.npz')
        summarize(truth, predictions, names, times, output, paired=False)
        print(f'{gateway} {name}: MAE={np.abs(predictions[name] - truth).mean():.4f} dB', flush=True)
        del model
    summarize(truth, predictions, names, times, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'metrics', 'train'])
    parser.add_argument('--gateway', choices=['A', 'C', 'all'], default='all')
    parser.add_argument('--output', type=Path, help='New output root (default: outputs/ACTION)')
    parser.add_argument('--source', type=Path,
                        help='For metrics: completed training root containing A/C predictions.npz')
    args = parser.parse_args()
    if args.source and args.action != 'metrics':
        parser.error('--source is only available with metrics')
    gateways = ['A', 'C'] if args.gateway == 'all' else [args.gateway]
    output = args.output or ROOT / 'outputs' / args.action
    if args.action != 'check':
        for gateway in gateways:
            if (output / gateway).exists():
                parser.error(f'{output / gateway} exists; choose another --output')
    torch.set_num_threads(6)
    for gateway in gateways:
        if args.action == 'check':
            data = prepare(gateway)
            if set(load_configs(gateway)) != set(DEFAULTS):
                raise ValueError('Incomplete model configuration')
            print(gateway, data['quality_report']['valid_windows'])
        elif args.action == 'train':
            train(gateway, output / gateway)
        else:
            if args.source:
                with np.load(args.source / gateway / 'predictions.npz', allow_pickle=False) as saved:
                    truth = saved['y_true']
                    names, times = saved['target_names'].tolist(), saved['target_timestamps']
                    predictions = {k: saved[k] for k in saved.files
                                   if k != 'y_true' and saved[k].shape == truth.shape}
            else:
                truth, predictions, names, times = arrays_from_csv(
                    ROOT / 'reference' / gateway / 'predictions.csv.gz')
            summarize(truth, predictions, names, times, output / gateway)
            if not args.source:
                verify_reference(output / gateway, ROOT / 'reference' / gateway)
            print(f'{gateway}: {len(predictions)} models, {len(names)} links; results in {output / gateway}')


if __name__ == '__main__':
    main()
