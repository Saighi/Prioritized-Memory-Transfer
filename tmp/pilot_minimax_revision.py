"""Read existing equations, evaluate revised protocol choices, write pilot diagnostics only."""
import os
import json
import time
from pathlib import Path
import numpy as np

os.environ['PMT_NO_SHOW'] = '1'
root = Path(__file__).resolve().parents[1]
source = root / 'paper_figure/notebook/05_settled_free_energy_workbench.py'
namespace = {'__file__': str(source)}
exec(compile(source.read_text(encoding='utf-8').split('# %% cached experiment')[0], str(source), 'exec'), namespace)
C = namespace['CFG']
C.update(calibration_snapshots=(40, 80, 160, 320), event_sizes=(0.01,0.003,0.001),
         max_relax_time=300000.0, max_step=500.0, cycles=120, cycle_h=0.003)
# Separate pilot seeds from the final ensemble.
output = []
start = time.perf_counter()
for seed in (9010, 9011, 9012):
    net = namespace['make_network'](seed)
    snapshots = namespace['calibrate'](net)
    rng = np.random.default_rng(seed+50)
    for stage, W in snapshots.items():
        for init in range(3):
            final = namespace['relax'](net,W,namespace['normalize'](rng.normal(size=48)),np.zeros(48))
            ref = namespace['benchmark'](W,net['U'])
            G = namespace['update_direction'](W,final['y'])
            pred = np.sum(G*G)
            ratios = [(ref['Fmax']-namespace['benchmark'](W+h*G,net['U'])['Fmax'])/(h*pred)
                      for h in C['event_sizes']]
            row=dict(seed=seed,stage=stage,kind='selection',init=init,
                     **{k:float(v) if isinstance(v,np.floating) else v for k,v in final.items() if k not in ('x','y')},
                     ratios=ratios,gap=ref['gap'])
            output.append(row)
        print(seed,stage,[(a['settled'],round(a['distance'],5),a['time']) for a in output[-3:]],flush=True)
    W=snapshots[80].copy()
    x=namespace['normalize'](rng.normal(size=48)); y=np.zeros(48)
    for cycle in range(C['cycles']):
        final=namespace['relax'](net,W,x,y)
        x,y=final['x'],final['y']
        ref=namespace['benchmark'](W,net['U'])
        W+=C['cycle_h']*namespace['update_direction'](W,y)
        output.append(dict(seed=seed,kind='cycle',cycle=cycle,settled=final['settled'],
                           distance=float(final['distance']),time=final['time'],Fmax=ref['Fmax']))
    rows=[a for a in output if a['seed']==seed and a['kind']=='cycle']
    print('cycles',seed,'settled',sum(a['settled'] for a in rows),'max distance',max(a['distance'] for a in rows),
          'energies',rows[0]['Fmax'],rows[-1]['Fmax'],'seconds',time.perf_counter()-start,flush=True)
    (root/'tmp/minimax_revision_pilot.json').write_text(json.dumps(output,indent=2))
