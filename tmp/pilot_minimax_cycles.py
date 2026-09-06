"""Independent pilots of a student missing one teacher-memory direction."""
import os, json, time
from pathlib import Path
import numpy as np
os.environ['PMT_NO_SHOW']='1'
root=Path(__file__).resolve().parents[1]
source=root/'paper_figure/notebook/05_settled_free_energy_workbench.py'
ns={'__file__':str(source)}
exec(compile(source.read_text(encoding='utf-8').split('# %% cached experiment')[0],str(source),'exec'),ns)
ns['CFG'].update(max_relax_time=300000.,max_step=500.,cycle_h=0.005)
all_rows=[]
start=time.perf_counter()
for seed in (9010,9011,9012):
    net=ns['make_network'](seed)
    Q=np.linalg.qr(net['patterns'][:,:5])[0]; P=Q@Q.T
    W0=P/(1-np.diag(P))[:,None]; np.fill_diagonal(W0,0)
    rng=np.random.default_rng(seed+10)
    for protocol in ('intrinsic','random'):
        W=W0.copy(); x=ns['normalize'](rng.normal(size=48)); y=np.zeros(48)
        for event in range(400):
            ref=ns['benchmark'](W,net['U'])
            if protocol=='intrinsic':
                a=ns['relax'](net,W,x,y); x,y=a['x'],a['y']
            else:
                x=net['U']@ns['normalize'](rng.normal(size=6));y=ref['B']@x
                a=dict(settled=True,distance=0.,time=0.)
            G=ns['update_direction'](W,y); W+=0.005*G
            new=ns['benchmark'](W,net['U'])
            all_rows.append(dict(seed=seed,protocol=protocol,event=event,settled=a['settled'],
                time=a['time'],distance=float(a['distance']),Fmax=ref['Fmax'],after=new['Fmax'],
                ratio=(ref['Fmax']-new['Fmax'])/(0.005*np.sum(G*G))))
        rows=[a for a in all_rows if a['seed']==seed and a['protocol']==protocol]
        print(seed,protocol,'settled',sum(a['settled'] for a in rows),'maxdist',max(a['distance'] for a in rows),
              'F',rows[0]['Fmax'],rows[-1]['after'],'minratio',min(a['ratio'] for a in rows),
              'seconds',time.perf_counter()-start,flush=True)
    (root/'tmp/minimax_cycle_pilot.json').write_text(json.dumps(all_rows,indent=2))
