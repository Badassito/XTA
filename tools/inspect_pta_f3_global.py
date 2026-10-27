import json,pathlib,numpy as np
p=pathlib.Path('H:/XTA_TRAINING/v1/F3_10_4_30_2026_8bit_Y_Filtered');d=json.loads((p/'edits.json').read_text())
for i,s in enumerate(d['segments']):
 if 'Global' in s['name'] or 'global' in s['name']:print('GLOBAL',i,json.dumps(s));f=s['file']
a=np.load(p/f); print('npzkeys',a.files)
for k in a.files:
 x=a[k]; print(k,x.shape,str(x.dtype),str(x)[:300])
