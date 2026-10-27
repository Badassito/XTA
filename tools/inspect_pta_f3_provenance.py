import json,pathlib
base=pathlib.Path('H:/F3_10_4_30_2026_8bit_Y_pta_158077_20261019_processfast/clean-native')
d=json.loads((base/'report.json').read_text());print('REPORT TOP',json.dumps({k:v for k,v in d.items() if k not in ('layers','source_full_sha256')},indent=2))
print('layer_count',len(d['layers']));print('conflicting_total',sum(l.get('conflicting_voxels_preserved',0) for l in d['layers']))
print('UNOBSERVED',sum((l.get('unobserved_components_preserved',0) or 0) for l in d['layers']))
for p in [base/'F3_10_4_30_2026_8bit_Y.nrrd',pathlib.Path('H:/XTA_TRAINING/v1/F3_10_4_30_2026_8bit_Y_Input/F3_10_4_30_2026_8bit_Y.nrrd')]:
 print('HEADER',str(p),'bytes',p.stat().st_size)
 with p.open('rb') as f:
  for _ in range(80):
   line=f.readline()
   if not line.strip():break
   print(line.decode(errors='replace').rstrip())
p=pathlib.Path('H:/XTA_TRAINING/v1/F3_10_4_30_2026_8bit_Y_Filtered/edits.json'); d=json.loads(p.read_text());print('EDITS META',json.dumps({k:v for k,v in d.items() if k!='segments'},indent=2))
print('segment_count',len(d['segments']),'foreground_sum',sum(l['foreground_voxels'] for l in d['segments']))

