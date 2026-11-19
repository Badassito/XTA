"""Adopt exact completed detector-layer publications into a scalar TTA sink.

The floor view's stock NRRD writer already decided geometry, suffix, color and
native/LQ rows. A later cutoff restores those decisions without reading source
arrays or rewriting payloads. Scalar Global, summary and terminal run manifests
remain owned by the existing downstream pipeline.
"""
from __future__ import annotations

from concurrent.futures import Future
import json
import math
from pathlib import Path
from typing import Callable


PUBLICATION_SCHEMA = 'xta.nrrd_layer_publication/1'
_FILE_FIELDS = ('size_bytes','modified_time_ns','device','file_id')
_REFERENCE_FIELDS = (
    ('layer_key','key',''), ('model_name','model_name',''),
    ('view_name','view_name',''), ('physical_view_name','physical_view_name',''),
    ('tta_aug_id','aug_id',''), ('tta_angle_deg','angle_deg',0.),
    ('view_family','view_family',''), ('source','source',''),
    ('mask_kind','mask_kind',''), ('pass_index','pass_index',0),
    ('interpolation_walk_back_index','interpolation_walk_back_index',0),
    ('interpolation_candidate_index','interpolation_candidate_index',0),
    ('tile_config_id','tile_config_id',''), ('tile_acceptance','tile_acceptance',''),
    ('stage','stage',''), ('description','description',''),
    ('layer_role','layer_role','additive_component'),
)


def layer_payload_identity(path):
    """Cheap immutable-publication identity; no scientific payload hash or read."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Prepublished NRRD payload is missing or not a regular file: {path}')
    value = path.stat()
    return dict(size_bytes=int(value.st_size),modified_time_ns=int(value.st_mtime_ns),
                device=int(value.st_dev),file_id=int(value.st_ino))


def _shape(value, description):
    if (not isinstance(value,(list,tuple)) or len(value)!=3
            or any(type(item) is not int or item<1 for item in value)):
        raise ValueError(f'{description} requires three positive integer dimensions')
    return tuple(value)


def _plain_token(value, description):
    if (not isinstance(value,str) or not value or value in {'.','..'}
            or Path(value).name!=value or '/' in value or '\\' in value
            or Path(value).is_absolute() or Path(value).drive):
        raise ValueError(f'{description} must be one plain relative name')
    return value


def _check_file(path, identity):
    if (not isinstance(identity,dict) or any(type(identity.get(key)) is not int or identity[key]<0
            for key in _FILE_FIELDS)):
        raise ValueError('Prepublished layer is missing its exact file-stat identity')
    observed=layer_payload_identity(path)
    if any(observed[key]!=identity[key] for key in _FILE_FIELDS):
        raise ValueError(f'Prepublished NRRD payload changed after publication: {path}')


def _check_layout(row, sink):
    # A run never mixes layouts: rows carry the cropped placement keys exactly when the
    # adopting sink writes cropped layers (NrrdLayerSink.layer_layout; historical sinks are full).
    cropped=getattr(sink,'layer_layout','full')=='cropped'
    if ('stored_offset_tyx' in row)!=cropped or ('stored_layout' in row)!=cropped:
        raise ValueError('Prepublished layer stored layout differs from the scalar sink layer layout')


def _check_row(row, ref, *, output_shape, native, native_row, provenance):
    if not isinstance(row,dict):
        raise ValueError('Prepublished layer manifest row must be an object')
    if _shape(row.get('output_shape_tyx'),'Published output grid') != tuple(output_shape):
        raise ValueError('Prepublished layer output grid differs from the scalar sink')
    if _shape(row.get('backing_shape_tyx'),'Published backing grid') != tuple(ref.shape):
        raise ValueError('Prepublished layer backing grid differs from its reference')
    stored=_shape(row.get('stored_shape_tyx'),'Published stored grid')
    if any(actual>bound for actual,bound in zip(stored,output_shape)):
        raise ValueError('Prepublished layer stored crop exceeds its output grid')
    # Cropped-layout rows place their stored box; rows without these keys store the full grid.
    offset=row.get('stored_offset_tyx')
    if 'stored_offset_tyx' in row:
        if (not isinstance(offset,(list,tuple)) or len(offset)!=3
                or any(type(item) is not int or item<0 for item in offset)):
            raise ValueError('Prepublished layer stored offset requires three nonnegative integers')
        if any(start+size>bound for start,size,bound in zip(offset,stored,output_shape)):
            raise ValueError('Prepublished layer stored crop exceeds its output grid')
    if 'stored_layout' in row:
        layout=row.get('stored_layout')
        if layout not in ('segment_bounding_box','full_reference'):
            raise ValueError('Prepublished layer has an unsupported stored layout')
        if layout=='full_reference' and (stored!=tuple(output_shape)
                or (offset is not None and tuple(offset)!=(0,0,0))):
            raise ValueError('Prepublished full_reference layer stored grid is not its complete output grid')
    if type(row.get('empty_segment')) is not bool:
        raise ValueError('Prepublished layer lacks its completed empty-mask decision')
    if row.get('exported_axes') != '(X, Y, t)':
        raise ValueError('Prepublished layer has unsupported exported axes')
    for field,attribute,default in _REFERENCE_FIELDS:
        actual,expected=row.get(field),getattr(ref,attribute,default)
        if (isinstance(actual,bool) and not isinstance(expected,bool)) or actual!=expected:
            raise ValueError(f'Prepublished layer reference identity mismatch: {field}')
    expected_op=str(getattr(ref,'recomposition_op','union') if native else
                    getattr(ref,'low_quality_recomposition_op',getattr(ref,'recomposition_op','union')))
    if row.get('recomposition_op')!=expected_op:
        raise ValueError('Prepublished layer recomposition differs from its scalar reference')
    if not native and row.get('full_quality_recomposition_op')!=str(getattr(ref,'recomposition_op','union')):
        raise ValueError('Prepublished mirror lost its native recomposition identity')
    for field in ('filename','suffix','segment_name','segment_color_rgb','model_name','layer_key'):
        if row.get(field)!=native_row.get(field):
            raise ValueError(f'Prepublished mirror/native identity mismatch: {field}')
    provenance=json.loads(json.dumps(provenance,allow_nan=False))
    for key,value in provenance.items():
        if row.get(key)!=value:
            raise ValueError(f'Prepublished layer provenance mismatch: {key}')


def _validated_record(sink, ref, record, *, basename_for_suffix, provenance_for_directory):
    # JSON snapshot also rejects nonfinite metadata and severs every source alias.
    try:
        value=json.loads(json.dumps(record,allow_nan=False))
    except (ValueError,TypeError) as error:
        raise ValueError('Prepublished layer record is not strict JSON metadata') from error
    if not isinstance(value,dict) or value.get('schema')!=PUBLICATION_SCHEMA:
        raise ValueError('Unsupported prepublished NRRD layer record')
    if value.get('run_id')!=sink.run_id or not isinstance(sink.run_id,str) or not sink.run_id:
        raise ValueError('Prepublished layer belongs to another scalar run')
    if value.get('stem')!=sink.stem:
        raise ValueError('Prepublished layer belongs to another scalar output stem')
    if _shape(value.get('output_shape_tyx'),'Publication grid')!=tuple(sink.output_shape):
        raise ValueError('Prepublished layer belongs to another scalar output grid')
    if (getattr(ref,'mask_kind','')!='yolo' or getattr(ref,'layer_role','additive_component')!='additive_component'
            or getattr(ref,'recomposition_op','union')!='union'):
        raise ValueError('Only direct additive detector layers can be adopted before scalar downstream work')
    row=value.get('native')
    if not isinstance(row,dict):
        raise ValueError('Prepublished layer has no native manifest row')
    allocator=value.get('allocator')
    if not isinstance(allocator,dict):
        raise ValueError('Prepublished layer is missing its scalar suffix/color allocator state')
    base=allocator.get('requested_suffix')
    if not isinstance(base,str) or not base:
        raise ValueError('Prepublished layer has an invalid requested suffix')
    occurrence=allocator.get('occurrence')
    if type(occurrence) is not int or occurrence<1:
        raise ValueError('Prepublished layer has an invalid suffix occurrence')
    suffix=base if occurrence==1 else f'{base}_{occurrence:02d}'
    filename=_plain_token(row.get('filename'),'Published NRRD filename')
    if (row.get('suffix')!=suffix or row.get('segment_name')!=f'{sink.stem}_{suffix}'
            or filename!=basename_for_suffix(sink.stem,suffix)):
        raise ValueError('Prepublished layer physical/logical suffix identity differs')
    color=allocator.get('color_rgb')
    if (not isinstance(color,list) or len(color)!=3 or any(isinstance(item,bool)
            or not isinstance(item,(int,float)) or not math.isfinite(item) or not 0<=item<=1 for item in color)
            or row.get('segment_color_rgb')!=[round(float(item),6) for item in color]):
        raise ValueError('Prepublished layer lost its exact scalar color allocation')
    shards=row.get('z_shards')
    if type(shards) is not int or shards<1 or row.get('z_shards_policy')!='execution_time':
        raise ValueError('Prepublished layer lacks completed execution-time writer metadata')
    _check_layout(row,sink)
    _check_row(row,ref,output_shape=sink.output_shape,native=True,native_row=row,
               provenance=provenance_for_directory(ref,relative_to=sink.nrrd_dir))
    mirrors=value.get('low_quality')
    if not isinstance(mirrors,dict):
        raise ValueError('Prepublished layer has invalid low-quality manifest rows')
    specs={str(spec.token):spec for spec in sink.low_quality_specs}
    expected_tokens=set(specs) if bool(getattr(ref,'mirror_low_quality',True)) else set()
    if set(mirrors)!=expected_tokens:
        raise ValueError('Prepublished mirrors differ from the scalar output selection')
    files=value.get('files')
    if not isinstance(files,dict) or set(files.get('low_quality',{}))!=expected_tokens:
        raise ValueError('Prepublished layer file identities differ from its mirror rows')
    path=sink.nrrd_dir/filename
    _check_file(path,files.get('native'))
    for token,mirror in mirrors.items():
        _plain_token(token,'Published downbin token')
        spec=specs[token]
        shape=_shape(list(spec.output_shape_t_y_x),'Scalar mirror grid')
        if not isinstance(mirror,dict):
            raise ValueError('Prepublished layer manifest row must be an object')
        _check_layout(mirror,sink)
        _check_row(mirror,ref,output_shape=shape,native=False,native_row=row,
            provenance=provenance_for_directory(ref,relative_to=sink._lq_nrrd_dir(spec)))
        for field,expected in (('downbin_token',token),('downbin_value',str(spec.raw_value)),
                                ('downbin_scale',float(spec.scale))):
            if mirror.get(field)!=expected:
                raise ValueError(f'Prepublished mirror specification differs: {field}')
        _check_file(sink._lq_nrrd_dir(spec)/filename,files['low_quality'][token])
    return value,path,base,occurrence,tuple(float(item) for item in color)


def adopt_published_layer(sink, ref, record, *, basename_for_suffix: Callable,
                          provenance_for_directory: Callable):
    """Restore one stock publication atomically, allowing unrelated view writes.

    No payload is opened or rewritten. File-stat snapshots fence accidental
    mutation, while the parent's hashed view journal binds record/ref ownership.
    Completed futures preserve the stock sink's progress and re-export behavior.
    """
    if ref is None and record is None:
        return None
    if ref is None or record is None:
        raise ValueError('Prepublished layer reference and record must both be present or absent')
    value,path,base,occurrence,color=_validated_record(sink,ref,record,
        basename_for_suffix=basename_for_suffix,provenance_for_directory=provenance_for_directory)
    row=value['native']
    with sink._lock:
        name=row['filename'].casefold()
        if (name in sink._physical_names or any(str(item.get('filename','')).casefold()==name
                or item.get('suffix')==row['suffix'] for item in sink._manifest)):
            raise ValueError('Prepublished layer conflicts with a scalar output already registered')
        # All rejecting work precedes mutation. Pending fallback views already
        # reserved their own names/colors under this same stock sink lock.
        sink._physical_names[name]=row['segment_name']
        sink._suffix_counts[base]=max(int(sink._suffix_counts.get(base,0)),occurrence)
        sink._segment_colors_in_use.add(color)
        sink._manifest.append(row)
        for token,mirror in value['low_quality'].items():
            sink._lq_manifests.setdefault(token,[]).append(mirror)
        completed=Future()
        completed.set_result(path)
        sink._futures.append(completed)
        sink._publication_records.setdefault((str(ref.key),base),[]).append((completed,value))
    return path


__all__=['PUBLICATION_SCHEMA','adopt_published_layer','layer_payload_identity']
