#!/usr/bin/env python3
"""Verify a staged PTA video/mask pair against its completed preparation receipt.

Standard library only. Verification compares content hashes, not the original
absolute paths or mtimes, so a staged or mirrored pair can be verified. It does
not rerun numerical reconstruction or certify anatomical accuracy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_identity(path):
    path=Path(path)
    if not path.is_file():raise ValueError(f'Missing regular file: {path}')
    before=path.stat();h=hashlib.sha256()
    if before.st_size<=0:raise ValueError(f'Empty staged file: {path}')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError(f'File changed during verification: {path}')
    return {'bytes':before.st_size,'sha256':h.hexdigest()}


def verify(receipt_path,video,mask,expected_mode,expected_output_stem=None):
    receipt_path=Path(receipt_path);video=Path(video);mask=Path(mask)
    receipt_id=file_identity(receipt_path)
    receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if receipt.get('format')!='pta-preview-input/1' or receipt.get('status')!='complete':
        raise ValueError('Preparation receipt is not a completed supported preparation.')
    identity=receipt.get('identity',{})
    if identity.get('mode')!=expected_mode:
        raise ValueError(f'Prepared mode {identity.get("mode")!r} differs from expected {expected_mode!r}.')
    label_stem=identity.get('output_stem',identity.get('stem'))
    if expected_output_stem and label_stem!=expected_output_stem:
        raise ValueError('Prepared output stem differs from the requested label.')
    if video.stem!=mask.stem:
        raise ValueError('Staged video and mask must have identical basenames for PTA pairing.')
    actual_video=file_identity(video);actual_mask=file_identity(mask)
    for role,actual,recorded in [('video',actual_video,identity.get('intensity',{})),
                                  ('mask',actual_mask,receipt.get('mask',{}))]:
        if not recorded.get('sha256'):
            raise ValueError(f'Receipt has no hashed {role} identity; prepare with continuous_field or a hashed reference/copy mode.')
        if any(actual[k]!=recorded.get(k) for k in ('bytes','sha256')):
            raise ValueError(f'Staged {role} content differs from the preparation receipt.')
    # upscale and native policies are fixed by the preparation tool itself. Every
    # other mode records its reconstruction contract under the mode name, and
    # that contract must name the receipt's policy.
    contract=identity.get(expected_mode)
    if expected_mode not in {'upscale','native'} and (
            not isinstance(contract,dict) or not identity.get('policy')
            or contract.get('policy')!=identity['policy']):
        raise ValueError('Prepared policy identity is missing or inconsistent.')
    if file_identity(receipt_path)!=receipt_id:
        raise ValueError('Preparation receipt changed during verification.')
    return {'status':'verified','verified_utc':datetime.now(timezone.utc).isoformat(),
            'receipt':str(receipt_path.resolve()),'receipt_sha256':receipt_id['sha256'],
            'prepared_mode':identity['mode'],'policy':identity.get('policy'),
            'generation':identity.get('generation'),
            'staged_video':str(video.resolve()),'staged_mask':str(mask.resolve()),
            'video':actual_video,'mask':actual_mask,
            'scope':'Content and mode pairing. Code/runtime versions are provenance only; not anatomical or training validation.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True,help='preparation.json written by prepare_pta_input.py')
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--expected-mode',required=True,help='Preparation mode the receipt must record, e.g. continuous_field')
    p.add_argument('--expected-output-stem')
    p.add_argument('--output',type=Path,help='Optional new verification receipt outside PTA input/output directories')
    a=p.parse_args(argv)
    result=verify(a.receipt,a.video,a.mask,a.expected_mode,a.expected_output_stem)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
