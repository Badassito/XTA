"""Export and disassemble task-local CuPy cache binaries without using a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Existing verification evidence directory')
    args = parser.parse_args()
    root = args.output.resolve()
    target = root / 'disassembly'
    target.mkdir(exist_ok=True)
    command = shutil.which('cuobjdump')
    if command is None:
        raise RuntimeError('cuobjdump is unavailable')
    rows = []
    seen = set()
    for source in root.rglob('*.cubin'):
        data = source.read_bytes()
        if not re.match(rb'[0-9a-f]{40}', data):
            continue  # Exported ELF binaries are not CuPy cache entries.
        offset = data.find(b'\x7fELF', 40, 65536)
        if offset < 0:
            continue
        binary = data[offset:]
        digest = hashlib.sha256(binary).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        row = dict(cache=str(source), raw_binary_sha256=digest, raw_bytes=len(binary),
                   header_bytes=offset, appended_zero_bytes=0)
        header = struct.unpack('<16sHHIQQQIHHHHHH', binary[:64])
        extent = max(header[5] + header[9] * header[10], header[6] + header[11] * header[12])
        if len(binary) == extent - 1:
            # In this CuPy/NVRTC build the exported bytes omit the last NUL.
            # It lies in the final program-header p_align field; all section
            # bytes and instructions must already be present. Preserve raw
            # cache bytes and document the minimal file-format repair.
            if header[9] != 56 or header[10] < 1 or header[11] != 64:
                raise RuntimeError('Unexpected ELF layout: ' + str(source))
            for index in range(header[12]):
                start = header[6] + index * header[11]
                section = struct.unpack('<IIQQQQIIQQ', binary[start:start+64])
                if section[1] != 8 and section[4] + section[5] > len(binary):
                    raise RuntimeError('ELF is missing section data: ' + str(source))
            binary += b'\0'
            row['appended_zero_bytes'] = 1
        elif len(binary) < extent:
            raise RuntimeError('Unexpected truncated ELF: ' + str(source))
        exported = target / (digest[:24] + '.cubin')
        exported.write_bytes(binary)
        result = subprocess.run([command, '--dump-sass', str(exported)], capture_output=True, text=True)
        listing = result.stdout + result.stderr
        sass = exported.with_suffix('.sass.txt')
        sass.write_text(listing, encoding='utf-8')
        row.update(cubin=str(exported), sass=str(sass), returncode=result.returncode,
                   disassembled='Function :' in listing,
                   functions=re.findall(r'Function : ([^\r\n]+)', listing),
                   instructions={name: len(re.findall(pattern, listing)) for name, pattern in
                                 (('redux_or', r'\bREDUX\.OR\b'), ('shared_load',r'\bLDS\b'),
                                  ('shared_store',r'\bSTS\b'), ('fmul',r'\bFMUL\b'),
                                  ('fadd',r'\bFADD\b'), ('ffma',r'\bFFMA\b'))})
        rows.append(row)
    report = dict(tool=command, scope='Disassembly of actual cached NVRTC binaries; no nvcc recompilation.',
                  export_note='Original cache bytes preserved. A missing trailing zero in an ELF program header is restored only after proving every section is already complete. This does not change instruction sections.',
                  entries=rows)
    (root / 'disassembly.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'binaries': len(rows), 'disassembled': sum(r['disassembled'] for r in rows)}))


if __name__ == '__main__':
    main()
