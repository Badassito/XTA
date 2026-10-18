"""Independent fixtures for the nonblocking, metadata-only diagnostic reader."""
import hashlib
import copy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from tools import collect_sam_runtime_receipts as collector


def add_member(stream, name, payload):
    header = tarfile.TarInfo(name)
    header.size = len(payload)
    stream.write(header.tobuf(format=tarfile.PAX_FORMAT))
    offset = stream.tell()
    stream.write(payload)
    stream.write(b"\0" * (-len(payload) % 512))
    return offset


def transaction(stream, members, number=1, replace=False, commit=True, corrupt_commit=False):
    identity = f"{number:032x}"
    offsets, rows = {}, []
    for name, payload in members.items():
        offsets[name] = add_member(stream, "DATA/" + identity + "/" + name, payload)
        rows.append(dict(name=name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    if commit:
        value = dict(schema="xta.artifact-archive/1", transaction=identity, replace=replace, members=rows)
        value["sha256"] = "0" * 64 if corrupt_commit else hashlib.sha256(collector.encoded(value)).hexdigest()
        add_member(stream, "COMMIT/" + identity + ".json", collector.encoded(value))
    return offsets


def receipt(device=1):
    return dict(schema="xta.sam-raw-tracker-run/1", run_id="run-one",
                crop_xyxy=[3, 5, 101, 109], expected_frames=[3, 5, 8],
                timings=dict(worker_started_monotonic=12.5, tracker_seconds=.4),
                cpu_task_retirement=dict(execution_device_id=device, worker_pid=101),
                feature_cache_before=dict(hits=3), raw_masks=dict(path="must-not-copy.npz"))


def index(device=1):
    return collector.encoded(dict(groups={}, masks={"ignored": {"offset": 0}},
        runs={"run-one": dict(run_id="run-one", group_id="family", expected_frames=[3, 5, 8],
                              runtime_receipt=receipt(device))}))


def collect(path):
    destination = io.BytesIO()
    report = collector.collect_archive(path, collector.RowWriter(destination))
    return report, [json.loads(line) for line in destination.getvalue().splitlines()]


def test_committed_receipts_skip_binary_and_do_not_touch_source_or_lock(tmp_path):
    path = tmp_path / "sam-artifacts.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/family/evidence/index.json": index(),
                             "sam_interpolation/family/evidence/masks.bin": b"not-json" * 100_000})
        stream.write(bytes(1024))
    before = path.read_bytes()
    report, rows = collect(path)
    assert report["rows"] == 1 and report["indexes_read"] == 1
    assert report["json_bytes_read"] == len(index())
    assert rows[0]["runtime_receipt"]["expected_frames"] == [3, 5, 8]
    assert rows[0]["runtime_receipt"]["feature_cache_before"] == {"hits": 3}
    assert "raw_masks" not in rows[0]["runtime_receipt"]
    assert path.read_bytes() == before
    assert not Path(str(path) + ".lock").exists()


def test_uncommitted_and_partial_tail_never_publish_rows(tmp_path):
    path = tmp_path / "live.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/old/index.json": index(0)})
        transaction(stream, {"sam_interpolation/new/index.json": index(1)}, number=2, commit=False)
        stream.write(b"partial header")
    report, rows = collect(path)
    assert report["tail_status"] == "incomplete_header"
    assert len(rows) == 1
    assert rows[0]["runtime_receipt"]["cpu_task_retirement"]["execution_device_id"] == 0


def test_racing_complete_header_preserves_authenticated_prefix(tmp_path):
    path = tmp_path / "live.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/old/index.json": index()})
        stream.write(b"incomplete-writing" + b"x" * 1024)
    report, rows = collect(path)
    assert report["tail_status"] == "stopped_before_unverified_tail"
    assert len(rows) == 1


def test_checksums_cover_both_commit_and_selected_payload(tmp_path):
    bad_commit = tmp_path / "commit.tar"
    with bad_commit.open("wb") as stream:
        transaction(stream, {"sam_interpolation/x/index.json": index()}, corrupt_commit=True)
    report, rows = collect(bad_commit)
    assert not rows and report["tail_status"] == "stopped_before_unverified_tail"
    bad_payload = tmp_path / "payload.tar"
    with bad_payload.open("w+b") as stream:
        offsets = transaction(stream, {"sam_interpolation/x/index.json": index()})
        stream.seek(offsets["sam_interpolation/x/index.json"])
        stream.write(b"!")
    report, rows = collect(bad_payload)
    assert not rows and "checksum mismatch" in report["errors"][0]["error"]


def test_replaced_logical_index_uses_latest_committed_version(tmp_path):
    path = tmp_path / "replaced.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/x/index.json": index(0)})
        transaction(stream, {"sam_interpolation/x/index.json": index(1)}, number=2, replace=True)
    report, rows = collect(path)
    assert report["commits"] == 2 and len(rows) == 1
    assert rows[0]["runtime_receipt"]["cpu_task_retirement"]["execution_device_id"] == 1


def test_json_budget_is_reported_without_partial_rows(tmp_path, monkeypatch):
    path = tmp_path / "large.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/x/index.json": index()})
    monkeypatch.setattr(collector, "MAX_INDEX", 1)
    report, rows = collect(path)
    assert not rows and "budget" in report["errors"][0]["error"]


def test_live_manifest_capture_deduplicates_and_preserves_exact_frames(tmp_path):
    directory = tmp_path / "run-123"
    directory.mkdir()
    (directory / "manifest.json").write_bytes(collector.encoded(receipt()))
    (directory / "raw_masks.npz").write_bytes(b"mask bytes are not inputs")
    destination = io.BytesIO()
    report = collector.collect_live(tmp_path, collector.RowWriter(destination), .025, interval=.001)
    rows = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert report["polls"] > 1 and report["rows"] == 1
    assert rows[0]["runtime_receipt"]["expected_frames"] == [3, 5, 8]
    assert b"mask bytes" not in destination.getvalue()


def test_tiled_receipts_preserve_original_and_tile_identity():
    run = dict(run_id="parent", tile_evidence=[dict(run_id="tile", runtime_receipt=receipt())])
    found = list(collector.run_receipts(run))
    assert found[0][0] == "/tile_evidence/0"
    assert found[0][1]["run_id"] == "tile"


def test_snapshot_reader_does_not_admit_later_appends(tmp_path):
    path = tmp_path / "prefix"
    path.write_bytes(b"first")
    with path.open("rb") as stream:
        snapshot = collector.SnapshotReader(stream, 5)
        with path.open("ab") as append:
            append.write(b"later")
        assert snapshot.read() == b"first"
        assert snapshot.read() == b""


def test_pax_long_path_and_cli_archive_live_zip(tmp_path):
    path = tmp_path / "sam-artifacts.tar"
    logical = "sam_interpolation/" + "long-family-" * 30 + "/evidence/index.json"
    with path.open("wb") as stream:
        transaction(stream, {logical: index()})
    live = tmp_path / "runtime"
    (live / "run-1").mkdir(parents=True)
    (live / "run-1" / "manifest.json").write_bytes(collector.encoded(receipt()))
    output = tmp_path / "capture.zip"
    assert collector.main(["--archive", str(path), "--live-root", str(live),
                           "--watch-seconds", "0", "--output", str(output)]) == 0
    with zipfile.ZipFile(output) as result:
        assert set(result.namelist()) == {"summary.json", "archive_receipts.jsonl", "live_receipts.jsonl"}
        summary = json.loads(result.read("summary.json"))
        assert summary["archive"]["rows"] == summary["live"]["rows"] == 1
        assert not summary["archive"]["errors"] and not summary["live"]["errors"]
        assert json.loads(result.read("archive_receipts.jsonl"))["source"]["member"] == logical
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        collector.main(["--archive", str(path), "--output", str(output)])
    assert output.read_bytes() == before


def test_output_budget_rejects_whole_row(monkeypatch):
    output = io.BytesIO()
    writer = collector.RowWriter(output)
    monkeypatch.setattr(collector, "MAX_OUTPUT", 1)
    with pytest.raises(ValueError, match="output budget"):
        writer.write({"runtime_receipt": receipt()})
    assert writer.rows == writer.bytes == 0 and not output.getvalue()


@pytest.mark.parametrize("tail,expected", [(b"partial-header", "incomplete_header"),
    (b"", "end_of_snapshot")])
def test_partial_header_after_commit_is_explicit(tmp_path, tail, expected):
    path = tmp_path / "partial.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/x/index.json": index()})
        stream.write(tail)
    report, rows = collect(path)
    assert report["tail_status"] == expected and len(rows) == 1


def test_partial_pax_tail_is_explicit(tmp_path):
    path = tmp_path / "partial-pax.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/x/index.json": index()})
        header = tarfile.TarInfo("PaxHeader")
        header.type, header.size = tarfile.XHDTYPE, 100
        stream.write(header.tobuf())
        stream.write(b"0123456789")
    report, rows = collect(path)
    assert report["tail_status"] == "incomplete_pax_payload" and len(rows) == 1


@pytest.mark.parametrize("kind", ["manifest-list", "dispatch-null", "commit-null"])
def test_malformed_json_shapes_become_capture_errors(tmp_path, kind):
    path = tmp_path / "malformed.tar"
    value = json.loads(index())
    if kind == "dispatch-null":
        value["runs"]["run-one"]["runtime_receipt"]["dispatch"] = None
    members = {"sam_interpolation/x/index.json": collector.encoded(value)}
    if kind == "manifest-list":
        members["sam_interpolation/x/manifest.json"] = b"[]"
    with path.open("wb") as stream:
        transaction(stream, members, commit=kind != "commit-null")
        if kind == "commit-null":
            add_member(stream, "COMMIT/" + f"{1:032x}" + ".json", b"null")
    report, rows = collect(path)
    assert not rows
    assert report["errors"] or report["tail_status"] == "stopped_before_unverified_tail"


def test_gnu_longname_rejected_before_tarfile_reads_payload(monkeypatch):
    header = tarfile.TarInfo("././@LongLink")
    header.type, header.size = tarfile.GNUTYPE_LONGNAME, 1024 ** 3
    class Guarded(io.BytesIO):
        def fileno(self):
            return 12345

        def read(self, size=-1):
            assert 0 <= size <= 512, "attempted to read the declared GNU extension payload"
            return super().read(size)
    monkeypatch.setattr(collector.os, "fstat", lambda fd: SimpleNamespace(st_size=2 * 1024 ** 3))
    entries, report = collector.scan_committed(Guarded(header.tobuf()))
    assert not entries and "Unsupported raw TAR header type" in report["tail_error"]


def test_pax_chain_and_cumulative_limits_are_checked_before_payload_read(monkeypatch):
    data = b"12 path=abc\n"
    stream = io.BytesIO()
    header = tarfile.TarInfo("PaxHeader")
    header.type, header.size = tarfile.XHDTYPE, len(data)
    stream.write(header.tobuf())
    stream.write(data + bytes(-len(data) % 512))
    stream.seek(0)
    monkeypatch.setattr(collector, "MAX_PAX_TOTAL", 512)
    with pytest.raises(ValueError, match="cumulative"):
        collector.header_available(stream, 0, len(stream.getvalue()))
    monkeypatch.setattr(collector, "MAX_PAX_TOTAL", 64 * collector.MIB)
    monkeypatch.setattr(collector, "MAX_PAX_CHAIN", 0)
    with pytest.raises(ValueError, match="chain"):
        collector.header_available(stream, 0, len(stream.getvalue()))


def test_pax_sparse_directives_are_rejected():
    with pytest.raises(ValueError, match="Unsupported PAX"):
        collector.validate_pax(b"22 GNU.sparse.major=1\n")


def test_deeply_nested_commit_keeps_authenticated_prefix(tmp_path):
    path = tmp_path / "nested-commit.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/old/index.json": index(0)})
        transaction(stream, {"sam_interpolation/new/index.json": index(1)}, number=2, commit=False)
        add_member(stream, "COMMIT/" + f"{2:032x}" + ".json",
                   b"[" * 10_000 + b"0" + b"]" * 10_000)
    report, rows = collect(path)
    assert len(rows) == 1 and report["commits"] == 1
    assert report["tail_status"] == "stopped_before_unverified_tail"
    assert report["tail_error"].startswith("RecursionError:")
    assert rows[0]["runtime_receipt"]["cpu_task_retirement"]["execution_device_id"] == 0


def shared_rows():
    shared = {"$ref": "this is original data", "values": list(range(400))}
    first = dict(scope={"left": shared, "right": [shared, None]}, runtime_receipt=receipt())
    second = copy.deepcopy(first)
    second["runtime_receipt"]["timings"]["worker_started_monotonic"] = 15.
    return [first, second, first]


def compact_bytes(rows):
    output = io.BytesIO()
    writer = collector.CompactRowWriter(output)
    for row in rows:
        writer.write(row)
    return output.getvalue(), writer.summary()


def test_compact_deterministic_exact_roundtrip_and_caller_mutation_isolated():
    original = shared_rows()
    first, summary = compact_bytes(original)
    second, _ = compact_bytes(original)
    assert first == second
    def reordered(value):
        if isinstance(value, dict):
            return {key: reordered(child) for key, child in reversed(list(value.items()))}
        return [reordered(child) for child in value] if isinstance(value, list) else value
    assert compact_bytes(reordered(original))[0] == first
    assert list(collector.iter_compact_rows(io.BytesIO(first))) == original
    assert summary["rows"] == 3 and summary["bytes"] == len(first)
    iterator = collector.iter_compact_rows(io.BytesIO(first))
    yielded = next(iterator)
    yielded["scope"]["left"]["values"][0] = -999
    assert yielded["scope"]["right"][0]["values"][0] == 0
    assert next(iterator) == original[1]
    assert next(iterator) == original[2]


@pytest.mark.parametrize("damage", ["missing", "checksum", "duplicate"])
def test_compact_reader_rejects_missing_corrupt_or_duplicate_objects(damage):
    payload, _ = compact_bytes(shared_rows())
    records = [json.loads(line) for line in payload.splitlines()]
    assert records[0]["kind"] == "object"
    if damage == "missing":
        del records[0]
    elif damage == "duplicate":
        records.insert(1, records[0])
    else:
        assert isinstance(records[0]["value"], list)
        records[0]["value"][0] = -1
    with pytest.raises(ValueError, match="[Mm]issing|checksum|duplicate"):
        list(collector.iter_compact_rows(io.BytesIO(b"".join(collector.encoded(r) + b"\n" for r in records))))


def test_compact_budget_rejection_leaves_no_objects_or_row(monkeypatch):
    output = io.BytesIO()
    writer = collector.CompactRowWriter(output)
    monkeypatch.setattr(collector, "MAX_OUTPUT", 800)
    with pytest.raises(collector.CaptureBudgetExceeded):
        writer.write(shared_rows()[0])
    assert not output.getvalue() and not writer.known
    assert writer.rows == writer.bytes == 0


def multi_run_index(count):
    value = json.loads(index())
    template = value["runs"]["run-one"]
    value["runs"] = {f"run-{i}": dict(template, run_id=f"run-{i}") for i in range(count)}
    return collector.encoded(value)


def test_mid_index_resume_preserves_all_appearances_once(tmp_path, monkeypatch):
    path = tmp_path / "paged.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/a/index.json": multi_run_index(3)})
        transaction(stream, {"sam_interpolation/b/index.json": multi_run_index(2)}, number=2)
    full_report, expected = collect(path)
    assert full_report["coverage_complete"] and len(expected) == 5
    monkeypatch.setattr(collector, "MAX_ROWS", 1)
    cursor, combined, reports = None, [], []
    for _ in range(10):
        output = io.BytesIO()
        report = collector.collect_archive(path, collector.CompactRowWriter(output), cursor=cursor)
        assert not report["errors"]
        reports.append(report)
        combined.extend(collector.iter_compact_rows(io.BytesIO(output.getvalue())))
        cursor = report["next_cursor"]
        if cursor is None:
            break
    assert combined == expected and len(reports) == 5
    assert reports[0]["next_cursor"]["row"] == 1
    assert reports[0]["indexes_total"] == 2 and not reports[0]["coverage_complete"]
    assert reports[-1]["coverage_complete"] and reports[-1]["indexes_remaining"] == 0


def test_resume_rejects_changed_inventory_without_emitting_rows(tmp_path, monkeypatch):
    path = tmp_path / "changed.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/a/index.json": multi_run_index(2)})
    monkeypatch.setattr(collector, "MAX_ROWS", 1)
    first = collector.collect_archive(path, collector.RowWriter(io.BytesIO()))
    with path.open("ab") as stream:
        transaction(stream, {"sam_interpolation/b/index.json": index()}, number=2)
    output = io.BytesIO()
    second = collector.collect_archive(path, collector.RowWriter(output), cursor=first["next_cursor"])
    assert not output.getvalue() and not second["coverage_complete"]
    assert "inventory does not match" in second["errors"][0]["error"]


def test_json_read_budget_has_resumable_explicit_coverage(tmp_path, monkeypatch):
    path = tmp_path / "json-budget.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/a/index.json": index()})
        transaction(stream, {"sam_interpolation/b/index.json": index()}, number=2)
    monkeypatch.setattr(collector, "MAX_JSON_TOTAL", len(index()) + 10)
    first = collector.collect_archive(path, collector.RowWriter(io.BytesIO()))
    assert first["indexes_completed"] == 1 and first["indexes_remaining"] == 1
    assert first["stop_reason"] == "json_read_budget" and not first["errors"]
    second = collector.collect_archive(path, collector.RowWriter(io.BytesIO()), cursor=first["next_cursor"])
    assert second["coverage_complete"] and second["indexes_completed"] == 1


def test_cli_compact_resume_from_previous_zip(tmp_path, monkeypatch):
    path = tmp_path / "sam-artifacts.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/a/index.json": multi_run_index(2)})
    monkeypatch.setattr(collector, "MAX_ROWS", 1)
    first, second = tmp_path / "part1.zip", tmp_path / "part2.zip"
    collector.main(["--archive", str(path), "--format", "compact", "--output", str(first)])
    collector.main(["--archive", str(path), "--format", "compact", "--resume-from", str(first),
                    "--output", str(second)])
    rows = []
    for archive in (first, second):
        with zipfile.ZipFile(archive) as saved:
            with saved.open("archive_receipts.compact.jsonl") as stream:
                rows.extend(collector.iter_compact_rows(stream))
            summary = json.loads(saved.read("summary.json"))
            assert summary["schema"] == collector.COMPACT_SCHEMA
    assert [row["source"]["run_key"] for row in rows] == ["run-0", "run-1"]
    assert summary["archive"]["coverage_complete"]


def test_compact_releases_ancestor_encoding_before_descending(monkeypatch):
    leaf = {"payload": "x" * 4096}
    branch = {"leaf": leaf}
    row = {"scope": {"branch": branch}, "runtime_receipt": receipt()}
    parents = {id(row["scope"]): id(row), id(branch): id(row["scope"]), id(leaf): id(branch)}
    watched = set(parents) | set(parents.values())
    live = {identity: 0 for identity in watched}
    original_encoder = collector.encoded

    class TrackedBytes(bytes):
        def __new__(cls, value, identity):
            instance = super().__new__(cls, value)
            instance.identity = identity
            live[identity] += 1
            return instance

        def __del__(self):
            live[self.identity] -= 1

    def tracked_encoder(value):
        identity = id(value)
        if identity in parents:
            assert live[parents[identity]] == 0, "full ancestor encoding survives recursive descent"
        data = original_encoder(value)
        return TrackedBytes(data, identity) if identity in watched else data

    monkeypatch.setattr(collector, "encoded", tracked_encoder)
    payload, _ = compact_bytes([row])
    assert not any(live.values())
    assert list(collector.iter_compact_rows(io.BytesIO(payload))) == [row]


def test_resume_rejects_byte_identical_replacement_transaction(tmp_path, monkeypatch):
    path = tmp_path / "recommitted.tar"
    logical = "sam_interpolation/a/index.json"
    payload = multi_run_index(2)
    with path.open("wb") as stream:
        transaction(stream, {logical: payload})
    monkeypatch.setattr(collector, "MAX_ROWS", 1)
    first = collector.collect_archive(path, collector.RowWriter(io.BytesIO()))
    with path.open("ab") as stream:
        transaction(stream, {logical: payload}, number=2, replace=True)
    output = io.BytesIO()
    second = collector.collect_archive(path, collector.RowWriter(output), cursor=first["next_cursor"])
    assert not output.getvalue() and not second["coverage_complete"]
    assert "inventory does not match" in second["errors"][0]["error"]


@pytest.mark.parametrize("writer_type", [collector.RowWriter, collector.CompactRowWriter])
def test_single_oversized_row_reports_nonresumable_error(tmp_path, monkeypatch, writer_type):
    path = tmp_path / "oversized-row.tar"
    with path.open("wb") as stream:
        transaction(stream, {"sam_interpolation/a/index.json": index()})
    monkeypatch.setattr(collector, "MAX_OUTPUT", 100)
    output = io.BytesIO()
    report = collector.collect_archive(path, writer_type(output))
    assert not output.getvalue() and report["rows"] == 0
    assert not report["coverage_complete"] and report["indexes_remaining"] == 1
    assert report["next_cursor"] is None
    assert report["stop_reason"] == "single_receipt_exceeds_capture_budget"
    assert report["errors"][0]["resumable"] is False


@pytest.mark.parametrize("writer_type", [collector.RowWriter, collector.CompactRowWriter])
def test_resumed_oversized_row_keeps_prior_prefix_and_stable_failure(tmp_path, monkeypatch, writer_type):
    path = tmp_path / "resumed-oversized.tar"
    logical = "sam_interpolation/a/index.json"
    value = json.loads(multi_run_index(2))
    value["runs"]["run-1"]["runtime_receipt"] = copy.deepcopy(receipt())
    value["runs"]["run-1"]["runtime_receipt"]["profile"] = {"large": "x" * 10_000}
    with path.open("wb") as stream:
        transaction(stream, {logical: collector.encoded(value)})
    _, expected = collect(path)
    measured = writer_type(io.BytesIO())
    measured.write(expected[0])
    monkeypatch.setattr(collector, "MAX_OUTPUT", measured.bytes + 20)
    first_output = io.BytesIO()
    first = collector.collect_archive(path, writer_type(first_output))
    assert first["rows"] == 1 and not first["errors"]
    assert first["next_cursor"]["row"] == 1
    retained = first_output.getvalue()
    second_output = io.BytesIO()
    second = collector.collect_archive(path, writer_type(second_output), cursor=first["next_cursor"])
    assert not second_output.getvalue() and first_output.getvalue() == retained
    assert second["rows_skipped_for_cursor"] == 1 and second["indexes_remaining"] == 1
    assert second["next_cursor"] is None and not second["coverage_complete"]
    assert second["errors"][0]["member"] == logical and second["errors"][0]["row"] == 1
    assert second["errors"][0]["resumable"] is False


@pytest.mark.parametrize("depth", [600, 800])
def test_compact_reader_matches_writer_valid_json_depth(depth):
    nested = {"leaf": [1, 2, 3]}
    for _ in range(depth):
        nested = [nested]
    row = {"scope": nested, "runtime_receipt": receipt()}
    original = collector.encoded(row)
    payload, _ = compact_bytes([row])
    restored = list(collector.iter_compact_rows(io.BytesIO(payload)))
    assert len(restored) == 1
    assert collector.encoded(restored[0]) == original
