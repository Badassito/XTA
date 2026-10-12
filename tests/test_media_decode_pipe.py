"""Real subprocess EOF, diagnostics, cancellation and Windows pipe ownership."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from unittest import mock

from XTA import media


class DecodePipeTests(unittest.TestCase):
    def _command(self, source):
        return [sys.executable, '-B', '-c', source]

    def _finish(self, proc):
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    def test_large_output_reaches_exact_eof_and_communicate_closes_both_pipes(self):
        command = self._command(
            "import sys; sys.stdout.buffer.write(bytes(range(256)) * 8192); "
            "sys.stdout.buffer.flush(); sys.stderr.buffer.write(b'decoder-diagnostic')"
        )
        proc = media._spawn_decode_process(command)
        reader, diagnostics = proc.stdout, proc.stderr
        try:
            output, error = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 0)
            self.assertEqual(output, bytes(range(256)) * 8192)
            self.assertEqual(error, b'decoder-diagnostic')
            self.assertTrue(reader.closed)
            self.assertTrue(diagnostics.closed)
        finally:
            self._finish(proc)

    def test_nonzero_exit_retains_stdout_and_stderr(self):
        proc = media._spawn_decode_process(self._command(
            "import sys; sys.stdout.buffer.write(b'partial'); "
            "sys.stderr.buffer.write(b'decode-failed'); sys.exit(7)"
        ))
        try:
            output, error = proc.communicate(timeout=10)
            self.assertEqual((output, error, proc.returncode), (b'partial', b'decode-failed', 7))
        finally:
            self._finish(proc)

    def test_kill_and_detach_stdout_matches_decoder_abort_cleanup(self):
        proc = media._spawn_decode_process(self._command(
            "import sys, time; sys.stdout.buffer.write(b'ready'); "
            "sys.stdout.buffer.flush(); time.sleep(30)"
        ))
        reader, diagnostics = proc.stdout, proc.stderr
        try:
            self.assertEqual(reader.read(5), b'ready')
            proc.kill()
            reader.close()
            proc.stdout = None
            output, error = proc.communicate(timeout=10)
            self.assertIsNone(output)
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue(reader.closed)
            self.assertTrue(diagnostics.closed)
        finally:
            self._finish(proc)

    def test_non_windows_uses_the_original_subprocess_pipe_arguments(self):
        with mock.patch.object(media.os, 'name', 'posix'), mock.patch.object(media.subprocess, 'Popen') as spawn:
            result = media._spawn_decode_process(['decoder'])
            spawn.assert_called_once_with(['decoder'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertIs(result, spawn.return_value)

    @unittest.skipUnless(os.name == 'nt', 'Windows handle ownership')
    def test_windows_closes_parent_writer_immediately_so_empty_output_reaches_eof(self):
        original_popen = subprocess.Popen
        with mock.patch.object(media.subprocess, 'Popen', wraps=original_popen) as spawn:
            proc = media._spawn_decode_process(self._command('pass'))
        try:
            self.assertTrue(spawn.call_args.kwargs['stdout'].closed)
            output, error = proc.communicate(timeout=10)
            self.assertEqual((output, error, proc.returncode), (b'', b'', 0))
        finally:
            self._finish(proc)

    @unittest.skipUnless(os.name == 'nt', 'Windows handle ownership')
    def test_windows_failed_conversions_and_spawn_close_every_pipe_handle(self):
        import _winapi
        import msvcrt
        import psutil

        original_pipe = _winapi.CreatePipe
        for target, attribute, fail_at in (
            (msvcrt, 'open_osfhandle', 1), (msvcrt, 'open_osfhandle', 2),
            (media.os, 'fdopen', 1), (media.os, 'fdopen', 2),
            (media.subprocess, 'Popen', 1),
        ):
            with self.subTest(attribute=attribute, fail_at=fail_at):
                handles = []
                calls = 0
                original = getattr(target, attribute)
                before = psutil.Process().num_handles()

                def create_pipe(*args):
                    pair = original_pipe(*args)
                    handles.extend(pair)
                    return pair

                def injected(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise OSError('injected pipe setup failure')
                    return original(*args, **kwargs)

                with mock.patch.object(_winapi, 'CreatePipe', side_effect=create_pipe), \
                        mock.patch.object(target, attribute, side_effect=injected):
                    with self.assertRaisesRegex(OSError, 'injected pipe setup failure'):
                        media._spawn_decode_process(self._command('pass'))
                self.assertEqual(len(handles), 2)
                self.assertEqual(psutil.Process().num_handles(), before, 'pipe handle leaked')

    @unittest.skipUnless(os.name == 'nt', 'Windows handle ownership')
    def test_windows_create_pipe_failure_never_launches_a_child(self):
        import _winapi

        with mock.patch.object(_winapi, 'CreatePipe', side_effect=OSError('pipe unavailable')), \
                mock.patch.object(media.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(OSError, 'pipe unavailable'):
                media._spawn_decode_process(self._command('pass'))
            spawn.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows handle ownership')
    def test_writer_close_failure_kills_child_and_closes_its_diagnostics(self):
        original_fdopen = os.fdopen
        original_popen = subprocess.Popen
        writers = []
        children = []

        class Writer:
            def __init__(self, stream):
                self.stream = stream
                self.failed = False

            def fileno(self):
                return self.stream.fileno()

            def close(self):
                if not self.failed:
                    self.failed = True
                    raise OSError('injected writer close failure')
                self.stream.close()

        def fdopen(fd, mode, **kwargs):
            stream = original_fdopen(fd, mode, **kwargs)
            if mode == 'wb':
                stream = Writer(stream)
                writers.append(stream)
            return stream

        def spawn(*args, **kwargs):
            proc = original_popen(*args, **kwargs)
            children.append(proc)
            return proc

        with mock.patch.object(media.os, 'fdopen', side_effect=fdopen), \
                mock.patch.object(media.subprocess, 'Popen', side_effect=spawn):
            with self.assertRaisesRegex(OSError, 'injected writer close failure'):
                media._spawn_decode_process(self._command('import time; time.sleep(30)'))
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stderr.closed)
        self.assertTrue(writers[0].stream.closed)


if __name__ == '__main__':
    unittest.main()
