import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from prepareData.manifest import build_manifest, file_sha256, git_commit, write_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ManifestUnitTests(unittest.TestCase):
    def test_file_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'a.txt')
            with open(path, 'wb') as fout:
                fout.write(b'hello' * 500000)
            self.assertEqual(file_sha256(path), hashlib.sha256(b'hello' * 500000).hexdigest())

    def test_build_and_write_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, 'in.log'), os.path.join(tmp, 'out.csv')
            for path, text in ((src, 'abc'), (out, 'de')):
                with open(path, 'w') as fout:
                    fout.write(text)
            manifest = build_manifest('s.py', {'window_size': 100}, [src], [out], seed=7, stats={'n': 1})
            self.assertEqual(manifest['seed'], 7)
            self.assertEqual(manifest['params'], {'window_size': 100})
            self.assertEqual(manifest['inputs'][src]['bytes'], 3)
            self.assertEqual(manifest['inputs'][src]['sha256'], hashlib.sha256(b'abc').hexdigest())
            with open(write_manifest(tmp, manifest)) as fin:
                self.assertEqual(json.load(fin)['outputs'][out]['bytes'], 2)

    def test_hash_can_be_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'in.log')
            with open(src, 'w') as fout:
                fout.write('abc')
            self.assertNotIn('sha256', build_manifest('s.py', {}, [src], [], hash_files=False)['inputs'][src])

    def test_git_commit_is_none_outside_a_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(git_commit(cwd=tmp))


class ScriptRunner(unittest.TestCase):
    def run_module(self, module, data_dir, log_name, *extra):
        subprocess.run([sys.executable, '-m', module, '--data-dir', data_dir, '--log-name', log_name, *extra],
                       cwd=REPO_ROOT, check=True, capture_output=True)

    def read(self, path):
        with open(path) as fin:
            return fin.read()


class SessionWindowManifestTests(ScriptRunner):
    def make_hdfs(self, tmp):
        with open(os.path.join(tmp, 'HDFS.log'), 'w') as fout:
            for i in range(30):
                fout.write('081109 203615 148 INFO dfs.DataNode: block blk_%d done\n' % i)
        with open(os.path.join(tmp, 'anomaly_label.csv'), 'w') as fout:
            fout.write('BlockId,Label\n')
            for i in range(30):
                fout.write('blk_%d,%s\n' % (i, 'Anomaly' if i % 5 == 0 else 'Normal'))

    def test_seed_controls_split_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_hdfs(tmp)
            train_path = os.path.join(tmp, 'train.csv')
            self.run_module('prepareData.session_window', tmp, 'HDFS.log', '--seed', '1')
            first = self.read(train_path)
            manifest = json.loads(self.read(os.path.join(tmp, 'manifest.json')))
            self.assertEqual(manifest['seed'], 1)
            self.assertEqual(manifest['stats']['train_sessions'], 24)
            self.assertEqual(manifest['outputs'][train_path]['sha256'], hashlib.sha256(first.encode()).hexdigest())
            self.run_module('prepareData.session_window', tmp, 'HDFS.log', '--seed', '1')
            self.assertEqual(self.read(train_path), first)
            self.run_module('prepareData.session_window', tmp, 'HDFS.log', '--seed', '2')
            self.assertNotEqual(self.read(train_path), first)


if __name__ == '__main__':
    unittest.main()
