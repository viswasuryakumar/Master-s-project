import hashlib
import json
import os
import tempfile
import unittest

from prepareData.manifest import build_manifest, file_sha256, git_commit, write_manifest


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


if __name__ == '__main__':
    unittest.main()
