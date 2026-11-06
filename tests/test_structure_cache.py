"""structure_log cache behaviour."""
import os
import tempfile
import unittest

from prepareData.helper import structure_log


class StructureCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.log = os.path.join(self.dir, 'x.log')
        self.csv = os.path.join(self.dir, 'x.log_structured.csv')
        with open(self.log, 'w') as f:
            f.write('- hello\n')

    def run_structure(self, use_cache):
        structure_log(self.dir, self.dir, 'x.log', '<Label> <Content>', use_cache=use_cache)

    def test_cache_skips_reparse_when_csv_is_fresh(self):
        self.run_structure(False)
        with open(self.csv, 'w') as f:
            f.write('sentinel')
        os.utime(self.csv, (os.path.getmtime(self.log) + 10,) * 2)
        self.run_structure(True)
        with open(self.csv) as f:
            self.assertEqual(f.read(), 'sentinel')

    def test_stale_cache_is_rebuilt(self):
        self.run_structure(False)
        os.utime(self.csv, (os.path.getmtime(self.log) - 10,) * 2)
        self.run_structure(True)
        with open(self.csv) as f:
            self.assertIn('hello', f.read())

    def test_cache_off_always_reparses(self):
        self.run_structure(False)
        with open(self.csv, 'w') as f:
            f.write('sentinel')
        os.utime(self.csv, (os.path.getmtime(self.log) + 10,) * 2)
        self.run_structure(False)
        with open(self.csv) as f:
            self.assertIn('hello', f.read())

    def test_missing_cache_builds_csv(self):
        self.run_structure(True)
        self.assertTrue(os.path.exists(self.csv))


if __name__ == '__main__':
    unittest.main()
