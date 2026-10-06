import unittest

import pandas as pd

from prepareData.validate import LogValidationError, validate_logs


class ValidateLogsTests(unittest.TestCase):
    def good(self):
        return pd.DataFrame({'Label': [0, 1], 'Content': ['a', 'b']})

    def test_valid_dataframe_is_returned(self):
        df = self.good()
        self.assertIs(validate_logs(df), df)

    def test_missing_column(self):
        with self.assertRaisesRegex(LogValidationError, 'Content'):
            validate_logs(self.good().drop(columns='Content'))

    def test_empty_dataframe(self):
        with self.assertRaisesRegex(LogValidationError, 'empty'):
            validate_logs(self.good().iloc[0:0])

    def test_bad_label_values(self):
        df = self.good()
        df.loc[0, 'Label'] = 2
        with self.assertRaisesRegex(LogValidationError, 'Label'):
            validate_logs(df)

    def test_null_content(self):
        df = self.good()
        df.loc[0, 'Content'] = None
        with self.assertRaisesRegex(LogValidationError, 'blank'):
            validate_logs(df)

    def test_blank_content(self):
        df = self.good()
        df.loc[1, 'Content'] = '   '
        with self.assertRaises(LogValidationError):
            validate_logs(df)

    def test_allow_empty_content(self):
        df = self.good()
        df.loc[1, 'Content'] = ''
        validate_logs(df, allow_empty_content=True)


if __name__ == '__main__':
    unittest.main()
