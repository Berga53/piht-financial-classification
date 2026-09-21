from __future__ import annotations

import unittest

from piht_classification.data import _window_starts


class DataTests(unittest.TestCase):
    def test_one_year_input_depth_has_valid_windows(self):
        self.assertEqual(_window_starts(1, input_depth=1, target_depth=3), list(range(2009, 2016)))
        self.assertEqual(_window_starts(2, input_depth=1, target_depth=3), list(range(2016, 2022)))


if __name__ == "__main__":
    unittest.main()


class NationalCohortTests(unittest.TestCase):
    def test_default_includes_regions_and_zero_fills_missing_indicator_records(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import pandas as pd
        from piht_classification.data import prepare_bankit_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data'
            (data / 'Indicatori').mkdir(parents=True)
            pd.DataFrame({'BDAP': [1, 2, 3], 'Comune': ['COMUNE DI A', 'COMUNE DI B', 'COMUNE DI C'],
                          'Regione': ['SARDEGNA', 'LOMBARDIA', 'SICILIA']}).set_index('BDAP').to_csv(data / 'comuni.csv')
            pd.DataFrame(columns=['Anno', 'Comune']).to_csv(data / 'CriticitàComuni.csv', sep=';', index=False)
            pd.DataFrame({'BDAP': [1, 2, 3], **{str(y): [100, 200, 300] for y in range(2009, 2024)}}).set_index('BDAP').to_csv(data / 'popolazione.csv')
            for year in range(2016, 2024):
                pd.DataFrame({'BDAP': [2, 3], 'R1': [2., 3.]}).set_index('BDAP').to_csv(data / 'Indicatori' / f'Indicatori {year}.csv')
            result = prepare_bankit_dataset(root, period=2, input_depth=1, target_depth=3, features='indicatori')
            self.assertEqual(result.metadata['municipality_count'], 3)
            self.assertFalse(result.metadata['excluded_autonomous_regions'])
            self.assertEqual(result.X.shape, (18, 1, 1))
            np.testing.assert_array_equal(result.X[:3, 0, 0], [0, 2, 3])
            self.assertEqual(set(result.groups), {1, 2, 3})
