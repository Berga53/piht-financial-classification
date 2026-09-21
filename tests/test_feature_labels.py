from __future__ import annotations

import unittest

from piht_classification.feature_labels import FEATURE_LABELS, display_feature_name


class FeatureLabelTests(unittest.TestCase):
    def test_complete_catalogue_codes(self):
        self.assertEqual(len(FEATURE_LABELS), 93)
        self.assertEqual(
            display_feature_name("Impegno totale SPESE IN CONTO CAPITALE"),
            r"$S2_{P1}$ COM",
        )
        self.assertEqual(
            display_feature_name("Riscossioni in CR TRASFERIMENTI CORRENTI"),
            r"$E8_{P2}$ COL PY",
        )
        self.assertEqual(display_feature_name("A2"), r"$A2$")
        self.assertEqual(display_feature_name("R9"), r"$R9$")

    def test_lag_suffix_is_preserved_for_multi_year_inputs(self):
        self.assertEqual(display_feature_name("R3@t-0"), r"$R3$")
        self.assertEqual(display_feature_name("R3@t-2"), r"$R3$ (t-2)")


if __name__ == "__main__":
    unittest.main()
