import tempfile
import unittest
from pathlib import Path
from train_nagato import make_windows, load_nagato_lines

class NagatoTrainingTests(unittest.TestCase):
    def test_shifted_targets(self):
        self.assertEqual(make_windows([1,2,3,4,5], 3, 2),
                         [([1,2,3],[2,3,4]),([2,3,4],[3,4,5])])
    def test_short_text(self):
        self.assertEqual(make_windows([1,2], 3, 1), [])
    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            make_windows([1,2,3], 2, 0)
    def test_corpus_header(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"data-nagato.txt"
            p.write_text("@NagatFakeBot Tweet Messages, (C) 2016\n本文A\n\n本文B\n",encoding="utf-8")
            self.assertEqual(load_nagato_lines(p),["本文A","本文B"])

if __name__ == "__main__":
    unittest.main()
