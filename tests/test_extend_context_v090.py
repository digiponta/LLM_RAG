import tempfile
import unittest
from pathlib import Path

from extend_context_v090 import validate_context


class ContextExtensionV090Tests(unittest.TestCase):
    def test_valid_64_to_256(self):
        validate_context(64, 256)

    def test_valid_256_to_512(self):
        validate_context(256, 512)

    def test_same_context_rejected(self):
        with self.assertRaises(ValueError):
            validate_context(64, 64)

    def test_smaller_context_rejected(self):
        with self.assertRaises(ValueError):
            validate_context(256, 64)

    def test_very_large_context_rejected(self):
        with self.assertRaises(ValueError):
            validate_context(64, 4096)

    def test_tiny_source_rejected(self):
        with self.assertRaises(ValueError):
            validate_context(1, 256)


if __name__ == "__main__":
    unittest.main()
