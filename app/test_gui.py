"""
Smoke test for the Streamlit interface (app/gui.py): the page must render without
raising, with and without an inspection result.
"""

import os
import unittest

from streamlit.testing.v1 import AppTest

GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui.py")
SAMPLE = os.path.join(os.path.dirname(GUI_PATH), "..", "data", "val", "tampered", "tampered_028.png")


class TestGui(unittest.TestCase):

    def test_renders_empty_state(self):
        at = AppTest.from_file(GUI_PATH, default_timeout=60).run()
        self.assertFalse(at.exception)
        markup = "".join(m.value for m in at.markdown)
        self.assertIn("Inspect the code.", markup)
        self.assertIn("Known limitations", markup)

    @unittest.skipUnless(os.path.exists(SAMPLE), "validation dataset not present")
    def test_renders_sample_result(self):
        at = AppTest.from_file(GUI_PATH, default_timeout=60)
        at.session_state["mode"] = "Samples"
        at.run()
        at.radio[0].set_value("Tampered, ECC violation").run()
        self.assertFalse(at.exception)
        markup = "".join(m.value for m in at.markdown)
        self.assertIn("Tampering detected", markup)
        self.assertIn("Reed-Solomon syndrome mismatch", markup)


if __name__ == "__main__":
    unittest.main()
