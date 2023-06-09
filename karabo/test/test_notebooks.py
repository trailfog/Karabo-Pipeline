import os
import unittest

from karabo.util.plotting_util import Font

RUN_SLOW_TESTS = os.environ.get("RUN_SLOW_TESTS", "false").lower() == "true"
IS_GITHUB_RUNNER = os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true"


class TestJupyterNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        path_to_notebooks = os.path.join("karabo", "examples")
        os.chdir(path_to_notebooks)

    def _test_notebook(self, notebook):
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        print(Font.BOLD + Font.BLUE + "Testing notebook " + notebook + Font.END)

        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=-1)
            try:
                assert (
                    ep.preprocess(nb) is not None
                ), f"Got empty notebook for {notebook}"
            except Exception:
                assert False, f"Failed executing {notebook}"

    @unittest.skipIf(IS_GITHUB_RUNNER, "IS_GITHUB_RUNNER")
    def test_source_detection_notebook(self):
        self._test_notebook(notebook="source_detection.ipynb")

    @unittest.skipIf(IS_GITHUB_RUNNER, "IS_GITHUB_RUNNER")
    def test_source_detection_assesment_notebook(self):
        self._test_notebook(notebook="source_detection_assessment.ipynb")

    @unittest.skipIf(IS_GITHUB_RUNNER, "IS_GITHUB_RUNNER")
    @unittest.skipIf(not RUN_SLOW_TESTS, "SLOW_TESTS")
    def test_HIIM_Img_Recovery_notebook(self):
        self._test_notebook(notebook="HIIM_Img_Recovery.ipynb")
