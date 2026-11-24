import re

import AlinGo-Prompt


def test_metadata():
    assert AlinGo-Prompt.__name__ == "AlinGo-Prompt"
    assert re.match(r"\d+\.\d+\.\d+", AlinGo-Prompt.__version__)
    assert AlinGo-Prompt.__author__ == "Omar Khattab"
    assert AlinGo-Prompt.__author_email__ == "okhattab@stanford.edu"
    assert AlinGo-Prompt.__url__ == "https://github.com/stanfordnlp/AlinGo-Prompt"
    assert AlinGo-Prompt.__description__ == "DSPy"
