# Code by zzjchen
# This code includes functions for evaluation
"""
A modified ROUGE-L package which enables setting 'beta' for calculation.
Main structure & code borrowed from
    https://github.com/pltrdy/rouge/tree/master/rouge
"""

from __future__ import absolute_import
from .rouge import FilesRouge, Rouge


# Vendored from https://github.com/zzjchen/Tailored-Visions (rougeL/), which is
# itself derived from https://github.com/pltrdy/rouge. Kept verbatim except for
# relative imports and dropping the `six` dependency, because the paper's
# ROUGE-L uses beta=5 (recall-weighted) and Google's `rouge_score` package
# hardcodes beta=1.
