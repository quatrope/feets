import mock

from feets.extractors import (
    Extractor,
    register_extractor,
    sort_by_dependencies,
)

import numpy as np


@mock.patch("feets.extractors.register._extractors", {})
def test_sort_by_dependencies():
    @register_extractor
    class A(Extractor):
        features = ["test_a"]

        def extract(self, magnitude):
            return {"test_a": None}

    @register_extractor
    class B1(Extractor):
        features = ["test_b1"]

        def extract(self, magnitude, test_a):
            return {"test_b1": None}

    @register_extractor
    class B2(Extractor):
        features = ["test_b2"]

        def extract(self, magnitude, test_a):
            return {"test_b2": None}

    @register_extractor
    class C(Extractor):
        features = ["test_c"]

        def extract(self, magnitude, test_a, test_b1, test_b2):
            return {"test_c": None}

    a, b1, b2, c = A(), B1(), B2(), C()
    exts = [c, b1, a, b2]
    plan = sort_by_dependencies(exts)

    np.testing.assert_equal(len(plan), 4)
    for idx, ext in enumerate(plan):
        if idx == 0:
            np.testing.assert_equal(ext, a)
        elif idx in (1, 2):
            np.testing.assert_(np.isin(ext, (b1, b2)))
        elif idx == 3:
            np.testing.assert_equal(ext, c)
