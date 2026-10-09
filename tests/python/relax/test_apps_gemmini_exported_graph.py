# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed fixture admission before costly exported graph simulation."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from verify_exported_graph import load_fixture, validate_manifest


class FixtureAdmission(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gemmini-exported-fixture-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.npz"
        self.manifest = {"inputs": [{"shape": [2, 3], "dtype": "int8", "bytes": 6}],
                         "outputs": [{"shape": [2, 4], "dtype": "int32", "bytes": 32}, {"shape": [2, 4], "dtype": "float32", "bytes": 32}]}
        self.arrays = {"input_0": np.arange(12, dtype=np.int8).reshape(2, 2, 3),
                       "output_0": np.arange(16, dtype=np.int32).reshape(2, 2, 4),
                       "output_1": np.arange(16, dtype=np.float32).reshape(2, 2, 4)}

    def test_complete_mixed_output_stream_preserves_exact_bits(self):
        self.arrays["output_1"][0, 0, 0] = -0.0
        np.savez(self.path, **self.arrays)
        actual, samples = load_fixture(self.path, self.manifest)
        self.assertEqual(samples, 2)
        for name, array in self.arrays.items():
            self.assertEqual(actual[name].dtype, array.dtype)
            self.assertEqual(actual[name].tobytes(), array.tobytes())

    def test_implicit_dtype_conversion_and_truncated_stream_refused(self):
        for name, replacement in (("input_0", self.arrays["input_0"].astype(np.float32)),
                                  ("output_0", self.arrays["output_0"][:1]),
                                  ("output_1", self.arrays["output_1"][0]),
                                  ("output_0", self.arrays["output_0"][:, :, :3])):
            with self.subTest(name=name, shape=replacement.shape, dtype=replacement.dtype):
                arrays = dict(self.arrays, **{name: replacement})
                np.savez(self.path, **arrays)
                with self.assertRaises(ValueError):
                    load_fixture(self.path, self.manifest)

    def test_understated_overstated_and_invalid_tensor_bytes_refused(self):
        import copy
        np.savez(self.path, **self.arrays)
        malformed = [{"bytes": 31}, {"bytes": 33}, {"bytes": True},
                     {"shape": [2, 0]}, {"shape": [2, True]}, {"shape": [2, 4.0]},
                     {"dtype": "float64"}, {"shape": [1 << 62, 8], "bytes": 1 << 67}]
        for mutation in malformed:
            manifest = copy.deepcopy(self.manifest)
            manifest["outputs"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_fixture(self.path, manifest)

    def test_internal_tensor_storage_envelope_is_checked(self):
        import copy
        manifest = dict(self.manifest, constants=[{"dtype": "int8", "shape": [3], "bytes": 3, "offset": 0}],
                        intermediates=[{"dtype": "int32", "shape": [8], "bytes": 32, "offset": 0, "origin": "workspace"}],
                        constants_bytes=64, workspace_bytes=64, explicit_tensor_bytes=198)
        validate_manifest(manifest)
        for group, mutation in (("constants", {"bytes": 2}), ("constants", {"offset": 64}),
                                ("intermediates", {"bytes": 16}), ("intermediates", {"offset": 1})):
            altered = copy.deepcopy(manifest)
            altered[group][0].update(mutation)
            with self.subTest(group=group, mutation=mutation), self.assertRaises(ValueError):
                validate_manifest(altered)
        with self.assertRaises(ValueError):
            validate_manifest(dict(manifest, explicit_tensor_bytes=197))

    def test_extra_missing_nonfinite_and_empty_streams_refused(self):
        malformed = [dict(self.arrays, extra=np.zeros(1)),
                     {name: value for name, value in self.arrays.items() if name != "output_1"},
                     dict(self.arrays, output_1=np.full((2, 2, 4), np.nan, np.float32)),
                     {name: value[:0] for name, value in self.arrays.items()}]
        for arrays in malformed:
            np.savez(self.path, **arrays)
            with self.assertRaises(ValueError):
                load_fixture(self.path, self.manifest)


if __name__ == "__main__":
    unittest.main()
