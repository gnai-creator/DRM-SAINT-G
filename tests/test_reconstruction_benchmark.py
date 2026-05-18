import unittest

from saint.reconstruction import (
    BenchmarkCase,
    block_codebook_reconstruction,
    original_reconstruction,
    repeated_block_matrix,
    run_reconstruction_benchmark,
)


class ReconstructionBenchmarkTests(unittest.TestCase):
    def test_runner_returns_result_per_case_and_baseline(self):
        cases = [
            BenchmarkCase("repeated", repeated_block_matrix(4, 4, seed=1)),
            BenchmarkCase("small", [[1, 2], [3, 4]]),
        ]
        baselines = [
            original_reconstruction,
            lambda matrix: block_codebook_reconstruction(
                matrix,
                block_size=2,
                signature_mode="exact",
            ),
        ]

        results = run_reconstruction_benchmark(cases, baselines)

        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].case_name, "repeated")
        self.assertGreaterEqual(results[0].compression_ratio, 1.0)
        self.assertIn("reuse_ratio", results[0].metadata)


if __name__ == "__main__":
    unittest.main()
