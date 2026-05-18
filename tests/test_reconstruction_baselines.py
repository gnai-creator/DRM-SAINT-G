import unittest

from saint.blocks import reconstruction_error
from saint.reconstruction import (
    block_codebook_reconstruction,
    hierarchical_codebook_reconstruction,
    low_rank_matrix,
    low_rank_reconstruction,
    multi_scale_codebook_reconstruction,
    original_reconstruction,
    repeated_block_matrix,
    uniform_quantization_reconstruction,
)


class ReconstructionBaselineTests(unittest.TestCase):
    def test_original_reconstruction_is_lossless(self):
        matrix = [[1, 2], [3, 4]]

        result = original_reconstruction(matrix)
        metrics = reconstruction_error(matrix, result.reconstructed)

        self.assertEqual(metrics.l1_error, 0.0)
        self.assertEqual(result.parameter_count, 4)

    def test_uniform_quantization_reconstruction_rounds_values(self):
        matrix = [[0.04, 0.16], [0.24, 0.31]]

        result = uniform_quantization_reconstruction(matrix, step=0.1)

        self.assertEqual(result.reconstructed, [[0.0, 0.2], [0.2, 0.30000000000000004]])
        self.assertGreater(result.parameter_count, 0)

    def test_block_codebook_reconstruction_reuses_repeated_blocks(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=3)

        result = block_codebook_reconstruction(
            matrix,
            block_size=2,
            signature_mode="exact",
        )
        metrics = reconstruction_error(matrix, result.reconstructed)

        self.assertEqual(metrics.l1_error, 0.0)
        self.assertLess(result.metadata["prototype_count"], result.metadata["block_count"])

    def test_multi_scale_selects_a_candidate(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=3)

        result = multi_scale_codebook_reconstruction(
            matrix,
            block_sizes=(4, 2),
            signature_mode="exact",
        )

        self.assertEqual(result.name, "multi_scale_codebook")
        self.assertIn("selected", result.metadata)
        self.assertEqual(len(result.metadata["candidates"]), 2)

    def test_hierarchical_codebook_reconstructs_repeated_blocks(self):
        matrix = repeated_block_matrix(8, 8, block_size=2, prototypes=2, seed=3)

        result = hierarchical_codebook_reconstruction(
            matrix,
            block_sizes=(4, 2),
            signature_mode="exact",
        )
        metrics = reconstruction_error(matrix, result.reconstructed)

        self.assertEqual(result.name, "hierarchical_codebook")
        self.assertEqual(metrics.l1_error, 0.0)
        self.assertGreater(result.metadata["prototype_count"], 0)
        self.assertGreater(result.metadata["leaf_count"], 0)

    def test_low_rank_baseline_reduces_error_on_low_rank_matrix(self):
        matrix = low_rank_matrix(4, 4, rank=1, seed=2)

        result = low_rank_reconstruction(matrix, rank=1, iterations=10)
        metrics = reconstruction_error(matrix, result.reconstructed)

        self.assertLess(metrics.relative_l1_error, 0.05)
        self.assertEqual(result.parameter_count, 9)


if __name__ == "__main__":
    unittest.main()
