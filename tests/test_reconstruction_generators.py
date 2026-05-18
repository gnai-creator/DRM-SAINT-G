import unittest

from saint.reconstruction import (
    gaussian_matrix,
    low_rank_matrix,
    repeated_block_matrix,
    sparse_matrix,
)


class ReconstructionGeneratorTests(unittest.TestCase):
    def test_generators_are_deterministic(self):
        self.assertEqual(
            gaussian_matrix(3, 4, seed=7),
            gaussian_matrix(3, 4, seed=7),
        )
        self.assertEqual(
            low_rank_matrix(3, 4, rank=2, seed=7),
            low_rank_matrix(3, 4, rank=2, seed=7),
        )
        self.assertEqual(
            sparse_matrix(3, 4, density=0.25, seed=7),
            sparse_matrix(3, 4, density=0.25, seed=7),
        )
        self.assertEqual(
            repeated_block_matrix(4, 4, block_size=2, seed=7),
            repeated_block_matrix(4, 4, block_size=2, seed=7),
        )

    def test_repeated_block_matrix_has_requested_shape(self):
        matrix = repeated_block_matrix(5, 7, block_size=2, seed=1)

        self.assertEqual(len(matrix), 5)
        self.assertEqual(len(matrix[0]), 7)


if __name__ == "__main__":
    unittest.main()
