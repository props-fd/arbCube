"""Check move boundaries, overlapping counts, and variable-length BPE tokens."""

from collections import Counter
import unittest

from arbcube.bpe import (
    _count_occurrences,
    bpe_move_block_lengths,
    bpe_solution_length_curve,
    find_sequence_occurrences,
    learn_move_bpe,
)


class MoveBpeTests(unittest.TestCase):
    def test_inspector_counts_overlaps_and_records_positions(self):
        matches = find_sequence_occurrences(["R R R", "R R", "R"], "R R")
        self.assertEqual(matches.solution_index.tolist(), [0, 0, 1])
        self.assertEqual(matches.start_move.tolist(), [0, 1, 0])
        self.assertEqual(matches.start_fraction.tolist(), [0.0, 0.5, 0.0])

    def test_inspector_matches_whole_moves_and_handles_whitespace(self):
        matches = find_sequence_occurrences(["R2 U", "R' U", " R   U ", "R", "U"], "R U")
        self.assertEqual(matches.solution_index.tolist(), [2])
        self.assertEqual(matches.start_move.tolist(), [0])
        self.assertTrue(find_sequence_occurrences(["R", "U"], "R U").empty)

    def test_inspector_rejects_empty_sequence(self):
        with self.assertRaises(ValueError):
            find_sequence_occurrences(["R U"], " ")

    def test_counts_overlapping_and_nested_patterns(self):
        self.assertEqual(
            _count_occurrences(["AAAA", "AA", "AB"], ["AA", "AAA", "AB"]),
            Counter({"AA": 4, "AAA": 2, "AB": 1}),
        )

    def test_matches_do_not_cross_solution_boundaries(self):
        self.assertEqual(_count_occurrences(["A", "B"], ["AB"]), Counter())
        self.assertTrue(learn_move_bpe(["R", "U"] * 10).empty)

    def test_learns_longer_tokens_and_distinguishes_final_uses(self):
        phrases = learn_move_bpe(["R U F B"] * 20, num_merges=10)
        longest = phrases.loc[phrases.Sequence == "R U F B"].iloc[0]
        self.assertEqual(int(longest.Moves), 4)
        self.assertEqual(int(longest.Occurrences), 20)
        self.assertEqual(int(longest["BPE uses"]), 20)
        self.assertTrue(phrases["BPE uses"].lt(phrases.Occurrences).any())

    def test_whole_moves_keep_their_modifiers(self):
        phrases = learn_move_bpe(["R R' R2"] * 10, num_merges=10)
        self.assertIn("R R' R2", phrases.Sequence.tolist())
        for row in phrases.itertuples():
            self.assertEqual(len(row.Sequence.split()), row.Moves)
            self.assertLessEqual(set(row.Sequence.split()), {"R", "R'", "R2"})

    def test_overlaps_are_counted_beyond_bpe_segmentation(self):
        phrases = learn_move_bpe(["R R R"] * 10, num_merges=1)
        pair = phrases.loc[phrases.Sequence == "R R"].iloc[0]
        self.assertEqual(int(pair.Occurrences), 20)
        self.assertEqual(int(pair["BPE uses"]), 10)

    def test_empty_input_and_zero_budget(self):
        for solutions, budget in (([], 10), ([""], 10), (["R U"], 0)):
            phrases = learn_move_bpe(solutions, num_merges=budget)
            self.assertTrue(phrases.empty)
            self.assertEqual(
                phrases.columns.tolist(), ["Sequence", "Moves", "Occurrences", "BPE uses"]
            )

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            learn_move_bpe(["R U"], num_merges=-1)


class MoveBlockLengthTests(unittest.TestCase):
    def test_includes_single_moves_and_counts_each_learned_block_once(self):
        lengths = bpe_move_block_lengths(["R U F B"] * 20, num_merges=10)
        self.assertEqual(lengths.tolist(), [1, 1, 1, 1, 2, 2, 4])

    def test_empty_input_zero_budget_and_early_stopping(self):
        self.assertTrue(bpe_move_block_lengths([]).empty)
        self.assertTrue(bpe_move_block_lengths([""]).empty)
        self.assertEqual(bpe_move_block_lengths(["R U"], num_merges=0).tolist(), [1, 1])
        self.assertEqual(bpe_move_block_lengths(["R", "U"], num_merges=10).tolist(), [1, 1])

    def test_negative_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            bpe_move_block_lengths(["R U"], num_merges=-1)


class BpeLengthCurveTests(unittest.TestCase):
    def test_baseline_and_quantiles_use_each_solutions_encoded_length(self):
        curve = bpe_solution_length_curve(["R R R", "R R", "R"], num_merges=1)
        self.assertEqual(curve["Vocabulary size"].tolist(), [1, 2])
        self.assertEqual(curve.Merges.tolist(), [0, 1])
        self.assertEqual(curve.Mean.iloc[0], 2)
        self.assertAlmostEqual(curve.Mean.iloc[1], 4 / 3)
        self.assertEqual(curve.Median.tolist(), [2, 1])
        for name, expected in {"10%": 1, "25%": 1, "75%": 1.5, "90%": 1.8}.items():
            self.assertAlmostEqual(curve[name].iloc[1], expected)

    def test_merge_prefixes_include_baseline_and_noncheckpoint_endpoint(self):
        curve = bpe_solution_length_curve(["R U F B"] * 20, num_merges=10, merge_step=2)
        self.assertEqual(curve.Merges.tolist(), [0, 2, 3])
        self.assertEqual(curve["Vocabulary size"].tolist(), [4, 6, 7])
        self.assertEqual(curve.Mean.iloc[0], 4)
        self.assertEqual(curve.Mean.iloc[-1], 1)
        self.assertTrue(curve.Mean.is_monotonic_decreasing)

    def test_zero_budget_and_no_frequent_pairs_keep_baseline(self):
        for budget in (0, 10):
            curve = bpe_solution_length_curve(iter(["R", "U", "R2", "R'"]), num_merges=budget)
            self.assertEqual(len(curve), 1)
            self.assertEqual(curve["Vocabulary size"].iloc[0], 4)
            self.assertEqual(curve.Mean.iloc[0], 1)

    def test_empty_corpus_and_empty_solutions(self):
        self.assertTrue(bpe_solution_length_curve([]).empty)
        curve = bpe_solution_length_curve(["", "  "])
        self.assertEqual(len(curve), 1)
        self.assertTrue(curve.eq(0).all().all())

    def test_invalid_budgets_and_steps(self):
        for kwargs in ({"num_merges": -1}, {"merge_step": 0}, {"merge_step": -1}):
            with self.assertRaises(ValueError):
                bpe_solution_length_curve(["R U"], **kwargs)


if __name__ == "__main__":
    unittest.main()
