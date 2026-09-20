from pathlib import Path
import unittest

from neuro_symbolic.embeddings import load_word_embeddings
from neuro_symbolic.pipeline import generate_plan, normalize_object_name
from neuro_symbolic.planning import PDDLParser
from neuro_symbolic.perception import load_projection_model, predict_object


ROOT = Path(__file__).resolve().parents[1]


class NeuroSymbolicTests(unittest.TestCase):
    def test_embedding_checkpoint_shape(self):
        vocabulary, embeddings = load_word_embeddings(ROOT / "models" / "word_embeddings.pth")
        self.assertEqual(len(vocabulary), 523)
        self.assertEqual(embeddings.shape, (523, 128))

    def test_projection_checkpoint_loads_without_download(self):
        _, class_words, text_embeddings, metadata = load_projection_model(
            ROOT / "models" / "image_projection.pth"
        )
        self.assertEqual(len(class_words), 100)
        self.assertEqual(tuple(text_embeddings.shape), (100, 128))
        self.assertEqual(metadata["epoch"], 25)

    def test_image_model_forward_pass(self):
        import torch

        predictions = predict_object(
            torch.zeros(3, 32, 32),
            ROOT / "models" / "image_projection.pth",
            top_k=3,
            device="cpu",
        )
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(label, str) for label, _ in predictions))

    def test_label_normalization_matches_pddl_symbols(self):
        self.assertEqual(normalize_object_name("aquarium_fish"), "aquarium-fish")
        self.assertEqual(normalize_object_name("lawn mower"), "lawn-mower")

    def test_domain_has_expected_actions(self):
        actions = PDDLParser.parse_domain(ROOT / "pddl" / "domain.pddl")
        self.assertTrue({"pick-up", "put-down", "slice-object", "take-photo"}.issubset(actions))

    def test_text_input_can_generate_slice_plan(self):
        plan = generate_plan(
            "apple",
            initial_state=["(at ?x lab)", "(whole ?x)"],
            goal_state=["(cut-into-pieces ?x)"],
            domain_file=ROOT / "pddl" / "domain.pddl",
            base_problem=ROOT / "pddl" / "base_problem.pddl",
            projection_checkpoint=ROOT / "models" / "image_projection.pth",
        )
        self.assertIsNotNone(plan)
        self.assertTrue(any("slice-object apple" in step for step in plan))


if __name__ == "__main__":
    unittest.main()
