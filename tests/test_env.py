import unittest
from env.knowledge_graph import KnowledgeGraph
from env.bots import ConsistentStudentBot, GuessingStudentBot

class TestKnowledgeGraph(unittest.TestCase):
    def setUp(self):
        """Initializes a fresh graph before each test."""
        self.kg = KnowledgeGraph(decay_rate=0.5)
        self.kg.add_node("arithmetic", "Arithmetic")
        self.kg.add_node("equations", "Equations")
        self.kg.add_node("functions", "Functions")
        
        # arithmetic -> equations -> functions
        self.kg.add_edge("arithmetic", "equations", weight=0.8)
        self.kg.add_edge("equations", "functions", weight=0.9)

    def test_add_edge_missing_node_error(self):
        """Error Case: Should raise ValueError if nodes don't exist."""
        with self.assertRaises(ValueError):
            self.kg.add_edge("geometry", "arithmetic", 0.5)

    def test_propagate_update_missing_target_error(self):
        """Error Case: Should raise KeyError if target node is missing."""
        state = {"arithmetic": 0.5}
        with self.assertRaises(KeyError):
            self.kg.propagate_update(state, "calculus", 0.1)

    def test_backward_inference_success(self):
        """Success Case: Correct answer propagates backward to prerequisites."""
        initial_state = {
            "arithmetic": 0.2,
            "equations": 0.3,
            "functions": 0.4
        }
        
        # Student got a 'functions' question right (+0.2)
        delta_p = 0.2
        new_state = self.kg.propagate_update(initial_state, "functions", delta_p)
        
        # target node should update exactly by delta_p
        self.assertAlmostEqual(new_state["functions"], 0.6)
        
        # parent node (equations) should receive delta * weight * decay
        # 0.2 * 0.9 * 0.5 = 0.09. New value: 0.3 + 0.09 = 0.39
        self.assertAlmostEqual(new_state["equations"], 0.39)
        
        # grandparent node (arithmetic) should receive recursive decay
        # 0.09 * 0.8 * 0.5 = 0.036. New value: 0.2 + 0.036 = 0.236
        self.assertAlmostEqual(new_state["arithmetic"], 0.236)

    def test_forward_penalty_success(self):
        """Success Case: Incorrect answer cascades penalty to dependent nodes."""
        initial_state = {
            "arithmetic": 0.8,
            "equations": 0.7,
            "functions": 0.6
        }
        
        # Student missed a basic 'arithmetic' question (-0.3)
        delta_p = -0.3
        new_state = self.kg.propagate_update(initial_state, "arithmetic", delta_p)
        
        self.assertAlmostEqual(new_state["arithmetic"], 0.5)
        # Equations penalty: -0.3 * 0.8 * 0.5 = -0.12. New value: 0.7 - 0.12 = 0.58
        self.assertAlmostEqual(new_state["equations"], 0.58)

    def test_probability_clamping(self):
        """Edge Case: Probabilities must never exceed 1.0 or drop below 0.0."""
        initial_state = {"arithmetic": 0.9}
        
        # Massive positive delta
        high_state = self.kg.propagate_update(initial_state, "arithmetic", 5.0)
        self.assertEqual(high_state["arithmetic"], 1.0)
        
        # Massive negative delta
        low_state = self.kg.propagate_update(initial_state, "arithmetic", -5.0)
        self.assertEqual(low_state["arithmetic"], 0.0)


class TestStudentBots(unittest.TestCase):
    def test_consistent_bot_behavior(self):
        """Test if the consistent bot answers logically based on its theta."""
        proficiencies = {"arithmetic": 2.0} # Very high proficiency
        bot = ConsistentStudentBot(proficiencies)
        
        # Due to randomness, we can't assert 1 or 0 perfectly without mocking random.
        # But we can test the internal IRT probability calculation.
        prob = bot._irt_probability(theta=2.0, difficulty_val=-1.0) # Easy question
        self.assertTrue(prob > 0.9) # Should be extremely likely to get it right
        
    def test_missing_proficiency_fallback(self):
        """Test if bots handle missing concept keys gracefully."""
        bot = ConsistentStudentBot({"arithmetic": 1.0})
        # Try to answer a concept not in the dict
        # Should fallback to -2.0 (low probability)
        # Ensure it doesn't throw a KeyError
        try:
            bot.answer_question("geometry", "easy")
            success = True
        except KeyError:
            success = False
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
