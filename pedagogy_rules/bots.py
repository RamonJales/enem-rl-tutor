import random
import math
from typing import Dict
from abc import ABC, abstractmethod

class BaseStudentBot(ABC):
    """Abstract base class for simulating student behavior using IRT."""
    def __init__(self, true_proficiencies: Dict[str, float]):
        # The hidden environment state (logits from -3.0 to 3.0)
        self.true_proficiencies = true_proficiencies

    @abstractmethod
    def answer_question(self, concept_id: str, difficulty_level: str) -> int:
        """Returns 1 for correct answer, 0 for incorrect."""
        pass

    @staticmethod
    def _irt_probability(theta: float, difficulty_val: float) -> float:
        """Calculates success probability using Item Response Theory logistic function."""
        # Guessing parameter (20% chance for 5-option multiple choice)
        c = 0.20 
        return c + (1 - c) / (1 + math.exp(-(theta - difficulty_val)))

    def _get_difficulty_value(self, difficulty_level: str) -> float:
        diff_map = {"easy": -1.0, "medium": 0.0, "hard": 1.5}
        return diff_map.get(difficulty_level, 0.0)


class ConsistentStudentBot(BaseStudentBot):
    """Answers purely based on their true proficiency without wild guessing."""
    def answer_question(self, concept_id: str, difficulty_level: str) -> int:
        d_val = self._get_difficulty_value(difficulty_level)
        # Defaults to -2.0 (low proficiency) if concept is unknown
        theta = self.true_proficiencies.get(concept_id, -2.0) 
        
        prob = self._irt_probability(theta, d_val)
        return 1 if random.random() < prob else 0


class GuessingStudentBot(BaseStudentBot):
    """High variance bot. Guesses on hard questions, prone to careless mistakes."""
    def answer_question(self, concept_id: str, difficulty_level: str) -> int:
        if difficulty_level == "hard":
            # Pure random guess on hard questions (20% chance)
            return 1 if random.random() < 0.20 else 0
            
        d_val = self._get_difficulty_value(difficulty_level)
        theta = self.true_proficiencies.get(concept_id, -2.0)
        
        # 20% penalty representing carelessness or lack of focus
        prob = self._irt_probability(theta, d_val) * 0.8 
        return 1 if random.random() < prob else 0
