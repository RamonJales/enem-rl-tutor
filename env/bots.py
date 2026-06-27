"""
bots.py
=======
Perfis simulados de alunos (Modelo do Estudante) para o Tutor Inteligente.

Define "bots" de aluno baseados em Teoria de Resposta ao Item (TRI): dado um
nível de proficiência (theta) oculto e a dificuldade da questão, devolvem
acerto/erro. Servem para AVALIAÇÃO e testes de robustez da política — rodar o
agente guloso contra diferentes perfis (consistente, chutador) e medir como ele
se comporta.

Posicionamento na arquitetura
------------------------------
Pertence ao domínio do Ambiente (`env/`). NÃO está no caminho crítico do
treino do DQN: o `StudentEnvironment` usado em `agent/train.py` mantém sua
própria simulação interna (intocada, para preservar o desempenho do modelo já
treinado). Estes bots são componentes auxiliares/plugáveis para experimentação
e validação, consumidos por testes e notebooks.
"""

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

    def success_probability(self, concept_id: str, difficulty_level: str) -> float:
        """
        Probabilidade de acerto P(y=1) do bot (mesma base TRI de answer_question).

        Existe para acoplar o bot a um ambiente que amostra o resultado por conta
        própria (ex.: a avaliação de robustez injeta esta probabilidade no lugar
        da dinâmica interna do StudentEnvironment, sem alterar o resto do MDP).
        Deve ser consistente com answer_question.
        """
        d_val = self._get_difficulty_value(difficulty_level)
        theta = self.true_proficiencies.get(concept_id, -2.0)
        return self._irt_probability(theta, d_val)

    def behavioral_probability(
        self, base_probability: float, difficulty_level: str
    ) -> float:
        """
        Aplica a DISTORÇÃO COMPORTAMENTAL do perfil sobre uma probabilidade-base.

        Diferente de `success_probability` (que deriva a habilidade do θ fixo do
        bot), este método recebe a probabilidade de acerto JÁ calculada pela
        dinâmica do ambiente (que inclui aprendizado e pré-requisitos) e devolve
        apenas o efeito do COMPORTAMENTO do aluno. Assim o perfil testa robustez
        a comportamento, sem quebrar os laços de aprendizado/currículo do MDP.

        Base (aluno consistente): sem distorção — devolve a própria base.
        """
        return base_probability


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

    def success_probability(self, concept_id: str, difficulty_level: str) -> float:
        """Espelha answer_question: chute puro nas difíceis; 20% de desatenção nas demais."""
        if difficulty_level == "hard":
            return 0.20
        return super().success_probability(concept_id, difficulty_level) * 0.8

    def behavioral_probability(
        self, base_probability: float, difficulty_level: str
    ) -> float:
        """
        Distorção do aluno chutador sobre a probabilidade-base do ambiente:
          - "hard": desiste e CHUTA (probabilidade fixa de 20%, 5 alternativas),
                    independentemente da habilidade real;
          - demais: 20% de penalidade por desatenção/falta de foco.
        """
        if difficulty_level == "hard":
            return 0.20
        return base_probability * 0.8
