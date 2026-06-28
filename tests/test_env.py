"""
test_env.py
===========
Testes do domínio do Ambiente (`env/`) do Tutor Inteligente.

Cobre três frentes, todas SEM efeitos colaterais sobre o banco de produção
(`data/enem_tutor.db`) nem sobre a dinâmica de treino do DQN:

1. `KnowledgeGraph` — regras de propagação no DAG (Modelo de Domínio).
2. Bots de aluno (TRI) — perfis simulados para avaliação.
3. `StudentEnvironment` + `DQNAgent` — teste de INTEGRAÇÃO do fluxo real
   (reset/step/recompensa e um passo de otimização do DQN) rodado contra um
   banco SQLite TEMPORÁRIO criado e descartado pelo próprio teste.

Execução (a partir da raiz do projeto):
    python -m unittest discover -s tests -v
"""

import os
import tempfile
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

    def test_guessing_bot_returns_binary(self):
        """Bots devem sempre devolver 0 ou 1 (contrato de resposta)."""
        bot = GuessingStudentBot({"arithmetic": 0.0})
        for nivel in ("easy", "medium", "hard"):
            resposta = bot.answer_question("arithmetic", nivel)
            self.assertIn(resposta, (0, 1))


class TestStudentEnvironmentFlow(unittest.TestCase):
    """
    Teste de INTEGRAÇÃO do fluxo real Ambiente <-> Agente.

    Constrói um banco SQLite TEMPORÁRIO (mini-DAG de 3 conceitos) usando o mesmo
    schema de `data/database_setup`, instancia o `StudentEnvironment` e o
    `DQNAgent` reais e roda alguns passos — validando o contrato (formato do
    estado, recompensa, `done`, `info`) e um passo de otimização do DQN.

    Importante: NÃO usa nem altera o banco de produção e NÃO reconfigura a
    dinâmica do simulador; apenas exercita o pipeline tal como em `agent.train`.
    """

    @classmethod
    def setUpClass(cls):
        # Banco temporário isolado (descartado no tearDown). Evita qualquer
        # efeito colateral sobre data/enem_tutor.db e sobre o treino real.
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls._tmpdir.name, "test_tutor.db")
        cls.db_url = f"sqlite:///{cls.db_path}"
        cls.estudante_id = 1
        cls._semear_banco(cls.db_url, cls.estudante_id)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    @staticmethod
    def _semear_banco(db_url: str, estudante_id: int) -> None:
        """Cria o schema e um mini-DAG (3 conceitos encadeados) no banco temp."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from data.database_setup import (
            Base,
            Conceito,
            ConceitoPreRequisito,
            EstadoAluno,
        )

        engine = create_engine(db_url, future=True)
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            # A -> B -> C (cadeia de pré-requisitos).
            a, b, c = Conceito(nome="A"), Conceito(nome="B"), Conceito(nome="C")
            session.add_all([a, b, c])
            session.flush()
            session.add_all([
                ConceitoPreRequisito(conceito_id=b.id, pre_requisito_id=a.id),
                ConceitoPreRequisito(conceito_id=c.id, pre_requisito_id=b.id),
            ])
            proficiencias = {a.id: 0.8, b.id: 0.4, c.id: 0.1}
            for cid, prof in proficiencias.items():
                session.add(
                    EstadoAluno(
                        estudante_id=estudante_id,
                        conceito_id=cid,
                        proficiencia=prof,
                    )
                )
            session.commit()
        engine.dispose()

    def test_estado_e_passo(self):
        """reset()/step() respeitam o contrato do ambiente (estilo Gym)."""
        import numpy as np

        from env.student_env import ACOES_VALIDAS, StudentEnvironment

        with StudentEnvironment(self.db_url, self.estudante_id) as env:
            # 3 conceitos -> estado = 3 * 3 = 9 dimensões.
            self.assertEqual(env.n_conceitos, 3)
            self.assertEqual(env.dim_estado, 9)

            estado = env.reset()
            self.assertIsInstance(estado, np.ndarray)
            self.assertEqual(estado.shape, (env.dim_estado,))

            novo_estado, recompensa, done, info = env.step(ACOES_VALIDAS[0])
            self.assertEqual(novo_estado.shape, (env.dim_estado,))
            self.assertIsInstance(recompensa, float)
            self.assertIsInstance(done, bool)
            self.assertIn("conceito_alvo_id", info)

    def test_acao_invalida(self):
        """Ação fora do espaço válido deve falhar de forma explícita."""
        from env.student_env import StudentEnvironment

        with StudentEnvironment(self.db_url, self.estudante_id) as env:
            env.reset()
            with self.assertRaises(ValueError):
                env.step("Pular")

    def test_integracao_agente_otimiza(self):
        """Fluxo ponta-a-ponta: agente escolhe ação, env responde, DQN otimiza."""
        from agent.dqn_agent import ACOES, DQNAgent
        from env.student_env import StudentEnvironment

        with StudentEnvironment(self.db_url, self.estudante_id) as env:
            agente = DQNAgent(dim_estado=env.dim_estado, dim_acoes=len(ACOES))
            estado = env.reset()

            acao_idx = agente.select_action(estado, epsilon=0.0)
            self.assertIn(acao_idx, range(len(ACOES)))

            # Alimenta o buffer e força ao menos um passo de otimização.
            batch_size = 4
            otimizou = False
            for _ in range(20):
                acao_idx = agente.select_action(estado, epsilon=1.0)
                proximo, recompensa, done, _info = env.step(ACOES[acao_idx])
                agente.store_transition(estado, acao_idx, recompensa, proximo, done)
                loss = agente.optimize_model(batch_size)
                if loss is not None:
                    self.assertIsInstance(loss, float)
                    otimizou = True
                estado = env.reset() if done else proximo
            self.assertTrue(otimizou, "DQN deveria ter otimizado ao menos uma vez.")


if __name__ == '__main__':
    unittest.main()
