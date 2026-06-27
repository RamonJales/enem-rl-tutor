"""Pacote do Ambiente Simulador (Modelo do Aluno) do Tutor Inteligente.

Reúne os três componentes do domínio do Ambiente:

- ``StudentEnvironment`` — o simulador (estilo Gym) que treina o agente DQN.
- ``KnowledgeGraph`` / ``ConceptNode`` — o Modelo de Domínio (regras do DAG de
  pré-requisitos), utilitário de propagação puro e reutilizável.
- ``BaseStudentBot`` / ``ConsistentStudentBot`` / ``GuessingStudentBot`` —
  perfis simulados de alunos (TRI) para avaliação/robustez da política.

Apenas ``StudentEnvironment`` está no caminho crítico do treino do DQN; os
demais são componentes de domínio auxiliares (testes, API, notebooks).
"""

from env.bots import (
    BaseStudentBot,
    ConsistentStudentBot,
    GuessingStudentBot,
)
from env.knowledge_graph import ConceptNode, KnowledgeGraph
from env.student_env import StudentEnvironment

__all__ = [
    "StudentEnvironment",
    "KnowledgeGraph",
    "ConceptNode",
    "BaseStudentBot",
    "ConsistentStudentBot",
    "GuessingStudentBot",
]
