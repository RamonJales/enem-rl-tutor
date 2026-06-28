"""
schemas.py
===========
Schemas Pydantic para a API FastAPI do Sistema Tutor Inteligente.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


# ─── Autenticação ─────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    nome: str
    estudante_id: int
    mensagem: str


# ─── Sessão / Estado ──────────────────────────────────────────────────────────

class ConceitoProficiencia(BaseModel):
    id: int
    nome: str
    nome_display: str
    proficiencia: float
    dominado: bool
    pre_requisitos: list[int]
    dependentes: list[int]


class SessaoResponse(BaseModel):
    estudante_id: int
    conceito_atual: ConceitoProficiencia
    proficiencias: list[ConceitoProficiencia]
    passos: int
    fadiga: float
    dominados: int
    total_conceitos: int


# ─── Questão ──────────────────────────────────────────────────────────────────

class QuestaoResponse(BaseModel):
    questao_id: int
    conceito_id: int
    conceito_nome: str
    conceito_display: str
    enunciado: str
    alternativas: list[str]   # ["A) texto", "B) texto", "C) texto", "D) texto"]
    dificuldade: float
    nivel: str                # "Fácil" | "Médio" | "Difícil"
    acao_rl: str              # "Avançar" | "Reforçar" | "Remediar"
    prob_esperada: float      # ŷ — probabilidade estimada de acerto
    passos: int
    dominados: int
    total_conceitos: int


class ResponderRequest(BaseModel):
    questao_id: int
    resposta: str


class ResponderResponse(BaseModel):
    correto: bool
    gabarito: str
    recompensa: float
    delta_proficiencia: float
    nova_proficiencia: float
    conceito_nome: str
    conceito_display: str
    mensagem: str
    passos: int
    fadiga: float
    dominados: int
    episodio_completo: bool


# ─── Desempenho ───────────────────────────────────────────────────────────────

class HistoricoItem(BaseModel):
    passo: int
    conceito: str
    conceito_display: str
    acertou: bool
    recompensa: float
    proficiencia_pos: float
    acao_rl: str
    timestamp: str


class DesempenhoResponse(BaseModel):
    total_questoes: int
    total_acertos: int
    taxa_acerto: float
    recompensa_total: float
    dominados: int
    total_conceitos: int
    proficiencias: list[ConceitoProficiencia]
    historico: list[HistoricoItem]
    recompensa_por_passo: list[float]
    acertos_por_passo: list[int]  # 1 = acerto, 0 = erro
