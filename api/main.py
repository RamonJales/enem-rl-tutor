"""
main.py
========
API FastAPI do Sistema Tutor Inteligente (ITS) com DQN.

Expõe o agente como serviço HTTP para o frontend.
O agente DQN escolhe a Ação Pedagógica (Avançar / Reforçar / Remediar)
que determina o conceito e a dificuldade da próxima questão apresentada
ao aluno real.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload

# ─── Caminho do projeto ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.database_setup import (
    GRAFO_CONHECIMENTO,
    DB_URL,
    Conceito,
    ConceitoPreRequisito,
    EstadoAluno,
    Interacao,
    Questao,
)
from agent.model import DQN
from api.schemas import (
    ConceitoProficiencia,
    DesempenhoResponse,
    HistoricoItem,
    LoginRequest,
    LoginResponse,
    QuestaoResponse,
    ResponderRequest,
    ResponderResponse,
    SessaoResponse,
)

# ─── Constantes do modelo do estudante (espelhadas de student_env.py) ─────────
SENSIBILIDADE_LOGISTICA = 5.0
TAXA_APRENDIZADO = 0.12
TAXA_ESQUECIMENTO = 0.05
ALPHA_PRE_REQUISITO = 0.6
FADIGA_POR_PASSO = 0.004
ALFA0 = 1.0
BETA0 = 1.0
LAMBDA_ESQUEC = 0.95
DESVIO_MAX = 0.2887
LIMIAR_DOMINIO = 0.8
FRACAO_DOMINIO_ALVO = 0.70

# ─── Usuários demo (sem tabela de usuários no banco) ─────────────────────────
USUARIOS: dict[str, dict] = {
    "aluno": {"senha": "enem2024", "nome": "Ana Beatriz Silva", "estudante_id": 1},
    "demo":  {"senha": "demo",     "nome": "Aluno Demo",        "estudante_id": 1},
    "admin": {"senha": "admin",    "nome": "Administrador",     "estudante_id": 1},
}

# Nome legível por conceito (remove underscores)
NOMES_DISPLAY: dict[str, str] = {
    "Matematica_Basica":          "Matemática Básica",
    "Regra_Tres":                 "Regra de Três",
    "Graficos_Tabelas":           "Gráficos e Tabelas",
    "Medidas_Tendencia_Central":  "Medidas de Tendência Central",
    "Funcao_1_Grau":              "Função do 1º Grau",
    "Funcao_2_Grau":              "Função do 2º Grau",
    "Padroes_Graficos":           "Padrões e Progressões",
    "Poligonos_Regulares":        "Polígonos Regulares",
    "Circunferencia_Circulo":     "Circunferência e Círculo",
    "Formulas_Areas_Plana":       "Fórmulas de Áreas Planas",
    "Geometria_Posicao":          "Geometria de Posição",
    "Volumes_Areas_Espacial":     "Volumes e Áreas Espaciais",
}


# ─── Estado de sessão por aluno ───────────────────────────────────────────────
class SessaoAluno:
    """Estado de uma sessão de estudo em memória."""

    def __init__(
        self,
        estudante_id: int,
        nome: str,
        conceito_ids: list[int],
        conceito_nomes: dict[int, str],
        prof_inicial: dict[int, float],
        dependentes_ids: dict[int, list[int]],
        pre_requisitos_ids: dict[int, list[int]],
    ) -> None:
        self.estudante_id = estudante_id
        self.nome = nome
        self.conceito_ids = conceito_ids
        self.conceito_nomes = conceito_nomes
        self.n_conceitos = len(conceito_ids)
        self.indice_por_conceito: dict[int, int] = {
            cid: i for i, cid in enumerate(conceito_ids)
        }
        self.dependentes_ids = dependentes_ids
        self.pre_requisitos_ids = pre_requisitos_ids

        # Estado mutável da sessão
        self.prof_real: dict[int, float] = dict(prof_inicial)
        self.alpha: dict[int, float] = {cid: ALFA0 for cid in conceito_ids}
        self.beta: dict[int, float]  = {cid: BETA0 for cid in conceito_ids}
        self.conceito_atual_id: int = conceito_ids[0]
        self.fadiga: float = 0.0
        self.passos: int = 0
        self.ja_dominados: set[int] = {
            cid for cid, p in prof_inicial.items() if p >= LIMIAR_DOMINIO
        }
        self.historico: list[dict] = []
        self.questao_pendente: Optional[dict] = None  # questão exibida aguardando resposta

    # ── Crença Bayesiana ──────────────────────────────────────────────────────
    def _crenca_media(self, cid: int) -> float:
        a, b = self.alpha[cid], self.beta[cid]
        return a / (a + b)

    def _crenca_desvio(self, cid: int) -> float:
        a, b = self.alpha[cid], self.beta[cid]
        n = a + b
        return math.sqrt(a * b / (n * n * (n + 1)))

    def _atualizar_crenca(self, cid: int, acertou: bool) -> None:
        """Atualização Bayesiana da crença Beta(α,β)."""
        self.alpha[cid] = LAMBDA_ESQUEC * self.alpha[cid] + (1.0 if acertou else 0.0)
        self.beta[cid]  = LAMBDA_ESQUEC * self.beta[cid]  + (0.0 if acertou else 1.0)

    # ── Dinâmica de proficiência ───────────────────────────────────────────────
    def _dominio_pre_requisitos(self, cid: int) -> float:
        pres = self.pre_requisitos_ids.get(cid, [])
        if not pres:
            return 1.0
        return sum(self.prof_real[p] for p in pres) / len(pres)

    def _prob_acerto(self, cid: int) -> float:
        """ŷ — probabilidade estimada via média da crença (BKT)."""
        return self._crenca_media(cid)

    def _atualizar_proficiencia(self, cid: int, acertou: bool) -> float:
        """Atualiza prof_real e retorna Δ."""
        antes = self.prof_real[cid]
        if acertou:
            pre = self._dominio_pre_requisitos(cid)
            delta = TAXA_APRENDIZADO * (0.5 + 0.5 * pre)
            self.prof_real[cid] = min(1.0, antes + delta)
        else:
            self.prof_real[cid] = max(0.0, antes - TAXA_ESQUECIMENTO)
        return self.prof_real[cid] - antes

    # ── Estado (vetor de observação para o DQN) ───────────────────────────────
    def get_state(self) -> np.ndarray:
        estado = np.zeros(3 * self.n_conceitos, dtype=np.float32)
        for cid, idx in self.indice_por_conceito.items():
            estado[idx] = self._crenca_media(cid)
            estado[2 * self.n_conceitos + idx] = (
                self._crenca_desvio(cid) / DESVIO_MAX
            )
        idx_atual = self.indice_por_conceito[self.conceito_atual_id]
        estado[self.n_conceitos + idx_atual] = 1.0
        return estado

    # ── Navegação no DAG ──────────────────────────────────────────────────────
    def selecionar_conceito_alvo(self, acao: str) -> int:
        if acao == "Avançar":
            candidatos = self.dependentes_ids.get(self.conceito_atual_id, [])
        elif acao == "Remediar":
            candidatos = self.pre_requisitos_ids.get(self.conceito_atual_id, [])
        else:
            candidatos = [self.conceito_atual_id]
        if not candidatos:
            return self.conceito_atual_id
        return min(candidatos, key=lambda cid: (self._crenca_media(cid), cid))

    # ── Resumo de proficiências ───────────────────────────────────────────────
    def to_proficiencias(self) -> list[ConceitoProficiencia]:
        result = []
        for cid in self.conceito_ids:
            nome = self.conceito_nomes[cid]
            result.append(
                ConceitoProficiencia(
                    id=cid,
                    nome=nome,
                    nome_display=NOMES_DISPLAY.get(nome, nome),
                    proficiencia=round(self.prof_real[cid], 4),
                    dominado=cid in self.ja_dominados,
                    pre_requisitos=self.pre_requisitos_ids.get(cid, []),
                    dependentes=self.dependentes_ids.get(cid, []),
                )
            )
        return result

    @property
    def dominados_count(self) -> int:
        return sum(1 for cid in self.conceito_ids if self.prof_real[cid] >= LIMIAR_DOMINIO)

    @property
    def episodio_completo(self) -> bool:
        dominados = self.dominados_count
        return dominados / self.n_conceitos >= FRACAO_DOMINIO_ALVO


# ─── Dados globais carregados na inicialização ────────────────────────────────
_engine = create_engine(DB_URL, future=True)
_conceito_ids: list[int] = []
_conceito_nomes: dict[int, str] = {}
_dependentes_ids: dict[int, list[int]] = {}
_pre_requisitos_ids: dict[int, list[int]] = {}
_dqn: Optional[torch.nn.Module] = None
_sessions: dict[str, SessaoAluno] = {}


def _carregar_grafo() -> None:
    """Carrega conceitos e relações DAG do banco (executado uma vez)."""
    global _conceito_ids, _conceito_nomes, _dependentes_ids, _pre_requisitos_ids

    with Session(_engine) as s:
        conceitos = list(
            s.scalars(
                select(Conceito)
                .order_by(Conceito.id)
                .options(
                    selectinload(Conceito.dependentes),
                    selectinload(Conceito.pre_requisitos),
                )
            ).all()
        )
    _conceito_ids = [c.id for c in conceitos]
    _conceito_nomes = {c.id: c.nome for c in conceitos}
    _dependentes_ids = {c.id: [d.id for d in c.dependentes] for c in conceitos}
    _pre_requisitos_ids = {c.id: [p.id for p in c.pre_requisitos] for c in conceitos}


def _carregar_dqn() -> None:
    """Carrega a policy_net do checkpoint DQN treinado."""
    global _dqn
    weights_path = os.path.join(PROJECT_ROOT, "data", "weights", "dqn_policy.pt")
    if not os.path.exists(weights_path):
        return  # opera sem DQN (usa heurística)
    n_obs = 3 * len(_conceito_ids)
    n_acoes = 3
    _dqn = DQN(n_obs, n_acoes)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Checkpoint pode ser um dict com "policy_net" ou diretamente o state_dict
    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        _dqn.load_state_dict(checkpoint["policy_net"])
    else:
        _dqn.load_state_dict(checkpoint)
    _dqn.eval()


def _selecionar_acao_dqn(sessao: SessaoAluno) -> str:
    """Usa DQN (ou heurística simples) para escolher a ação pedagógica."""
    acoes = ("Avançar", "Reforçar", "Remediar")
    if _dqn is not None:
        state = torch.tensor(sessao.get_state(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = _dqn(state)
        idx = int(q_values.argmax().item())
        return acoes[idx]
    # Heurística: se proficiência atual >= 0.75, avança; se < 0.35, remedia.
    prof = sessao.prof_real[sessao.conceito_atual_id]
    if prof >= 0.75:
        return "Avançar"
    elif prof < 0.35:
        return "Remediar"
    return "Reforçar"


def _buscar_questao(conceito_id: int, dificuldade_alvo: float) -> Optional[dict]:
    """Busca questão do banco mais próxima da dificuldade alvo."""
    with Session(_engine) as s:
        questoes = list(
            s.scalars(
                select(Questao).where(Questao.conceito_id == conceito_id)
            ).all()
        )
    if not questoes:
        return None
    questoes_sorted = sorted(questoes, key=lambda q: abs(q.dificuldade - dificuldade_alvo))
    pool = questoes_sorted[: min(3, len(questoes_sorted))]
    q = random.choice(pool)
    try:
        alternativas = json.loads(q.alternativas) if q.alternativas else []
    except (json.JSONDecodeError, TypeError):
        alternativas = []
    return {
        "questao_id": q.id,
        "conceito_id": q.conceito_id,
        "enunciado": q.enunciado,
        "gabarito": q.gabarito,
        "alternativas": alternativas,
        "dificuldade": float(q.dificuldade),
    }


def _nivel_dificuldade(dif: float) -> str:
    if dif < 0.40:
        return "Fácil"
    elif dif < 0.65:
        return "Médio"
    return "Difícil"


def _ler_prof_inicial(estudante_id: int) -> dict[int, float]:
    prof = {cid: 0.0 for cid in _conceito_ids}
    with Session(_engine) as s:
        registros = list(
            s.scalars(
                select(EstadoAluno).where(EstadoAluno.estudante_id == estudante_id)
            ).all()
        )
    for r in registros:
        if r.conceito_id in prof:
            prof[r.conceito_id] = float(r.proficiencia)
    return prof


def _get_sessao(token: str) -> SessaoAluno:
    sessao = _sessions.get(token)
    if sessao is None:
        raise HTTPException(status_code=401, detail="Sessão inválida ou expirada. Faça login novamente.")
    return sessao


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ITS ENEM — Tutor Adaptativo",
    description="Sistema Tutor Inteligente com DQN para ENEM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    _carregar_grafo()
    _carregar_dqn()
    dqn_status = "carregado" if _dqn is not None else "não encontrado (usando heurística)"
    print(f"[ITS] Conceitos carregados: {len(_conceito_ids)}")
    print(f"[ITS] Modelo DQN: {dqn_status}")


# ─── Frontend (serve arquivos estáticos) ──────────────────────────────────────
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ─── Rotas da API ─────────────────────────────────────────────────────────────

@app.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    user = USUARIOS.get(body.username.lower())
    if user is None or user["senha"] != body.password:
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos.")

    estudante_id = user["estudante_id"]
    prof_inicial = _ler_prof_inicial(estudante_id)

    token = str(uuid.uuid4())
    sessao = SessaoAluno(
        estudante_id=estudante_id,
        nome=user["nome"],
        conceito_ids=list(_conceito_ids),
        conceito_nomes=dict(_conceito_nomes),
        prof_inicial=prof_inicial,
        dependentes_ids=dict(_dependentes_ids),
        pre_requisitos_ids=dict(_pre_requisitos_ids),
    )
    _sessions[token] = sessao

    return LoginResponse(
        token=token,
        nome=user["nome"],
        estudante_id=estudante_id,
        mensagem=f"Bem-vindo(a), {user['nome']}! Sua trilha adaptativa está pronta.",
    )


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "")
    _sessions.pop(token, None)
    return {"mensagem": "Sessão encerrada."}


@app.get("/api/sessao", response_model=SessaoResponse)
def get_sessao(authorization: str = Header(default="")) -> SessaoResponse:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)

    conceito_id = sessao.conceito_atual_id
    nome = sessao.conceito_nomes[conceito_id]
    return SessaoResponse(
        estudante_id=sessao.estudante_id,
        conceito_atual=ConceitoProficiencia(
            id=conceito_id,
            nome=nome,
            nome_display=NOMES_DISPLAY.get(nome, nome),
            proficiencia=round(sessao.prof_real[conceito_id], 4),
            dominado=conceito_id in sessao.ja_dominados,
            pre_requisitos=sessao.pre_requisitos_ids.get(conceito_id, []),
            dependentes=sessao.dependentes_ids.get(conceito_id, []),
        ),
        proficiencias=sessao.to_proficiencias(),
        passos=sessao.passos,
        fadiga=round(sessao.fadiga, 3),
        dominados=sessao.dominados_count,
        total_conceitos=sessao.n_conceitos,
    )


@app.get("/api/questao/proxima", response_model=QuestaoResponse)
def proxima_questao(authorization: str = Header(default="")) -> QuestaoResponse:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)

    # DQN escolhe ação pedagógica
    acao = _selecionar_acao_dqn(sessao)

    # Navega no DAG para o conceito-alvo
    conceito_alvo_id = sessao.selecionar_conceito_alvo(acao)

    # ŷ = média da crença (probabilidade esperada de acerto)
    prob_esperada = sessao._crenca_media(conceito_alvo_id)

    # Busca questão com dificuldade ajustada à probabilidade esperada
    dificuldade_alvo = 1.0 - prob_esperada  # questão calibrada para o nível do aluno
    questao_data = _buscar_questao(conceito_alvo_id, dificuldade_alvo)
    if questao_data is None:
        raise HTTPException(status_code=404, detail=f"Nenhuma questão encontrada para o conceito {conceito_alvo_id}.")

    # Armazena questão pendente na sessão (será usada ao responder)
    sessao.questao_pendente = {
        **questao_data,
        "acao_rl": acao,
        "prob_esperada": prob_esperada,
        "conceito_alvo_id": conceito_alvo_id,
    }

    nome_conceito = sessao.conceito_nomes[conceito_alvo_id]
    return QuestaoResponse(
        questao_id=questao_data["questao_id"],
        conceito_id=conceito_alvo_id,
        conceito_nome=nome_conceito,
        conceito_display=NOMES_DISPLAY.get(nome_conceito, nome_conceito),
        enunciado=questao_data["enunciado"],
        alternativas=questao_data.get("alternativas", []),
        dificuldade=questao_data["dificuldade"],
        nivel=_nivel_dificuldade(questao_data["dificuldade"]),
        acao_rl=acao,
        prob_esperada=round(prob_esperada, 3),
        passos=sessao.passos,
        dominados=sessao.dominados_count,
        total_conceitos=sessao.n_conceitos,
    )


@app.post("/api/questao/responder", response_model=ResponderResponse)
def responder(body: ResponderRequest, authorization: str = Header(default="")) -> ResponderResponse:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)

    if sessao.questao_pendente is None:
        raise HTTPException(status_code=400, detail="Nenhuma questão ativa. Solicite a próxima questão primeiro.")

    pendente = sessao.questao_pendente
    if pendente["questao_id"] != body.questao_id:
        raise HTTPException(status_code=400, detail="ID de questão não corresponde à questão ativa.")

    gabarito = pendente["gabarito"]
    acertou = body.resposta.strip().lower() == gabarito.strip().lower()
    conceito_id = pendente["conceito_alvo_id"]
    prob_esperada = pendente["prob_esperada"]

    # Atualiza crença Bayesiana
    sessao._atualizar_crenca(conceito_id, acertou)

    # Atualiza proficiência real
    delta = sessao._atualizar_proficiencia(conceito_id, acertou)

    # Verifica domínio
    if sessao.prof_real[conceito_id] >= LIMIAR_DOMINIO:
        sessao.ja_dominados.add(conceito_id)

    # Recompensa simplificada (similar à função do ambiente)
    recompensa = float(acertou) - prob_esperada

    # Atualiza conceito atual (move para o conceito trabalhado)
    sessao.conceito_atual_id = conceito_id

    # Fadiga e passos
    sessao.fadiga = min(1.0, sessao.fadiga + FADIGA_POR_PASSO)
    sessao.passos += 1

    # Histórico da sessão
    nome_conceito = sessao.conceito_nomes[conceito_id]
    sessao.historico.append({
        "passo": sessao.passos,
        "conceito": nome_conceito,
        "conceito_display": NOMES_DISPLAY.get(nome_conceito, nome_conceito),
        "acertou": acertou,
        "recompensa": round(recompensa, 4),
        "proficiencia_pos": round(sessao.prof_real[conceito_id], 4),
        "acao_rl": pendente["acao_rl"],
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Limpa questão pendente
    sessao.questao_pendente = None

    mensagem = ""
    if acertou:
        if delta > 0.05:
            mensagem = "Excelente! Sua proficiência neste conceito aumentou significativamente."
        else:
            mensagem = "Correto! Continue praticando para consolidar o conhecimento."
    else:
        mensagem = f"Incorreto. A resposta correta era: {gabarito}. Não desanime!"

    return ResponderResponse(
        correto=acertou,
        gabarito=gabarito,
        recompensa=round(recompensa, 4),
        delta_proficiencia=round(delta, 4),
        nova_proficiencia=round(sessao.prof_real[conceito_id], 4),
        conceito_nome=nome_conceito,
        conceito_display=NOMES_DISPLAY.get(nome_conceito, nome_conceito),
        mensagem=mensagem,
        passos=sessao.passos,
        fadiga=round(sessao.fadiga, 3),
        dominados=sessao.dominados_count,
        episodio_completo=sessao.episodio_completo,
    )


@app.get("/api/desempenho", response_model=DesempenhoResponse)
def desempenho(authorization: str = Header(default="")) -> DesempenhoResponse:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)

    hist = sessao.historico
    total = len(hist)
    acertos = sum(1 for h in hist if h["acertou"])
    taxa = round(acertos / total, 4) if total > 0 else 0.0
    recompensa_total = round(sum(h["recompensa"] for h in hist), 4)

    return DesempenhoResponse(
        total_questoes=total,
        total_acertos=acertos,
        taxa_acerto=taxa,
        recompensa_total=recompensa_total,
        dominados=sessao.dominados_count,
        total_conceitos=sessao.n_conceitos,
        proficiencias=sessao.to_proficiencias(),
        historico=[HistoricoItem(**h) for h in hist],
        recompensa_por_passo=[h["recompensa"] for h in hist],
        acertos_por_passo=[1 if h["acertou"] else 0 for h in hist],
    )


@app.get("/api/conceitos")
def conceitos(authorization: str = Header(default="")) -> list[dict]:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)
    result = []
    for cid in sessao.conceito_ids:
        nome = sessao.conceito_nomes[cid]
        result.append({
            "id": cid,
            "nome": nome,
            "nome_display": NOMES_DISPLAY.get(nome, nome),
            "proficiencia": round(sessao.prof_real[cid], 4),
            "crenca_media": round(sessao._crenca_media(cid), 4),
            "dominado": sessao.prof_real[cid] >= LIMIAR_DOMINIO,
            "pre_requisitos": sessao.pre_requisitos_ids.get(cid, []),
            "dependentes": sessao.dependentes_ids.get(cid, []),
        })
    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "conceitos": len(_conceito_ids),
        "dqn": _dqn is not None,
        "sessoes_ativas": len(_sessions),
    }
