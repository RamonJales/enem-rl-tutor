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
from sqlalchemy import create_engine, func, select
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
    Usuario,
    gerar_hash_senha,
    verificar_senha,
)
from agent.model import DQN
# Fonte ÚNICA da dinâmica do aluno (mesma do treino): nada de constantes/lógica
# "espelhadas" na API — importamos direto do ambiente para não haver divergência.
from env.student_env import (
    StudentEnvironment,
    LIMIAR_DOMINIO,
    FRACAO_DOMINIO_ALVO,
    FADIGA_POR_PASSO,
    DESVIO_MAX,
)
from api.schemas import (
    ConceitoProficiencia,
    DesempenhoResponse,
    HistoricoItem,
    LoginRequest,
    LoginResponse,
    QuestaoResponse,
    RegisterRequest,
    ResponderRequest,
    ResponderResponse,
    SessaoResponse,
)

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
    """
    Estado de uma sessão de estudo em memória.

    A dinâmica (crença BKT, vetor de estado, navegação no DAG, atualização de
    proficiência) NÃO é reimplementada aqui: a sessão encapsula uma instância do
    MESMO `StudentEnvironment` usado no treino do DQN. Isso garante FONTE ÚNICA
    DE VERDADE — a API e o agente não podem divergir (foi uma divergência dessas,
    reimplementada à mão, que causou um bug de estado fora da distribuição).

    A API apenas ORQUESTRA o fluxo com o aluno REAL: serve a questão, recebe a
    resposta e repassa o resultado ao ambiente (em vez de o ambiente amostrar o
    acerto internamente, como faz no treino).
    """

    def __init__(self, estudante_id: int, nome: str) -> None:
        self.estudante_id = estudante_id
        self.nome = nome
        # Fonte única da dinâmica: o ambiente de treino, em modo de inferência.
        # reset() carrega a proficiência persistida (EstadoAluno.proficiencia);
        # logo abaixo carregamos também a CRENÇA persistida (α, β).
        self.env = StudentEnvironment(DB_URL, estudante_id)
        self.env.reset()
        self._carregar_crenca_persistida()
        self.conceito_nomes: dict[int, str] = dict(_conceito_nomes)
        self.passos: int = 0
        self.historico: list[dict] = []
        self.questao_pendente: Optional[dict] = None  # questão aguardando resposta

    # ── Estado vivo (delegado ao ambiente) ────────────────────────────────────
    @property
    def conceito_ids(self) -> list[int]:
        return self.env.conceito_ids

    @property
    def n_conceitos(self) -> int:
        return self.env.n_conceitos

    @property
    def prof_real(self) -> dict[int, float]:
        return self.env.prof_real

    @property
    def conceito_atual_id(self) -> int:
        return self.env.conceito_atual_id

    @conceito_atual_id.setter
    def conceito_atual_id(self, value: int) -> None:
        self.env.conceito_atual_id = value

    @property
    def fadiga(self) -> float:
        return self.env.fadiga

    @fadiga.setter
    def fadiga(self, value: float) -> None:
        self.env.fadiga = value

    @property
    def ja_dominados(self) -> set[int]:
        return self.env.ja_dominados

    @property
    def pre_requisitos_ids(self) -> dict[int, list[int]]:
        return self.env.pre_requisitos_ids

    @property
    def dependentes_ids(self) -> dict[int, list[int]]:
        return self.env.dependentes_ids

    # ── Dinâmica (delegada ao ambiente — fonte única) ─────────────────────────
    def get_state(self) -> np.ndarray:
        """Vetor de observação do DQN (crença + one-hot + incerteza)."""
        return self.env._get_state()

    def _crenca_media(self, cid: int) -> float:
        return self.env._crenca_media(cid)

    def selecionar_conceito_alvo(self, acao: str) -> int:
        return self.env._selecionar_conceito_alvo(acao)

    def _atualizar_crenca(self, cid: int, acertou: bool) -> None:
        """Repassa a resposta REAL à crença Bayesiana do ambiente."""
        self.env._atualizar_crenca(cid, 1 if acertou else 0)

    def _atualizar_proficiencia(self, cid: int, acertou: bool) -> float:
        """Atualiza a proficiência via ambiente e retorna o Δ."""
        return self.env._atualizar_proficiencia(cid, acertou)

    def fechar(self) -> None:
        """Libera a conexão de banco do ambiente ao encerrar a sessão."""
        self.env.close()

    # ── Persistência longitudinal (proficiência + crença por aluno) ───────────
    def _carregar_crenca_persistida(self) -> None:
        """Injeta no ambiente a crença Beta(α,β) salva no banco para este aluno.

        A proficiência já é carregada pelo `env.reset()`; aqui sobrescrevemos a
        crença (que o reset inicia no prior) com a que foi persistida, de modo
        que o tutor RETOME o que sabia sobre o aluno entre sessões.
        """
        with Session(_engine) as db:
            registros = db.scalars(
                select(EstadoAluno).where(
                    EstadoAluno.estudante_id == self.estudante_id
                )
            ).all()
        for reg in registros:
            if reg.conceito_id in self.env.alpha:
                self.env.alpha[reg.conceito_id] = float(reg.alpha)
                self.env.beta[reg.conceito_id] = float(reg.beta)

    def persistir(self, cid: int) -> None:
        """Grava (upsert) a proficiência e a crença do conceito no banco."""
        with Session(_engine) as db:
            reg = db.scalars(
                select(EstadoAluno).where(
                    EstadoAluno.estudante_id == self.estudante_id,
                    EstadoAluno.conceito_id == cid,
                )
            ).first()
            if reg is None:
                reg = EstadoAluno(estudante_id=self.estudante_id, conceito_id=cid)
                db.add(reg)
            reg.proficiencia = float(self.env.prof_real[cid])
            reg.alpha = float(self.env.alpha[cid])
            reg.beta = float(self.env.beta[cid])
            db.commit()

    # ── Crença observável (estimativa + incerteza) ────────────────────────────
    def incerteza(self, cid: int) -> float:
        """Incerteza da estimativa em [0,1] (1 = sem evidência; 0 = muito certo)."""
        return min(1.0, self.env._crenca_desvio(cid) / DESVIO_MAX)

    def estimado_dominado(self, cid: int) -> bool:
        """Domínio ESTIMADO: a crença média cruza o limiar de domínio."""
        return self._crenca_media(cid) >= LIMIAR_DOMINIO

    def conceito_proficiencia(self, cid: int) -> ConceitoProficiencia:
        """Monta o resumo de um conceito a partir da CRENÇA (não da verdade)."""
        nome = self.conceito_nomes[cid]
        return ConceitoProficiencia(
            id=cid,
            nome=nome,
            nome_display=NOMES_DISPLAY.get(nome, nome),
            proficiencia=round(self._crenca_media(cid), 4),   # estimativa
            incerteza=round(self.incerteza(cid), 4),
            dominado=self.estimado_dominado(cid),
            pre_requisitos=self.pre_requisitos_ids.get(cid, []),
            dependentes=self.dependentes_ids.get(cid, []),
        )

    # ── Resumo de proficiências ───────────────────────────────────────────────
    def to_proficiencias(self) -> list[ConceitoProficiencia]:
        return [self.conceito_proficiencia(cid) for cid in self.conceito_ids]

    @property
    def dominados_count(self) -> int:
        # Domínio medido pela ESTIMATIVA do sistema (não pela verdade oculta).
        return sum(1 for cid in self.conceito_ids if self.estimado_dominado(cid))

    @property
    def episodio_completo(self) -> bool:
        return self.dominados_count / self.n_conceitos >= FRACAO_DOMINIO_ALVO


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

def _abrir_sessao(estudante_id: int, nome: str) -> str:
    """Cria a sessão de estudo (carrega o estado persistido) e retorna o token."""
    token = str(uuid.uuid4())
    _sessions[token] = SessaoAluno(estudante_id=estudante_id, nome=nome)
    return token


@app.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    with Session(_engine) as db:
        user = db.scalars(
            select(Usuario).where(Usuario.username == body.username.strip().lower())
        ).first()
    if user is None or not verificar_senha(body.password, user.senha_hash):
        raise HTTPException(status_code=401, detail="Usuário ou senha inválidos.")

    # A sessão carrega proficiência + crença persistidas: o tutor RETOMA de onde
    # o aluno parou (não reinicia a crença no prior a cada login).
    token = _abrir_sessao(user.estudante_id, user.nome)
    return LoginResponse(
        token=token,
        nome=user.nome,
        estudante_id=user.estudante_id,
        mensagem=f"Bem-vindo(a) de volta, {user.nome}! Sua trilha continua de onde parou.",
    )


@app.post("/api/auth/register", response_model=LoginResponse)
def register(body: RegisterRequest) -> LoginResponse:
    username = body.username.strip().lower()
    if not username or not body.password:
        raise HTTPException(status_code=400, detail="Usuário e senha são obrigatórios.")
    if len(body.password) < 4:
        raise HTTPException(status_code=400, detail="A senha deve ter ao menos 4 caracteres.")

    with Session(_engine) as db:
        if db.scalars(select(Usuario).where(Usuario.username == username)).first():
            raise HTTPException(status_code=409, detail="Este usuário já existe.")
        # Novo estudante_id = maior existente + 1 (aluno começa "do zero").
        max_id = db.scalars(select(func.max(Usuario.estudante_id))).first() or 0
        novo = Usuario(
            username=username,
            senha_hash=gerar_hash_senha(body.password),
            nome=(body.nome.strip() or username.capitalize()),
            estudante_id=max_id + 1,
        )
        db.add(novo)
        db.commit()
        estudante_id, nome_user = novo.estudante_id, novo.nome

    token = _abrir_sessao(estudante_id, nome_user)
    return LoginResponse(
        token=token,
        nome=nome_user,
        estudante_id=estudante_id,
        mensagem=f"Conta criada! Bem-vindo(a), {nome_user}. Sua trilha adaptativa começa agora.",
    )


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "")
    sessao = _sessions.pop(token, None)
    if sessao is not None:
        sessao.fechar()  # libera a conexão de banco do ambiente.
    return {"mensagem": "Sessão encerrada."}


@app.get("/api/sessao", response_model=SessaoResponse)
def get_sessao(authorization: str = Header(default="")) -> SessaoResponse:
    token = authorization.replace("Bearer ", "")
    sessao = _get_sessao(token)

    conceito_id = sessao.conceito_atual_id
    return SessaoResponse(
        estudante_id=sessao.estudante_id,
        conceito_atual=sessao.conceito_proficiencia(conceito_id),
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

    # Busca questão calibrada ao nível do aluno: a dificuldade-alvo ACOMPANHA a
    # crença de acerto (quanto mais o aluno domina o conceito, mais difícil a
    # questão — Zona de Desenvolvimento Proximal). Antes usava 1 - prob_esperada,
    # que invertia (dava questão difícil a quem não dominava → frustração).
    dificuldade_alvo = prob_esperada
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

    # Estimativa do sistema (crença) ANTES de observar a resposta.
    est_antes = sessao._crenca_media(conceito_id)

    # Repassa a resposta REAL ao ambiente: atualiza a CRENÇA (o que o sistema
    # estima sobre o aluno). A proficiência interna do simulador também é
    # atualizada, mas NÃO é exibida ao aluno (é uma ficção para um aluno real).
    sessao._atualizar_crenca(conceito_id, acertou)
    sessao._atualizar_proficiencia(conceito_id, acertou)

    est_depois = sessao._crenca_media(conceito_id)
    delta_est = est_depois - est_antes

    # Sinal observado (acerto − esperado): registrado só para analytics/treino
    # futuro. NÃO é a recompensa do MDP (que depende da proficiência real oculta),
    # por isso NÃO é exibido ao aluno.
    sinal_observado = float(acertou) - prob_esperada
    try:
        with Session(_engine) as _db:
            _db.add(
                Interacao(
                    estudante_id=sessao.estudante_id,
                    questao_id=pendente["questao_id"],
                    acao_rl=pendente["acao_rl"],
                    prob_esperada=float(prob_esperada),
                    resultado_real=int(acertou),
                    recompensa=float(sinal_observado),
                )
            )
            _db.commit()
    except Exception as exc:  # pragma: no cover - log best-effort
        print(f"[ITS] Falha ao registrar Interacao: {exc}")

    # Move o foco para o conceito trabalhado; atualiza fadiga/passos.
    sessao.conceito_atual_id = conceito_id
    sessao.fadiga = min(1.0, sessao.fadiga + FADIGA_POR_PASSO)
    sessao.passos += 1

    # Persiste estimativa (crença) + proficiência do conceito (retomável depois).
    sessao.persistir(conceito_id)

    nome_conceito = sessao.conceito_nomes[conceito_id]
    sessao.historico.append({
        "passo": sessao.passos,
        "conceito": nome_conceito,
        "conceito_display": NOMES_DISPLAY.get(nome_conceito, nome_conceito),
        "acertou": acertou,
        "estimativa_pos": round(est_depois, 4),
        "acao_rl": pendente["acao_rl"],
        "timestamp": datetime.utcnow().isoformat(),
    })

    sessao.questao_pendente = None

    if acertou:
        if delta_est > 0.05:
            mensagem = "Boa! O sistema aumentou a estimativa de que você domina este conceito."
        else:
            mensagem = "Correto! Continue praticando para consolidar."
    else:
        mensagem = f"Incorreto. A resposta correta era: {gabarito}. Não desanime!"

    return ResponderResponse(
        correto=acertou,
        gabarito=gabarito,
        delta_estimativa=round(delta_est, 4),
        nova_estimativa=round(est_depois, 4),
        incerteza=round(sessao.incerteza(conceito_id), 4),
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

    return DesempenhoResponse(
        total_questoes=total,
        total_acertos=acertos,
        taxa_acerto=taxa,
        dominados=sessao.dominados_count,
        total_conceitos=sessao.n_conceitos,
        proficiencias=sessao.to_proficiencias(),
        historico=[HistoricoItem(**h) for h in hist],
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
            "proficiencia": round(sessao._crenca_media(cid), 4),   # estimativa
            "incerteza": round(sessao.incerteza(cid), 4),
            "dominado": sessao.estimado_dominado(cid),
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
