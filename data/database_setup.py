"""
database_setup.py
==================
Camada de persistência do Sistema Tutor Inteligente (ITS) com Deep RL (DQN).

Este módulo define o schema relacional (SQLAlchemy 2.0 - Declarative Base) que
sustenta os três pilares da modelagem de Aprendizado por Reforço:

    1. GRAFO PEDAGÓGICO (Conceito + ConceitoPreRequisito):
       Estrutura de DAG (Grafo Direcionado Acíclico) que define a ordem de
       pré-requisitos. É o "mapa" que o agente percorre quando escolhe a Ação:
         - "Avançar"  -> caminha para conceitos SUCESSORES (filhos no DAG).
         - "Reforçar" -> permanece no MESMO conceito (mais prática).
         - "Remediar" -> retrocede para os PRÉ-REQUISITOS (pais no DAG).

    2. ESTADO (S) DO RL (EstadoAluno):
       Guarda a Proficiência (probabilidade estimada de acerto) por conceito.
       Junto com histórico recente e fadiga (calculados em tempo de execução no
       pacote 'env'), compõe o vetor de Estado contínuo da Q-Network.

    3. EXPERIENCE REPLAY (Interacao):
       Cada linha é uma transição/experiência do DQN. Armazena a ação tomada,
       a probabilidade esperada de acerto (ŷ), o resultado real (y) e a
       recompensa dinâmica R_t = y - ŷ. Esse buffer é amostrado para treinar a
       rede neural de forma estável (quebrando a correlação temporal).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

# Caminho do SQLite local para desenvolvimento (arquivo ao lado deste script).
DB_PATH = os.path.join(os.path.dirname(__file__), "enem_tutor.db")
DB_URL = f"sqlite:///{DB_PATH}"



class Base(DeclarativeBase):
    """Base declarativa única para todo o schema do Tutor."""
    pass


class Conceito(Base):
    """
    Nó do Grafo de Conhecimento (ex: 'Funcao_1_Grau').

    O conjunto de Conceitos define implicitamente as transições possíveis do
    ambiente: o agente nunca recomenda "o ID de uma questão", mas uma Ação
    Pedagógica que o backend traduz em "buscar questão de qual conceito".
    """
    __tablename__ = "conceito"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nome: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)

    # Relação Many-to-Many AUTO-REFERENCIADA via tabela associativa.
    # 'pre_requisitos' -> conceitos que ESTE conceito exige (pais no DAG / Remediar).
    pre_requisitos: Mapped[list["Conceito"]] = relationship(
        "Conceito",
        secondary="conceito_pre_requisito",
        primaryjoin="Conceito.id == ConceitoPreRequisito.conceito_id",
        secondaryjoin="Conceito.id == ConceitoPreRequisito.pre_requisito_id",
        back_populates="dependentes",
    )
    # 'dependentes' -> conceitos que dependem deste (filhos no DAG / Avançar).
    dependentes: Mapped[list["Conceito"]] = relationship(
        "Conceito",
        secondary="conceito_pre_requisito",
        primaryjoin="Conceito.id == ConceitoPreRequisito.pre_requisito_id",
        secondaryjoin="Conceito.id == ConceitoPreRequisito.conceito_id",
        back_populates="pre_requisitos",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Conceito id={self.id} nome={self.nome!r}>"


class ConceitoPreRequisito(Base):
    """
    Tabela associativa (Many-to-Many) das arestas do DAG.

    Cada linha é uma aresta direcionada (pre_requisito_id -> conceito_id),
    significando "para estudar 'conceito_id', recomenda-se dominar
    'pre_requisito_id' antes". Modelar como tabela própria (e não só um
    Table()) permite múltiplos pré-requisitos por conceito e mantém a
    fidelidade do Grafo Direcionado Acíclico.
    """
    __tablename__ = "conceito_pre_requisito"

    conceito_id: Mapped[int] = mapped_column(
        ForeignKey("conceito.id"), primary_key=True
    )
    pre_requisito_id: Mapped[int] = mapped_column(
        ForeignKey("conceito.id"), primary_key=True
    )

class Questao(Base):
    """
    Item avaliativo vinculado a um Conceito.

    A 'dificuldade' (0.0 fácil -> 1.0 difícil) é central para o cálculo de ŷ
    (probabilidade esperada de acerto): comparando a proficiência do aluno no
    conceito com a dificuldade da questão, o sistema estima ŷ e, após a
    resposta, computa a recompensa R_t = y - ŷ.
    """
    __tablename__ = "questao"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conceito_id: Mapped[int] = mapped_column(
        ForeignKey("conceito.id"), nullable=False
    )
    dificuldade: Mapped[float] = mapped_column(Float, nullable=False)
    enunciado: Mapped[str] = mapped_column(Text, nullable=False)
    gabarito: Mapped[str] = mapped_column(String(4), nullable=False)   # "A" | "B" | "C" | "D"
    # JSON list: ["A) texto", "B) texto", "C) texto", "D) texto"]
    alternativas: Mapped[str] = mapped_column(Text, nullable=False, default="[]")

    conceito: Mapped["Conceito"] = relationship("Conceito")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Questao id={self.id} conceito_id={self.conceito_id} dif={self.dificuldade}>"


class EstadoAluno(Base):
    """
    Proficiência do aluno por conceito = componente principal do vetor de Estado S.

    Proficiencia in [0, 1] é a probabilidade estimada de acerto no conceito.
    O vetor completo de Estado da Q-Network é montado no pacote 'env' juntando:
      - Proficiência por conceito (esta tabela),
      - Histórico de acertos recentes,
      - Contexto da sessão (fadiga).

    PERSISTÊNCIA LONGITUDINAL: além da proficiência, guarda a CRENÇA Bayesiana
    Beta(alpha, beta) por conceito. Assim, ao reabrir a sessão, o tutor RETOMA o
    que sabia sobre o aluno (em vez de reiniciar a crença no prior a cada login).
    """
    __tablename__ = "estado_aluno"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    estudante_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    conceito_id: Mapped[int] = mapped_column(
        ForeignKey("conceito.id"), nullable=False
    )
    proficiencia: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Crença Bayesiana persistida (prior padrão = Beta(1, 1), uniforme).
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    beta: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    conceito: Mapped["Conceito"] = relationship("Conceito")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EstadoAluno estudante={self.estudante_id} "
            f"conceito_id={self.conceito_id} prof={self.proficiencia:.2f}>"
        )


class Usuario(Base):
    """
    Conta de usuário (aluno) com autenticação real.

    A senha é guardada como HASH (PBKDF2-HMAC-SHA256 salgado), nunca em texto
    puro. Cada usuário tem um `estudante_id` próprio, que liga a conta ao seu
    estado de aprendizado (EstadoAluno/Interacao).
    """
    __tablename__ = "usuario"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(80), unique=True, nullable=False, index=True
    )
    senha_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nome: Mapped[str] = mapped_column(String(120), nullable=False)
    estudante_id: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False, index=True
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Usuario {self.username!r} estudante_id={self.estudante_id}>"


class Interacao(Base):
    """
    Buffer de Experience Replay: uma transição (s, a, r) por linha.

    Campos espelham a tupla de RL:
      - acao_rl        : Ação A tomada ('Avançar' | 'Reforçar' | 'Remediar').
      - prob_esperada  : ŷ, probabilidade esperada de acerto ANTES de responder.
      - resultado_real : y in {0, 1}, acerto/erro observado.
      - recompensa     : R_t = y - ŷ (recompensa dinâmica).

    Amostrar este histórico em minibatches descorrelaciona as experiências e
    estabiliza o treino do DQN.
    """
    __tablename__ = "interacao"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    estudante_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    questao_id: Mapped[int] = mapped_column(
        ForeignKey("questao.id"), nullable=False
    )
    acao_rl: Mapped[str] = mapped_column(String(20), nullable=False)
    prob_esperada: Mapped[float] = mapped_column(Float, nullable=False)
    resultado_real: Mapped[int] = mapped_column(Integer, nullable=False)
    recompensa: Mapped[float] = mapped_column(Float, nullable=False)

    questao: Mapped["Questao"] = relationship("Questao")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Interacao estudante={self.estudante_id} acao={self.acao_rl!r} "
            f"R_t={self.recompensa:+.2f}>"
        )


# Grafo de Conhecimento exato (conceito -> lista de pré-requisitos).
# A ordem de inserção respeita o DAG: todo pré-requisito é criado antes do
# conceito que depende dele.
GRAFO_CONHECIMENTO: dict[str, list[str]] = {
    "Matematica_Basica": [],
    "Regra_Tres": ["Matematica_Basica"],
    "Graficos_Tabelas": ["Matematica_Basica"],
    "Medidas_Tendencia_Central": ["Graficos_Tabelas"],
    "Funcao_1_Grau": ["Matematica_Basica"],
    "Funcao_2_Grau": ["Funcao_1_Grau"],
    "Padroes_Graficos": ["Funcao_1_Grau"],
    "Poligonos_Regulares": ["Matematica_Basica"],
    "Circunferencia_Circulo": ["Matematica_Basica"],
    "Formulas_Areas_Plana": ["Poligonos_Regulares", "Circunferencia_Circulo"],
    "Geometria_Posicao": ["Formulas_Areas_Plana"],
    "Volumes_Areas_Espacial": ["Formulas_Areas_Plana", "Geometria_Posicao"],
}

# Aluno fictício usado para o estado inicial e simulação de interações.
ESTUDANTE_TESTE_ID = 1

# ---------------------------------------------------------------------------- #
# Banco de questões em GRADE (conceito × nível de dificuldade)
# ---------------------------------------------------------------------------- #
# Cada conceito recebe questões nos 3 níveis do ENEM (Fácil/Médio/Difícil),
# geradas de forma PARAMETRIZADA (números sorteados) para encher o banco barato.
# Para o simulador, o que importa é o VALOR de `dificuldade`; o enunciado é
# apenas ilustrativo. Itens reais do ENEM podem ser reservados para avaliação.

# Dificuldade-base por conceito (cresce com a profundidade no DAG, preservando o
# gradiente de currículo). Os 3 níveis aplicam um deslocamento em torno dela.
DIFICULDADE_BASE: dict[str, float] = {
    "Matematica_Basica": 0.25,
    "Regra_Tres": 0.35,
    "Graficos_Tabelas": 0.35,
    "Medidas_Tendencia_Central": 0.45,
    "Funcao_1_Grau": 0.45,
    "Funcao_2_Grau": 0.60,
    "Padroes_Graficos": 0.55,
    "Poligonos_Regulares": 0.40,
    "Circunferencia_Circulo": 0.45,
    "Formulas_Areas_Plana": 0.60,
    "Geometria_Posicao": 0.70,
    "Volumes_Areas_Espacial": 0.80,
}

# Nível -> (deslocamento da dificuldade-base, fator de magnitude dos números).
NIVEIS: dict[str, tuple[float, int]] = {
    "Fácil": (-0.12, 1),
    "Médio": (0.0, 2),
    "Difícil": (0.12, 3),
}

# Quantas questões por célula (conceito × nível). 12 × 3 × 5 = 180 questões.
QUESTOES_POR_CELULA = 5


# ---------------------------------------------------------------------------- #
# Helpers para geração de alternativas objetivas
# ---------------------------------------------------------------------------- #

def _montar_alternativas(gabarito_val: str, erradas: list[str]) -> tuple[list[str], str]:
    """
    Embaralha gabarito + 3 alternativas erradas e devolve
    (lista "A) ...", "B) ...", ..., letra_gabarito).
    """
    pool = list(dict.fromkeys([gabarito_val] + erradas))  # sem duplicatas
    while len(pool) < 4:
        # Adiciona variação numérica simples se faltar
        try:
            v = int(pool[0].replace("R$ ", "").replace("°", "").replace(",", ".").split()[0])
            pool.append(str(v + len(pool) * 3))
        except Exception:
            pool.append(pool[0] + "?")
    pool = pool[:4]
    random.shuffle(pool)
    letras = ["A", "B", "C", "D"]
    alternativas = [f"{letras[i]}) {pool[i]}" for i in range(4)]
    gabarito_letra = letras[pool.index(gabarito_val)]
    return alternativas, gabarito_letra


def _erros_num(val: int | float, n: int = 3) -> list[str]:
    """Gera n valores numéricos errados plausíveis a partir do valor correto."""
    fatores = [0.5, 0.75, 1.5, 2.0, 0.8, 1.25, 3.0]
    random.shuffle(fatores)
    erros: list[str] = []
    seen = {val}
    for f in fatores:
        v = round(val * f)
        if v not in seen and v > 0:
            seen.add(v)
            erros.append(str(v))
        if len(erros) >= n:
            break
    # fallback com offsets
    offset = 1
    while len(erros) < n:
        v = int(val) + offset * (1 if offset % 2 == 0 else -1)
        if v not in seen and v > 0:
            seen.add(v)
            erros.append(str(v))
        offset += 1
    return erros[:n]


# ---------------------------------------------------------------------------- #
# Geradores (enunciado, gabarito_literal, [erradas])
# ---------------------------------------------------------------------------- #

def _g_matematica_basica(m: int) -> tuple[str, str, list[str]]:
    p = random.choice([10, 20, 25, 50])
    n = 20 * random.randint(2, 6) * m
    correto = p * n // 100
    erradas = _erros_num(correto)
    return f"Quanto é {p}% de {n}?", str(correto), erradas


def _g_regra_tres(m: int) -> tuple[str, str, list[str]]:
    unit = random.randint(2, 9) * m
    a, b = random.randint(2, 6), random.randint(2, 8)
    correto = unit * b
    erradas = [f"R$ {unit * a}", f"R$ {correto + unit}", f"R$ {correto - unit}"]
    return (
        f"Se {a} kg de maçã custam R$ {unit * a}, quanto custam {b} kg?",
        f"R$ {correto}",
        erradas,
    )


def _g_graficos_tabelas(m: int) -> tuple[str, str, list[str]]:
    vals = [random.randint(10, 90) * m for _ in range(4)]
    correto = sum(vals)
    erradas = _erros_num(correto)
    return (
        f"Uma tabela registra as vendas diárias: {vals}. Qual foi o total?",
        str(correto),
        erradas,
    )


def _g_medidas(m: int) -> tuple[str, str, list[str]]:
    vals = [random.randint(2, 18) * m for _ in range(5)]
    vals[-1] += (5 - sum(vals) % 5) % 5
    correto = sum(vals) // 5
    erradas = [str(correto + d) for d in [1, -1, 2] if correto + d != correto][:3]
    return f"Qual é a média aritmética dos valores {vals}?", str(correto), erradas


def _g_funcao_1_grau(m: int) -> tuple[str, str, list[str]]:
    a, b = random.randint(2, 5), random.randint(1, 9)
    x0 = random.randint(1, 5) * m
    correto = a * x0 + b
    erradas = [str(a * x0), str(a * x0 - b), str(correto + a)]
    return f"Dada f(x) = {a}x + {b}, qual o valor de f({x0})?", str(correto), erradas


def _g_funcao_2_grau(m: int) -> tuple[str, str, list[str]]:
    r1, r2 = random.randint(1, 4 + m), random.randint(1, 4 + m)
    correto = f"{min(r1, r2)} e {max(r1, r2)}"
    s, p = r1 + r2, r1 * r2
    erradas = [
        f"{-max(r1,r2)} e {-min(r1,r2)}",
        f"{s} e {p}",
        f"{min(r1,r2)-1} e {max(r1,r2)+1}",
    ]
    return (
        f"Quais as raízes de x² - {s}x + {p} = 0?",
        correto,
        erradas,
    )


def _g_padroes_graficos(m: int) -> tuple[str, str, list[str]]:
    a0, r = random.randint(1, 9), random.randint(2, 5)
    n = random.randint(4, 6) * m
    correto = a0 + (n - 1) * r
    erradas = [str(a0 + n * r), str(a0 + (n - 2) * r), str(a0 * r + n)]
    return (
        f"Uma sequência começa em {a0} e cresce de {r} em {r}. Qual é o {n}º termo?",
        str(correto),
        erradas,
    )


def _g_poligonos(m: int) -> tuple[str, str, list[str]]:
    n = random.randint(3, 6 + m)
    correto = (n - 2) * 180
    erradas = [f"{n * 180}°", f"{(n - 1) * 180}°", f"{n * 90}°"]
    return (
        f"Qual a soma dos ângulos internos de um polígono de {n} lados?",
        f"{correto}°",
        erradas,
    )


def _g_circunferencia(m: int) -> tuple[str, str, list[str]]:
    r = random.randint(2, 6) * m
    correto = f"{2 * 3.14 * r:.2f}".replace(".", ",")
    erradas = [
        f"{3.14 * r:.2f}".replace(".", ","),
        f"{3.14 * r * r:.2f}".replace(".", ","),
        f"{4 * 3.14 * r:.2f}".replace(".", ","),
    ]
    return (
        f"Qual o comprimento de uma circunferência de raio {r}? (use π = 3,14)",
        correto,
        erradas,
    )


def _g_areas_plana(m: int) -> tuple[str, str, list[str]]:
    b, h = random.randint(3, 9) * m, random.randint(2, 8) * m
    correto = b * h
    erradas = [str(b * h // 2), str(2 * (b + h)), str(b + h)]
    return f"Qual a área de um retângulo de base {b} e altura {h}?", str(correto), erradas


def _g_geometria_posicao(m: int) -> tuple[str, str, list[str]]:
    dx, dy, dist = random.choice(
        [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15)]
    )
    x1, y1 = random.randint(0, 3), random.randint(0, 3)
    erradas = [str(dx + dy), str(dist + 2), str(dist - 1)]
    return (
        f"Qual a distância entre P({x1}, {y1}) e Q({x1 + dx}, {y1 + dy})?",
        str(dist),
        erradas,
    )


def _g_volumes(m: int) -> tuple[str, str, list[str]]:
    a = random.randint(2, 6) * m
    b, c = random.randint(2, 5), random.randint(2, 5)
    correto = a * b * c
    erradas = [str(a + b + c), str(2 * (a * b + b * c + a * c)), str(a * b)]
    return f"Qual o volume de um paralelepípedo {a} × {b} × {c}?", str(correto), erradas


# Conceito -> gerador parametrizado de (enunciado, gabarito).
GERADORES = {
    "Matematica_Basica": _g_matematica_basica,
    "Regra_Tres": _g_regra_tres,
    "Graficos_Tabelas": _g_graficos_tabelas,
    "Medidas_Tendencia_Central": _g_medidas,
    "Funcao_1_Grau": _g_funcao_1_grau,
    "Funcao_2_Grau": _g_funcao_2_grau,
    "Padroes_Graficos": _g_padroes_graficos,
    "Poligonos_Regulares": _g_poligonos,
    "Circunferencia_Circulo": _g_circunferencia,
    "Formulas_Areas_Plana": _g_areas_plana,
    "Geometria_Posicao": _g_geometria_posicao,
    "Volumes_Areas_Espacial": _g_volumes,
}


def _gerar_banco_questoes(conceitos: dict[str, Conceito]) -> list[Questao]:
    """
    Gera o banco em grade: para cada conceito, QUESTOES_POR_CELULA itens em cada
    um dos 3 níveis de dificuldade, com a `dificuldade` ancorada na base do
    conceito mais o deslocamento do nível (com leve ruído para variedade).
    Cada questão agora é objetiva: gabarito = letra ("A"–"D"), alternativas em JSON.
    """
    questoes: list[Questao] = []
    for nome, base in DIFICULDADE_BASE.items():
        gerar = GERADORES[nome]
        for _nivel, (offset, magnitude) in NIVEIS.items():
            dif_celula = min(0.95, max(0.05, base + offset))
            for _ in range(QUESTOES_POR_CELULA):
                enunciado, gabarito_val, erradas = gerar(magnitude)
                alternativas, gabarito_letra = _montar_alternativas(gabarito_val, erradas)
                dificuldade = round(
                    min(0.98, max(0.02, dif_celula + random.uniform(-0.02, 0.02))),
                    3,
                )
                questoes.append(
                    Questao(
                        conceito_id=conceitos[nome].id,
                        dificuldade=dificuldade,
                        enunciado=enunciado,
                        gabarito=gabarito_letra,
                        alternativas=json.dumps(alternativas, ensure_ascii=False),
                    )
                )
    return questoes


# ---------------------------------------------------------------------------- #
# Autenticação — hash de senha (PBKDF2-HMAC-SHA256 salgado, stdlib)
# ---------------------------------------------------------------------------- #
_PBKDF2_ITERACOES = 200_000


def gerar_hash_senha(senha: str) -> str:
    """Gera o hash salgado: 'pbkdf2$<iterações>$<salt_hex>$<hash_hex>'."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256", senha.encode("utf-8"), salt, _PBKDF2_ITERACOES
    )
    return f"pbkdf2${_PBKDF2_ITERACOES}${salt.hex()}${dk.hex()}"


def verificar_senha(senha: str, senha_hash: str) -> bool:
    """Confere a senha contra o hash (comparação em tempo constante)."""
    try:
        algo, iteracoes, salt_hex, hash_hex = senha_hash.split("$")
        if algo != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", senha.encode("utf-8"), bytes.fromhex(salt_hex), int(iteracoes)
        )
        return secrets.compare_digest(dk.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


# Usuários de demonstração criados no seed (username, senha, nome, estudante_id).
# 'aluno' usa o perfil seedado (estudante 1); os demais começam "do zero".
USUARIOS_DEMO = [
    ("aluno", "enem2024", "Ana Beatriz Silva", 1),
    ("demo", "demo", "Aluno Demo", 2),
    ("admin", "admin", "Administrador", 3),
]


def _calcular_recompensa(resultado_real: int, prob_esperada: float) -> float:
    """Recompensa dinâmica do RL: R_t = y - ŷ.

    Recompensa positiva quando o aluno supera a expectativa (acerto inesperado)
    e negativa quando falha algo "fácil" — incentivando o Tutor a manter o aluno
    na Zona de Desenvolvimento Proximal (nem trivial, nem impossível).
    """
    return float(resultado_real) - float(prob_esperada)


def criar_banco_e_popular(reset: bool = True) -> None:
    """
    Cria o schema e insere dados sintéticos de desenvolvimento.

    Etapas:
      1. (Opcional) Recria o arquivo SQLite do zero.
      2. Insere o Grafo de Conhecimento (nós + arestas de pré-requisito).
      3. Insere o banco de questões em grade (conceito × dificuldade).
      4. Insere o estado inicial de proficiência do aluno de teste.
      5. Simula 2 experiências na tabela de Experience Replay (Interacao).
    """
    # Semente fixa: banco de questões reprodutível entre execuções.
    random.seed(42)

    # 1. Reset do banco em DEV para garantir idempotência.
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    engine = create_engine(DB_URL, echo=False, future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        conceitos: dict[str, Conceito] = {}
        for nome in GRAFO_CONHECIMENTO:
            c = Conceito(nome=nome)
            conceitos[nome] = c
            session.add(c)
        session.flush()  # garante IDs antes de criar arestas.

        # ARESTAS DO DAG: liga cada conceito aos seus pré-requisitos.
        # (pre_requisito_id -> conceito_id) = "estude o pré-req antes".
        for nome, pre_reqs in GRAFO_CONHECIMENTO.items():
            for pr in pre_reqs:
                session.add(
                    ConceitoPreRequisito(
                        conceito_id=conceitos[nome].id,
                        pre_requisito_id=conceitos[pr].id,
                    )
                )
        session.flush()

        # Banco em grade (conceito × dificuldade) com questões parametrizadas.
        questoes = _gerar_banco_questoes(conceitos)
        session.add_all(questoes)
        session.flush()

        # Atalho: primeira questão de um dado conceito (para os exemplos abaixo).
        def _questao_de(nome_conceito: str) -> Questao:
            cid = conceitos[nome_conceito].id
            return next(q for q in questoes if q.conceito_id == cid)

        proficiencias_iniciais = {
            "Matematica_Basica": 0.85,
            "Regra_Tres": 0.60,
            "Graficos_Tabelas": 0.55,
            "Medidas_Tendencia_Central": 0.30,
            "Funcao_1_Grau": 0.50,
            "Funcao_2_Grau": 0.20,
            "Padroes_Graficos": 0.25,
            "Poligonos_Regulares": 0.40,
            "Circunferencia_Circulo": 0.45,
            "Formulas_Areas_Plana": 0.20,
            "Geometria_Posicao": 0.10,
            "Volumes_Areas_Espacial": 0.05,
        }
        # Crença inicial INFORMADA pela proficiência (como um "nivelamento"):
        # contagens pseudo-Beta com N efetivo modesto, então a estimativa começa
        # coerente com o perfil do aluno, mas ainda com alguma incerteza.
        # (Não afeta o treino, que reseta a crença ao prior (1,1) a cada episódio.)
        N_PSEUDO = 4.0
        for nome, prof in proficiencias_iniciais.items():
            session.add(
                EstadoAluno(
                    estudante_id=ESTUDANTE_TESTE_ID,
                    conceito_id=conceitos[nome].id,
                    proficiencia=prof,
                    alpha=1.0 + N_PSEUDO * prof,
                    beta=1.0 + N_PSEUDO * (1.0 - prof),
                )
            )
        session.flush()

        # Experiência 1 — Ação "Reforçar" em Função do 1º grau.
        #   ŷ = 0.50 (proficiência atual) ; aluno ACERTOU (y = 1).
        #   R_t = 1 - 0.50 = +0.50 (superou a expectativa -> bom sinal).
        prob1, y1 = 0.50, 1
        session.add(
            Interacao(
                estudante_id=ESTUDANTE_TESTE_ID,
                questao_id=_questao_de("Funcao_1_Grau").id,
                acao_rl="Reforçar",
                prob_esperada=prob1,
                resultado_real=y1,
                recompensa=_calcular_recompensa(y1, prob1),
            )
        )

        # Experiência 2 — Ação "Avançar" para Função do 2º grau (filho no DAG).
        #   ŷ = 0.20 (conceito ainda fraco) ; aluno ERROU (y = 0).
        #   R_t = 0 - 0.20 = -0.20 (avanço prematuro -> penalização leve).
        prob2, y2 = 0.20, 0
        session.add(
            Interacao(
                estudante_id=ESTUDANTE_TESTE_ID,
                questao_id=_questao_de("Funcao_2_Grau").id,
                acao_rl="Avançar",
                prob_esperada=prob2,
                resultado_real=y2,
                recompensa=_calcular_recompensa(y2, prob2),
            )
        )

        # Usuários de demonstração (senha em hash). Cada um com seu estudante_id.
        for username, senha, nome_user, est_id in USUARIOS_DEMO:
            session.add(
                Usuario(
                    username=username,
                    senha_hash=gerar_hash_senha(senha),
                    nome=nome_user,
                    estudante_id=est_id,
                )
            )

        session.commit()

        # Resumo informativo do seed.
        total_conceitos = len(session.scalars(select(Conceito)).all())
        print("Banco criado e populado com sucesso.")
        print(f"  - Conceitos (nós do DAG): {total_conceitos}")
        print(f"  - Usuários demo: {len(USUARIOS_DEMO)} ({', '.join(u[0] for u in USUARIOS_DEMO)})")
        print(
            f"  - Questões: {len(questoes)} "
            f"({len(DIFICULDADE_BASE)} conceitos × {len(NIVEIS)} níveis × "
            f"{QUESTOES_POR_CELULA})"
        )
        print(f"  - Estado inicial do aluno {ESTUDANTE_TESTE_ID}: {len(proficiencias_iniciais)} conceitos")
        print("  - Interações (Experience Replay): 2")
        print(f"  - Arquivo SQLite: {DB_PATH}")


if __name__ == "__main__":
    criar_banco_e_popular()
