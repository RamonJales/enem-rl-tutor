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

import os

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
    gabarito: Mapped[str] = mapped_column(String(10), nullable=False)

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
    """
    __tablename__ = "estado_aluno"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    estudante_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    conceito_id: Mapped[int] = mapped_column(
        ForeignKey("conceito.id"), nullable=False
    )
    proficiencia: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    conceito: Mapped["Conceito"] = relationship("Conceito")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EstadoAluno estudante={self.estudante_id} "
            f"conceito_id={self.conceito_id} prof={self.proficiencia:.2f}>"
        )


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
      3. Insere 5 questões com dificuldades variadas.
      4. Insere o estado inicial de proficiência do aluno de teste.
      5. Simula 2 experiências na tabela de Experience Replay (Interacao).
    """
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

        questoes = [
            Questao(
                conceito_id=conceitos["Matematica_Basica"].id,
                dificuldade=0.2,
                enunciado="Quanto é 15% de 200?",
                gabarito="30",
            ),
            Questao(
                conceito_id=conceitos["Regra_Tres"].id,
                dificuldade=0.4,
                enunciado="Se 3 operários fazem um muro em 6 dias, em quantos dias 6 operários fazem o mesmo muro?",
                gabarito="3",
            ),
            Questao(
                conceito_id=conceitos["Funcao_1_Grau"].id,
                dificuldade=0.5,
                enunciado="Dada f(x) = 2x + 1, qual o valor de f(3)?",
                gabarito="7",
            ),
            Questao(
                conceito_id=conceitos["Funcao_2_Grau"].id,
                dificuldade=0.7,
                enunciado="Quais as raízes de x^2 - 5x + 6 = 0?",
                gabarito="2 e 3",
            ),
            Questao(
                conceito_id=conceitos["Formulas_Areas_Plana"].id,
                dificuldade=0.8,
                enunciado="Qual a área de um círculo de raio 5? (use pi = 3,14)",
                gabarito="78,5",
            ),
        ]
        session.add_all(questoes)
        session.flush()

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
        for nome, prof in proficiencias_iniciais.items():
            session.add(
                EstadoAluno(
                    estudante_id=ESTUDANTE_TESTE_ID,
                    conceito_id=conceitos[nome].id,
                    proficiencia=prof,
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
                questao_id=questoes[2].id,  # Funcao_1_Grau
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
                questao_id=questoes[3].id,  # Funcao_2_Grau
                acao_rl="Avançar",
                prob_esperada=prob2,
                resultado_real=y2,
                recompensa=_calcular_recompensa(y2, prob2),
            )
        )

        session.commit()

        # Resumo informativo do seed.
        total_conceitos = len(session.scalars(select(Conceito)).all())
        print("Banco criado e populado com sucesso.")
        print(f"  - Conceitos (nós do DAG): {total_conceitos}")
        print(f"  - Questões: {len(questoes)}")
        print(f"  - Estado inicial do aluno {ESTUDANTE_TESTE_ID}: {len(proficiencias_iniciais)} conceitos")
        print("  - Interações (Experience Replay): 2")
        print(f"  - Arquivo SQLite: {DB_PATH}")


if __name__ == "__main__":
    criar_banco_e_popular()
