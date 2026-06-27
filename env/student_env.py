"""
student_env.py
==============
Ambiente de simulação (estilo OpenAI Gym) do Sistema Tutor Inteligente (ITS).

A classe `StudentEnvironment` é o "mundo" com o qual o agente DQN interage: ela
implementa a dinâmica do aluno simulado (Modelo do Estudante) e a economia de
recompensas. NÃO contém a rede neural.

Esta versão corrige os problemas de modelagem do MDP que impediam o aprendizado:

1. RECOMPENSA ORIENTADA À META (não mais R_t = y - ŷ, que tem média ZERO por
   construção). A recompensa agora premia o GANHO de aprendizado em conceitos
   profundos, o DOMÍNIO de conceitos novos e o ALCANCE do nível avançado:

       R_t = W_PROGRESSO * Δprof * peso_profundidade   (denso; evita travar no fácil)
           + bônus de domínio (1ª vez que um conceito cruza o limiar)
           - W_ZDP * |ŷ - 0.5|                         (mantém na Zona de Des. Proximal)
           - W_PASSO                                    (eficiência)
           + W_OBJETIVO  (terminal, ao dominar o conceito-alvo avançado)

2. ESTADO MARKOVIANO: o vetor de estado concatena as proficiências COM um
   one-hot do conceito atual, para que "Avançar/Reforçar/Remediar" sejam ações
   bem definidas (o agente "enxerga" onde o aluno está no DAG).

3. A AÇÃO CONTROLA O DESAFIO: a dificuldade vem do conceito-alvo escolhido pela
   ação (mais profundo = mais difícil), e não é mais "casada" com a proficiência
   (o que neutralizava o efeito da ação, fixando ŷ ≈ 0.5).

4. TREINO EPISÓDICO: a proficiência simulada vive EM MEMÓRIA e `reset()` a
   restaura ao estado inicial. Nada de mutar/persistir a proficiência no banco a
   cada passo (que tornava o treino não-estacionário e não-reprodutível).

5. ACOPLAMENTO DE PRÉ-REQUISITOS: o sucesso e a velocidade de aprendizado num
   conceito dependem do domínio dos seus pré-requisitos no DAG — é isso que dá
   sentido pedagógico a "Remediar" e cria a dinâmica de currículo.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload

from data.database_setup import Conceito, EstadoAluno, Interacao, Questao

# Conjunto canônico de ações pedagógicas (espaço de ações A do MDP).
ACOES_VALIDAS = ("Avançar", "Reforçar", "Remediar")

# --- Dinâmica de simulação do aluno -----------------------------------------
SENSIBILIDADE_LOGISTICA = 5.0   # 'k' da logística: inclinação de ŷ vs. folga.
TAXA_APRENDIZADO = 0.12         # Ganho-base de proficiência ao acertar.
TAXA_ESQUECIMENTO = 0.05        # Queda de proficiência ao errar.
ALPHA_PRE_REQUISITO = 0.6       # Peso do domínio dos pré-requisitos em ŷ.
FADIGA_POR_PASSO = 0.004        # Fadiga por interação (~250 passos por episódio).
FADIGA_MAXIMA = 1.0             # Limite de fadiga que encerra o episódio.
PROFICIENCIA_MIN = 0.0
PROFICIENCIA_MAX = 1.0

# --- Função de recompensa ----------------------------------------------------
LIMIAR_DOMINIO = 0.8            # τ: proficiência a partir da qual há "domínio".
FRACAO_DOMINIO_ALVO = 0.70     # "Nível avançado": fração do currículo dominada.
W_PROGRESSO = 4.0             # Peso do ganho denso de proficiência (trilha p/ o alvo).
W_DOMINIO = 1.5               # Peso do bônus de DOMÍNIO (1ª vez que cruza o limiar).
W_ZDP = 0.20                  # Penalidade por sair da Zona de Des. Proximal.
W_PASSO = 0.01                # Custo por passo (incentiva eficiência).
W_OBJETIVO = 15.0             # Bônus terminal por atingir o nível avançado.

# Dificuldade-base por profundidade no DAG (conceito sem questões cadastradas).
DIFICULDADE_RASA = 0.2
DIFICULDADE_PROFUNDA = 0.9


class StudentEnvironment:
    """
    Simulador do aluno (Modelo do Estudante) para treinar o agente DQN.

    API no estilo OpenAI Gym: `reset()` inicia um episódio e `step(acao)` avança
    um passo retornando (estado, recompensa, done, info).
    """

    def __init__(
        self,
        db_url: str,
        estudante_id: int,
        log_interacoes: bool = False,
    ) -> None:
        """
        Parâmetros
        ----------
        db_url : str
            String de conexão do banco (ex.: 'sqlite:///data/enem_tutor.db').
        estudante_id : int
            ID do estudante cujo estado inicial é lido de `EstadoAluno`.
        log_interacoes : bool
            Se True, registra cada interação na tabela `Interacao` (append-only,
            apenas para auditoria/análise). Desligado por padrão no treino, pois
            é desnecessário (o Experience Replay real vive no agente) e custa I/O.
        """
        self.db_url = db_url
        self.estudante_id = estudante_id
        self.log_interacoes = log_interacoes

        self.engine = create_engine(self.db_url, future=True)
        self.session: Session = Session(self.engine)

        # Carrega todos os conceitos com seus relacionamentos do DAG de uma vez.
        conceitos = list(
            self.session.scalars(
                select(Conceito)
                .order_by(Conceito.id)
                .options(
                    selectinload(Conceito.dependentes),
                    selectinload(Conceito.pre_requisitos),
                )
            ).all()
        )
        if not conceitos:
            raise RuntimeError(
                "Nenhum conceito encontrado no banco. Rode "
                "'data.database_setup.criar_banco_e_popular()' antes de treinar."
            )

        # Ordem canônica: posição i do vetor de estado <-> mesmo conceito sempre.
        self.conceito_ids: list[int] = [c.id for c in conceitos]
        self.indice_por_conceito: dict[int, int] = {
            cid: i for i, cid in enumerate(self.conceito_ids)
        }
        self.n_conceitos = len(self.conceito_ids)

        # Estado = [proficiências (N)] ++ [one-hot do conceito atual (N)].
        self.dim_estado: int = 2 * self.n_conceitos

        # Mapas do DAG (por id de conceito).
        self.dependentes_ids: dict[int, list[int]] = {
            c.id: [d.id for d in c.dependentes] for c in conceitos
        }
        self.pre_requisitos_ids: dict[int, list[int]] = {
            c.id: [p.id for p in c.pre_requisitos] for c in conceitos
        }

        # Profundidade no DAG (caminho mais longo a partir das raízes).
        self.profundidade: dict[int, int] = self._calcular_profundidades()
        self.profundidade_maxima = max(self.profundidade.values()) or 1

        # Dificuldade intrínseca por conceito (média das questões ou por nível).
        self.dificuldade_conceito: dict[int, float] = self._calcular_dificuldades()

        # Conceito(s)-alvo = os mais profundos do DAG (nível "avançado").
        self.conceitos_alvo: set[int] = {
            cid
            for cid, prof in self.profundidade.items()
            if prof == self.profundidade_maxima
        }

        # Proficiência INICIAL do aluno (lida 1x do banco, congelada em memória).
        self.proficiencia_inicial: dict[int, float] = self._ler_proficiencia_inicial()

        # Estado de simulação vivo (reescrito a cada reset).
        self.proficiencia: dict[int, float] = dict(self.proficiencia_inicial)
        self.conceito_atual_id: int = self.conceito_ids[0]
        self.fadiga: float = 0.0
        self.ja_dominados: set[int] = set()

    # ------------------------------------------------------------------ #
    # Pré-cálculos do DAG (executados uma vez no __init__)
    # ------------------------------------------------------------------ #
    def _calcular_profundidades(self) -> dict[int, int]:
        """Profundidade = comprimento do maior caminho desde uma raiz do DAG."""
        memo: dict[int, int] = {}

        def profundidade(cid: int) -> int:
            if cid in memo:
                return memo[cid]
            pres = self.pre_requisitos_ids.get(cid, [])
            memo[cid] = 0 if not pres else 1 + max(profundidade(p) for p in pres)
            return memo[cid]

        return {cid: profundidade(cid) for cid in self.conceito_ids}

    def _calcular_dificuldades(self) -> dict[int, float]:
        """
        Dificuldade intrínseca por conceito.

        Usa a média das dificuldades das questões cadastradas; se o conceito não
        tiver questões, deriva a dificuldade da profundidade no DAG (mais
        profundo => mais difícil), garantindo um simulador bem definido para
        TODOS os conceitos (o seed tem poucas questões).
        """
        questoes = list(self.session.scalars(select(Questao)).all())
        soma: dict[int, float] = {}
        cont: dict[int, int] = {}
        for q in questoes:
            soma[q.conceito_id] = soma.get(q.conceito_id, 0.0) + float(q.dificuldade)
            cont[q.conceito_id] = cont.get(q.conceito_id, 0) + 1

        dificuldades: dict[int, float] = {}
        for cid in self.conceito_ids:
            if cont.get(cid):
                dificuldades[cid] = soma[cid] / cont[cid]
            else:
                escala = self.profundidade[cid] / self.profundidade_maxima
                dificuldades[cid] = (
                    DIFICULDADE_RASA
                    + (DIFICULDADE_PROFUNDA - DIFICULDADE_RASA) * escala
                )
        return dificuldades

    def _ler_proficiencia_inicial(self) -> dict[int, float]:
        """Lê a proficiência inicial do aluno (0.0 para conceitos sem registro)."""
        prof = {cid: 0.0 for cid in self.conceito_ids}
        registros = self.session.scalars(
            select(EstadoAluno).where(
                EstadoAluno.estudante_id == self.estudante_id
            )
        ).all()
        for reg in registros:
            if reg.conceito_id in prof:
                prof[reg.conceito_id] = float(reg.proficiencia)
        return prof

    # ------------------------------------------------------------------ #
    # Estado (observação)
    # ------------------------------------------------------------------ #
    def _get_state(self) -> np.ndarray:
        """
        Monta o vetor de Estado: proficiências ++ one-hot do conceito atual.

        O one-hot torna o problema Markoviano: a semântica de "Avançar/Reforçar/
        Remediar" depende de ONDE o aluno está, e agora o agente enxerga isso.
        """
        estado = np.zeros(self.dim_estado, dtype=np.float32)
        for cid, idx in self.indice_por_conceito.items():
            estado[idx] = self.proficiencia[cid]
        idx_atual = self.indice_por_conceito[self.conceito_atual_id]
        estado[self.n_conceitos + idx_atual] = 1.0
        return estado

    def reset(self) -> np.ndarray:
        """
        Reinicia o episódio: restaura a proficiência inicial e zera o contexto.

        Restaurar a proficiência é o que torna o treino EPISÓDICO e reprodutível:
        cada episódio parte do mesmo aluno, em vez de continuar de um aluno já
        saturado por episódios anteriores.
        """
        self.proficiencia = dict(self.proficiencia_inicial)
        self.conceito_atual_id = self.conceito_ids[0]
        self.fadiga = 0.0
        # Conceitos já dominados no estado inicial não rendem bônus de domínio.
        self.ja_dominados = {
            cid
            for cid, prof in self.proficiencia.items()
            if prof >= LIMIAR_DOMINIO
        }
        return self._get_state()

    # ------------------------------------------------------------------ #
    # Tradução Ação -> conceito-alvo (via DAG)
    # ------------------------------------------------------------------ #
    def _selecionar_conceito_alvo(self, acao: str) -> int:
        """
        Traduz a Ação Pedagógica abstrata em um conceito-alvo concreto via DAG.

        - "Avançar"  -> o dependente (sucessor) MENOS dominado: a próxima
                        fronteira de aprendizado.
        - "Reforçar" -> o próprio conceito atual.
        - "Remediar" -> o pré-requisito (antecessor) MAIS fraco: a revisão de
                        maior retorno pedagógico.

        A seleção é DETERMINÍSTICA (menor proficiência, desempate por id) para
        que a navegação no DAG seja dirigível: o agente decide QUANDO avançar,
        reforçar ou remediar, mas cada ação leva a um destino bem definido — sem
        isso, "Avançar" cairia num filho aleatório e a meta (folha profunda)
        ficaria praticamente inalcançável.

        Sem candidatos (folha/raiz do DAG) -> permanece no conceito atual.
        """
        if acao == "Avançar":
            candidatos = self.dependentes_ids.get(self.conceito_atual_id, [])
        elif acao == "Remediar":
            candidatos = self.pre_requisitos_ids.get(self.conceito_atual_id, [])
        else:  # "Reforçar".
            candidatos = [self.conceito_atual_id]
        if not candidatos:
            return self.conceito_atual_id
        # Menor proficiência primeiro; desempate determinístico pelo id.
        return min(candidatos, key=lambda cid: (self.proficiencia[cid], cid))

    def _selecionar_questao(self, conceito_id: int) -> Questao | None:
        """Sorteia uma questão do conceito-alvo (para `info`/logging), se houver."""
        questoes = self.session.scalars(
            select(Questao).where(Questao.conceito_id == conceito_id)
        ).all()
        return random.choice(questoes) if questoes else None

    # ------------------------------------------------------------------ #
    # Dinâmica do aluno
    # ------------------------------------------------------------------ #
    def _dominio_pre_requisitos(self, conceito_id: int) -> float:
        """Domínio médio dos pré-requisitos do conceito (1.0 se não houver)."""
        pres = self.pre_requisitos_ids.get(conceito_id, [])
        if not pres:
            return 1.0
        return sum(self.proficiencia[p] for p in pres) / len(pres)

    def _probabilidade_acerto(self, conceito_id: int) -> float:
        """
        Estima ŷ (probabilidade de acerto) via logística estilo TRI.

        A folga combina (proficiência - dificuldade) com o domínio dos
        pré-requisitos: pré-requisitos fracos derrubam ŷ — é isso que dá valor a
        "Remediar" antes de "Avançar".
        """
        prof = self.proficiencia[conceito_id]
        dificuldade = self.dificuldade_conceito[conceito_id]
        pre = self._dominio_pre_requisitos(conceito_id)
        folga = (prof - dificuldade) + ALPHA_PRE_REQUISITO * (pre - 0.5)
        return 1.0 / (1.0 + math.exp(-SENSIBILIDADE_LOGISTICA * folga))

    def _atualizar_proficiencia(self, conceito_id: int, acertou: bool) -> float:
        """
        Atualiza (em memória) a proficiência após responder e retorna o Δ.

        O ganho ao acertar é escalado pelo domínio dos pré-requisitos: aprende-se
        mais rápido um conceito cujos pré-requisitos já estão sólidos.
        """
        antes = self.proficiencia[conceito_id]
        if acertou:
            pre = self._dominio_pre_requisitos(conceito_id)
            delta = TAXA_APRENDIZADO * (0.5 + 0.5 * pre)
        else:
            delta = -TAXA_ESQUECIMENTO
        depois = min(PROFICIENCIA_MAX, max(PROFICIENCIA_MIN, antes + delta))
        self.proficiencia[conceito_id] = depois
        return depois - antes

    # ------------------------------------------------------------------ #
    # Passo do ambiente
    # ------------------------------------------------------------------ #
    def step(self, acao: str) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Executa um passo do ambiente dada uma Ação Pedagógica.

        Retorna (novo_estado, recompensa, done, info).
        """
        if acao not in ACOES_VALIDAS:
            raise ValueError(
                f"Ação inválida: {acao!r}. Esperado um de {ACOES_VALIDAS}."
            )

        # 1. Ação -> conceito-alvo (a dificuldade vem do conceito, não da prof.).
        conceito_alvo_id = self._selecionar_conceito_alvo(acao)

        # 2. ŷ e resultado simulado do aluno (acerta com probabilidade ŷ).
        y_hat = self._probabilidade_acerto(conceito_alvo_id)
        y = 1 if random.random() < y_hat else 0

        # 3. Transição: atualiza a proficiência e computa o ganho.
        ganho = self._atualizar_proficiencia(conceito_alvo_id, acertou=bool(y))

        # 4. Recompensa orientada à meta (ver docstring do módulo).
        peso_prof = self.profundidade[conceito_alvo_id] / self.profundidade_maxima
        recompensa = W_PROGRESSO * ganho * peso_prof
        recompensa -= W_ZDP * abs(y_hat - 0.5)
        recompensa -= W_PASSO

        #    Bônus de domínio (apenas na 1ª vez que o conceito cruza o limiar).
        if (
            self.proficiencia[conceito_alvo_id] >= LIMIAR_DOMINIO
            and conceito_alvo_id not in self.ja_dominados
        ):
            self.ja_dominados.add(conceito_alvo_id)
            recompensa += W_DOMINIO * (0.3 + 0.7 * peso_prof)

        # 5. Reposiciona o aluno e atualiza a fadiga.
        self.conceito_atual_id = conceito_alvo_id
        self.fadiga += FADIGA_POR_PASSO

        # 6. Condições de término: nível avançado atingido (bônus) ou fadiga.
        #    "Nível avançado" = dominar >= FRACAO_DOMINIO_ALVO do currículo. A
        #    recompensa pondera a profundidade (peso_prof), então o agente é
        #    puxado a dominar conceitos cada vez mais avançados — e não a
        #    acumular apenas os fáceis — para chegar a essa fração.
        objetivo_atingido = (
            len(self.ja_dominados) >= FRACAO_DOMINIO_ALVO * self.n_conceitos
        )
        if objetivo_atingido:
            recompensa += W_OBJETIVO
        done = objetivo_atingido or self.fadiga >= FADIGA_MAXIMA

        # 7. (Opcional) auditoria append-only — não afeta o estado da simulação.
        questao = self._selecionar_questao(conceito_alvo_id)
        if self.log_interacoes and questao is not None:
            self.session.add(
                Interacao(
                    estudante_id=self.estudante_id,
                    questao_id=questao.id,
                    acao_rl=acao,
                    prob_esperada=float(y_hat),
                    resultado_real=int(y),
                    recompensa=float(recompensa),
                )
            )
            self.session.commit()

        info = {
            "acao": acao,
            "conceito_alvo_id": conceito_alvo_id,
            "questao_id": questao.id if questao else None,
            "dificuldade": self.dificuldade_conceito[conceito_alvo_id],
            "y_hat": y_hat,
            "y": y,
            "ganho": ganho,
            "fadiga": self.fadiga,
            "objetivo_atingido": objetivo_atingido,
            "n_dominados": len(self.ja_dominados),
        }
        return self._get_state(), float(recompensa), done, info

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Fecha a sessão do SQLAlchemy e libera o engine."""
        try:
            self.session.close()
        finally:
            self.engine.dispose()

    def __enter__(self) -> "StudentEnvironment":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
