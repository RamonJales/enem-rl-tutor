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
           + W_SONDA * max(0, Δerro_crença)             (sondagem: ganho de informação)
           - W_TEDIO * max(0, ŷ - LIMIAR_TEDIO)         (questão fácil demais)
           - W_FRUST * max(0, LIMIAR_FRUST - ŷ) [se errou] (difícil demais)
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

6. MODELO DE CRENÇA BAYESIANO (BKT): o agente NÃO observa a proficiência
   verdadeira do aluno (que determina y). O sistema mantém uma crença Beta(α, β)
   sobre P(acerto) por conceito (de onde sai ŷ), atualizada por Bayes a cada
   resposta. O agente observa a MÉDIA e o DESVIO (incerteza) da crença — uma
   estatística suficiente que transforma o POMDP num MDP de estado-de-crença
   (mantendo a DQN feedforward adequada). Essa assimetria dá função à sondagem:
   testar um conceito incerto/subestimado reduz o erro de crença (ganho de
   informação). Sem ela, ŷ seria a própria verdade e "sondar" não teria sentido.
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

# --- Modelo de crença BAYESIANO (BKT): proficiência REAL oculta × CRENÇA ------
# O agente NÃO observa a proficiência verdadeira (que determina se o aluno
# acerta). Em vez disso, o sistema mantém uma CRENÇA Bayesiana por conceito —
# uma Beta(α, β) sobre P(acerto) — e o agente observa sua MÉDIA e seu DESVIO
# (incerteza). Expor a incerteza torna a observação uma ESTATÍSTICA SUFICIENTE
# da crença (POMDP -> MDP de estado-de-crença), mantendo a DQN feedforward
# adequada e dando função à sondagem (reduzir o erro/incerteza da crença).
ALFA0 = 1.0                    # Prior Beta (α): pseudo-acertos. (1,1) = uniforme.
BETA0 = 1.0                    # Prior Beta (β): pseudo-erros.
LAMBDA_ESQUEC = 0.95           # Esquecimento das contagens (rastreia aluno que evolui).
DESVIO_MAX = 0.2887            # Desvio da Beta(1,1) = sqrt(1/12); normaliza a incerteza.

# --- Função de recompensa ----------------------------------------------------
LIMIAR_DOMINIO = 0.8            # τ: proficiência a partir da qual há "domínio".
FRACAO_DOMINIO_ALVO = 0.70     # "Nível avançado": fração do currículo dominada.
W_PROGRESSO = 4.0             # Peso do ganho denso de proficiência (trilha p/ o alvo).
W_DOMINIO = 1.5               # Peso do bônus de DOMÍNIO (1ª vez que cruza o limiar).
W_PASSO = 0.01                # Custo por passo (incentiva eficiência).
W_OBJETIVO = 15.0             # Bônus terminal por atingir o nível avançado.

# --- Sondagem / ganho de informação (recompensa ASSIMÉTRICA) -----------------
# Substitui a antiga penalidade simétrica de ZDP (W_ZDP·|ŷ-0.5|), que punia
# qualquer desvio de 50% e, com isso, DESINCENTIVAVA sondar. Aqui o agente é
# premiado por "sondar" uma questão mais difícil quando a aposta paga (o aluno
# acerta algo improvável = salto de conhecimento / atalho eficiente) e só é
# punido nos extremos: questão fácil demais (tédio/desperdício) ou difícil
# demais que o aluno errou (frustração). Fiel ao PDF (+0.8 acerto difícil /
# -0.9 erro elementar) e ao princípio da Zona de Desenvolvimento Proximal.
W_SONDA = 4.0                 # Peso do GANHO DE INFORMAÇÃO (redução do erro de crença).
W_TEDIO = 1.0                 # Penalidade por questão fácil demais (ŷ alto).
LIMIAR_TEDIO = 0.85           # ŷ acima disso => desperdício de tempo (tédio).
W_FRUST = 1.5                 # Penalidade por errar questão muito acima do nível.
LIMIAR_FRUST = 0.20           # ŷ abaixo disso (e erro) => frustração.

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

        # Estado (observação do agente) = [proficiência ESTIMADA (N)]
        #   ++ [one-hot do conceito atual (N)] ++ [evidência/certeza (N)].
        self.dim_estado: int = 3 * self.n_conceitos

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

        # Proficiência REAL inicial do aluno (a "verdade" oculta), lida 1x do
        # banco e congelada. O agente NÃO a observa diretamente.
        self.proficiencia_inicial: dict[int, float] = self._ler_proficiencia_inicial()

        # Estado de simulação vivo (reescrito a cada reset):
        #   prof_real    -> verdade oculta que determina se o aluno acerta (y);
        #   alpha/beta   -> crença Bayesiana Beta(α, β) sobre P(acerto) por
        #                   conceito; o agente observa sua MÉDIA e seu DESVIO.
        self.prof_real: dict[int, float] = dict(self.proficiencia_inicial)
        self.alpha: dict[int, float] = {cid: ALFA0 for cid in self.conceito_ids}
        self.beta: dict[int, float] = {cid: BETA0 for cid in self.conceito_ids}
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
        Monta o vetor de Estado (o que o agente OBSERVA):
          [MÉDIA da crença] ++ [one-hot do conceito atual] ++ [INCERTEZA da crença].

        - Média + incerteza da Beta(α,β) formam uma ESTATÍSTICA SUFICIENTE da
          crença: o agente "sabe o que sabe E o quanto tem certeza", o que torna
          o POMDP um MDP de estado-de-crença (feedforward DQN volta a bastar).
        - O one-hot torna o problema Markoviano (Avançar/Reforçar/Remediar
          dependem de ONDE o aluno está).
        - Incerteza alta sinaliza conceitos que valem uma SONDAGEM.
        """
        estado = np.zeros(self.dim_estado, dtype=np.float32)
        for cid, idx in self.indice_por_conceito.items():
            estado[idx] = self._crenca_media(cid)
            estado[2 * self.n_conceitos + idx] = (
                self._crenca_desvio(cid) / DESVIO_MAX
            )
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
        self.prof_real = dict(self.proficiencia_inicial)
        self.alpha = {cid: ALFA0 for cid in self.conceito_ids}
        self.beta = {cid: BETA0 for cid in self.conceito_ids}
        self.conceito_atual_id = self.conceito_ids[0]
        self.fadiga = 0.0
        # Domínio é medido na proficiência REAL (aprendizado de verdade).
        # Conceitos já dominados no estado inicial não rendem bônus de domínio.
        self.ja_dominados = {
            cid
            for cid, prof in self.prof_real.items()
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
        # Menor MÉDIA de crença primeiro (o sistema navega pela crença);
        # desempate determinístico pelo id.
        return min(candidatos, key=lambda cid: (self._crenca_media(cid), cid))

    def _selecionar_questao(self, conceito_id: int) -> Questao | None:
        """Sorteia uma questão do conceito-alvo (para `info`/logging), se houver."""
        questoes = self.session.scalars(
            select(Questao).where(Questao.conceito_id == conceito_id)
        ).all()
        return random.choice(questoes) if questoes else None

    # ------------------------------------------------------------------ #
    # Dinâmica do aluno
    # ------------------------------------------------------------------ #
    def _dominio_pre_requisitos(
        self, conceito_id: int, prof: dict[int, float]
    ) -> float:
        """Domínio médio dos pré-requisitos sob um dado mapa (real ou estimado)."""
        pres = self.pre_requisitos_ids.get(conceito_id, [])
        if not pres:
            return 1.0
        return sum(prof[p] for p in pres) / len(pres)

    def _probabilidade_acerto(
        self, conceito_id: int, prof: dict[int, float]
    ) -> float:
        """
        Probabilidade de acerto via logística estilo TRI, sob um mapa `prof`.

        Usada com `prof_real` para gerar a probabilidade VERDADEIRA (o resultado
        y). O ŷ ESTIMADO do sistema NÃO vem daqui: é a média da crença Beta
        (`_crenca_media`), mantida por atualização Bayesiana (BKT).

        A folga combina (proficiência - dificuldade) com o domínio dos
        pré-requisitos: pré-requisitos fracos derrubam a probabilidade — é isso
        que dá valor a "Remediar" antes de "Avançar".
        """
        p = prof[conceito_id]
        dificuldade = self.dificuldade_conceito[conceito_id]
        pre = self._dominio_pre_requisitos(conceito_id, prof)
        folga = (p - dificuldade) + ALPHA_PRE_REQUISITO * (pre - 0.5)
        return 1.0 / (1.0 + math.exp(-SENSIBILIDADE_LOGISTICA * folga))

    def _atualizar_proficiencia(self, conceito_id: int, acertou: bool) -> float:
        """
        Atualiza a proficiência REAL (verdade oculta) após responder e retorna Δ.

        O ganho ao acertar é escalado pelo domínio REAL dos pré-requisitos:
        aprende-se mais rápido um conceito cujos pré-requisitos já estão sólidos.
        """
        antes = self.prof_real[conceito_id]
        if acertou:
            pre = self._dominio_pre_requisitos(conceito_id, self.prof_real)
            delta = TAXA_APRENDIZADO * (0.5 + 0.5 * pre)
        else:
            delta = -TAXA_ESQUECIMENTO
        depois = min(PROFICIENCIA_MAX, max(PROFICIENCIA_MIN, antes + delta))
        self.prof_real[conceito_id] = depois
        return depois - antes

    def _crenca_media(self, conceito_id: int) -> float:
        """Média da crença Beta(α, β) = α / (α + β) = P(acerto) estimada."""
        a, b = self.alpha[conceito_id], self.beta[conceito_id]
        return a / (a + b)

    def _crenca_desvio(self, conceito_id: int) -> float:
        """Desvio-padrão da Beta(α, β): incerteza da crença (alto => sondar)."""
        a, b = self.alpha[conceito_id], self.beta[conceito_id]
        n = a + b
        return math.sqrt(a * b / (n * n * (n + 1.0)))

    def _atualizar_crenca(self, conceito_id: int, y: int) -> None:
        """
        Atualização Bayesiana (BKT) da crença após observar o resultado y.

        Aplica um leve esquecimento das contagens (aproxima-as do prior) para
        que a crença RASTREIE um aluno que evolui, e então incorpora a evidência
        (acerto -> +α, erro -> +β). A média se move na direção da verdade e a
        incerteza (desvio) encolhe com a evidência acumulada.
        """
        self.alpha[conceito_id] = ALFA0 + LAMBDA_ESQUEC * (self.alpha[conceito_id] - ALFA0)
        self.beta[conceito_id] = BETA0 + LAMBDA_ESQUEC * (self.beta[conceito_id] - BETA0)
        if y:
            self.alpha[conceito_id] += 1.0
        else:
            self.beta[conceito_id] += 1.0

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

        # 1. Ação -> conceito-alvo (o sistema navega pela ESTIMATIVA).
        conceito_alvo_id = self._selecionar_conceito_alvo(acao)

        # 2. ŷ = MÉDIA da crença (o que o sistema acha) vs. probabilidade REAL
        #    (verdade oculta). O aluno acerta segundo a verdade.
        y_hat = self._crenca_media(conceito_alvo_id)
        p_real = self._probabilidade_acerto(conceito_alvo_id, self.prof_real)
        y = 1 if random.random() < p_real else 0

        # 3. Erro de crença ANTES da observação (verdade vs. crença).
        erro_antes = abs(p_real - y_hat)

        # 4. Transição: o aluno aprende (prof_real) e o sistema reavalia (Bayes).
        ganho = self._atualizar_proficiencia(conceito_alvo_id, acertou=bool(y))
        self._atualizar_crenca(conceito_alvo_id, y)

        # 5. Recompensa orientada à meta (ver docstring do módulo).
        peso_prof = self.profundidade[conceito_alvo_id] / self.profundidade_maxima
        recompensa = W_PROGRESSO * ganho * peso_prof
        recompensa -= W_PASSO

        # Sondagem = GANHO DE INFORMAÇÃO: quanto a observação aproximou a MÉDIA
        # da crença da verdade (|p_real - ŷ| caiu). Premia DESCOBRIR domínio
        # oculto e é AUTO-LIMITADO (o erro de crença total é finito -> não dá
        # para farmar, ao contrário de um bônus de surpresa por passo).
        # Tédio/frustração penalizam os extremos (fácil demais / difícil demais).
        erro_depois = abs(p_real - self._crenca_media(conceito_alvo_id))
        sondagem = W_SONDA * max(0.0, erro_antes - erro_depois)
        tedio = W_TEDIO * max(0.0, y_hat - LIMIAR_TEDIO)
        frustracao = W_FRUST * max(0.0, LIMIAR_FRUST - y_hat) if y == 0 else 0.0
        recompensa += sondagem - tedio - frustracao

        #    Bônus de domínio REAL (1ª vez que o conceito de fato cruza o limiar).
        if (
            self.prof_real[conceito_alvo_id] >= LIMIAR_DOMINIO
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
            "y_hat": y_hat,            # média da crença (ŷ do sistema) ANTES da obs.
            "p_real": p_real,          # probabilidade real (verdade oculta).
            "y": y,
            "ganho": ganho,
            "sondagem": sondagem,      # ganho de informação (redução do erro).
            "frustracao": frustracao,
            "incerteza": self._crenca_desvio(conceito_alvo_id),  # desvio da crença.
            "erro_crenca": abs(p_real - y_hat),  # quão errada estava a crença.
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