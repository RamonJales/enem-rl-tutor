"""
avaliar_robustez.py
====================
Avaliação de ROBUSTEZ da política treinada do Tutor Inteligente (ITS).

Roda a política GULOSA (epsilon=0) já treinada contra DIFERENTES perfis de
aluno e mede como o desempenho se mantém fora da distribuição de treino. É um
teste de GENERALIZAÇÃO out-of-distribution: o agente nunca viu esses perfis
durante o treino.

Perfis avaliados:
    - "Modelo interno (treino)" : o próprio StudentEnvironment (distribuição de
      treino) — serve de BASELINE para comparação.
    - "Aluno consistente"       : ConsistentStudentBot (responde conforme a
      proficiência, sem chutes selvagens).
    - "Aluno chutador"          : GuessingStudentBot (alta variância: chuta nas
      difíceis e tem desatenção nas demais).

IMPORTANTE — não afeta o modelo nem o treino:
    - Modo INFERÊNCIA puro: sem optimize_model, sem salvar pesos (apenas LÊ
      data/weights/dqn_policy.pt).
    - NÃO altera o student_env.py. Para os perfis-bot, uma subclasse fina
      (`AmbienteComBot`) sobrescreve APENAS a fonte da "verdade oculta"
      (_probabilidade_acerto); todo o resto do MDP (estado, crença BKT,
      mapeamento ação->conceito no DAG, recompensa, fadiga) é idêntico, então
      os pesos treinados permanecem válidos.

Execução (a partir da raiz do projeto):
    python -m agent.avaliar_robustez
"""

from __future__ import annotations

import os

import matplotlib

# Backend "Agg": renderiza para arquivo sem precisar de display (servers/CI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import após set do backend)
import numpy as np  # noqa: E402

from agent.dqn_agent import ACOES, DQNAgent  # noqa: E402
from agent.train import (  # noqa: E402
    CAMINHO_PESOS,
    MAX_PASSOS_POR_EPISODIO,
    garantir_banco,
)
from data.database_setup import DB_URL, ESTUDANTE_TESTE_ID  # noqa: E402
from env.bots import ConsistentStudentBot, GuessingStudentBot  # noqa: E402
from env.student_env import StudentEnvironment  # noqa: E402


# Nº de episódios por perfil (a política é gulosa, mas o ambiente é estocástico
# na amostragem do resultado y, então a média sobre episódios é significativa).
N_EPISODIOS = 50

# Saída dos gráficos (pasta de figuras da documentação).
DIR_FIGURAS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "figuras"
)
CAMINHO_GRAFICO = os.path.join(DIR_FIGURAS, "avaliacao_robustez.png")


def _prof_para_theta(p: float) -> float:
    """Converte proficiência [0, 1] para θ (logit) em ~[-3, 3], escala dos bots."""
    return (p * 2.0 - 1.0) * 3.0


class AmbienteComBot(StudentEnvironment):
    """
    StudentEnvironment cujo COMPORTAMENTO de resposta vem de um bot (Opção A).

    A habilidade-base (incluindo os laços de APRENDIZADO e de PRÉ-REQUISITOS) é a
    do próprio simulador — preservada via `super()._probabilidade_acerto(...)`,
    que usa a `prof_real` viva. Sobre essa base, o bot aplica apenas sua
    DISTORÇÃO COMPORTAMENTAL (ex.: chute nas difíceis, desatenção). Assim o teste
    mede robustez a COMPORTAMENTO sem quebrar a dinâmica do aluno, e os pesos
    treinados permanecem válidos (estado, crença, DAG e recompensa intactos).
    """

    def __init__(self, db_url: str, estudante_id: int, bot=None, **kwargs) -> None:
        super().__init__(db_url, estudante_id, **kwargs)
        # Definido após a construção (precisa de proficiencia_inicial/conceito_ids).
        self.bot = bot

    def _nivel_por_dificuldade(self, conceito_id: int) -> str:
        """Mapeia a dificuldade [0,1] do conceito para o nível esperado pelo bot."""
        d = self.dificuldade_conceito[conceito_id]
        if d < 0.4:
            return "easy"
        if d < 0.7:
            return "medium"
        return "hard"

    def _probabilidade_acerto(self, conceito_id, prof):  # type: ignore[override]
        # 1. Habilidade-base do ambiente (aprende com prof_real + acopla pré-req).
        base = super()._probabilidade_acerto(conceito_id, prof)
        # 2. Camada comportamental do perfil (consistente = sem distorção).
        nivel = self._nivel_por_dificuldade(conceito_id)
        return self.bot.behavioral_probability(base, nivel)


def construir_bot(env: AmbienteComBot, classe_bot):
    """
    Instancia o bot do perfil.

    Na Opção A a habilidade-base vem do ambiente, então o θ por conceito do bot
    não entra na probabilidade de acerto da avaliação; ainda assim o semeamos a
    partir da proficiência inicial para manter o bot um objeto válido e coerente
    (útil se `answer_question`/`success_probability` forem usados isoladamente).
    """
    thetas = {
        str(cid): _prof_para_theta(env.proficiencia_inicial[cid])
        for cid in env.conceito_ids
    }
    return classe_bot(thetas)


def avaliar(
    agente: DQNAgent,
    env: StudentEnvironment,
    n_episodios: int = N_EPISODIOS,
    max_passos: int = MAX_PASSOS_POR_EPISODIO,
) -> dict[str, np.ndarray | float]:
    """
    Roda a política gulosa por `n_episodios` e coleta métricas por episódio.

    Retorna
    -------
    dict com:
        recompensas  : np.ndarray (recompensa acumulada por episódio)
        dominados    : np.ndarray (nº de conceitos dominados ao fim)
        taxa_sucesso : float      (fração de episódios que atingiram o nível avançado)
    """
    recompensas: list[float] = []
    dominados: list[int] = []
    sucessos: list[int] = []

    for _ in range(n_episodios):
        estado = env.reset()
        total = 0.0
        info: dict = {}
        for _ in range(max_passos):
            acao_idx = agente.select_action(estado, epsilon=0.0)  # gulosa.
            estado, recompensa, done, info = env.step(ACOES[acao_idx])
            total += recompensa
            if done:
                break
        recompensas.append(total)
        dominados.append(int(info.get("n_dominados", 0)))
        sucessos.append(int(info.get("objetivo_atingido", False)))

    return {
        "recompensas": np.asarray(recompensas, dtype=np.float32),
        "dominados": np.asarray(dominados, dtype=np.float32),
        "taxa_sucesso": float(np.mean(sucessos)),
    }


def salvar_graficos(
    resultados: dict[str, dict],
    n_conceitos: int,
    caminho: str = CAMINHO_GRAFICO,
) -> None:
    """
    Gera o painel comparativo de robustez (3 métricas) e salva em `caminho`.

    Painéis:
      1. Distribuição da recompensa acumulada por perfil (boxplot).
      2. Taxa de "nível avançado" atingido por perfil (barras, %).
      3. Conceitos dominados por perfil (barras, média ± desvio).
    """
    rotulos = list(resultados.keys())
    cores = ["#1f77b4", "#2ca02c", "#d62728"][: len(rotulos)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Recompensa acumulada (boxplot da distribuição por episódio).
    dados_box = [resultados[r]["recompensas"] for r in rotulos]
    caixas = axes[0].boxplot(dados_box, patch_artist=True)
    for caixa, cor in zip(caixas["boxes"], cores):
        caixa.set_facecolor(cor)
        caixa.set_alpha(0.6)
    # Define os rótulos do eixo x sem usar o parâmetro `labels` do boxplot
    # (depreciado a partir do Matplotlib 3.9), mantendo compatibilidade.
    axes[0].set_xticks(range(1, len(rotulos) + 1))
    axes[0].set_xticklabels(rotulos)
    axes[0].set_title("Recompensa acumulada por episódio")
    axes[0].set_ylabel("Recompensa")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)

    # 2. Taxa de nível avançado (%).
    taxas = [resultados[r]["taxa_sucesso"] * 100.0 for r in rotulos]
    barras = axes[1].bar(rotulos, taxas, color=cores, alpha=0.85)
    axes[1].set_title("Taxa de nível avançado atingido")
    axes[1].set_ylabel("Episódios com sucesso (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)
    for barra, taxa in zip(barras, taxas):
        axes[1].text(
            barra.get_x() + barra.get_width() / 2,
            taxa + 1.5,
            f"{taxa:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 3. Conceitos dominados (média ± desvio).
    medias = [float(np.mean(resultados[r]["dominados"])) for r in rotulos]
    desvios = [float(np.std(resultados[r]["dominados"])) for r in rotulos]
    barras = axes[2].bar(
        rotulos, medias, yerr=desvios, color=cores, alpha=0.85, capsize=5
    )
    axes[2].set_title("Conceitos dominados ao fim do episódio")
    axes[2].set_ylabel(f"Conceitos dominados (de {n_conceitos})")
    axes[2].set_ylim(0, n_conceitos)
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=15)
    for barra, media, desvio in zip(barras, medias, desvios):
        axes[2].text(
            barra.get_x() + barra.get_width() / 2,
            media + desvio + 0.2,  # acima do topo da barra de erro.
            f"{media:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(
        "Avaliação de Robustez — Política gulosa contra perfis de aluno",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"\nGráfico de robustez salvo em: {caminho}")


def avaliar_robustez(n_episodios: int = N_EPISODIOS) -> dict[str, dict]:
    """
    Carrega a política treinada e a avalia contra cada perfil de aluno.

    Retorna o dicionário de resultados {rótulo: métricas} e salva o painel de
    gráficos para a documentação.
    """
    garantir_banco()

    if not os.path.exists(CAMINHO_PESOS):
        raise FileNotFoundError(
            f"Pesos treinados não encontrados em {CAMINHO_PESOS}. "
            "Treine o agente primeiro com 'python -m agent.train'."
        )

    # Carrega o agente treinado UMA vez (dim_estado é o mesmo para todos os
    # ambientes, pois compartilham o banco/quantidade de conceitos).
    base_tmp = StudentEnvironment(DB_URL, ESTUDANTE_TESTE_ID)
    dim_estado = base_tmp.dim_estado
    n_conceitos = base_tmp.n_conceitos
    base_tmp.close()

    agente = DQNAgent(dim_estado=dim_estado, dim_acoes=len(ACOES))
    agente.carregar(CAMINHO_PESOS)

    print(
        f"Avaliando robustez | device={agente.device} | "
        f"episódios/perfil={n_episodios}\n"
    )

    resultados: dict[str, dict] = {}

    # Fábrica de ambientes por perfil. O baseline usa o simulador interno;
    # os demais injetam o bot na verdade oculta.
    def _construir_ambiente_bot(classe_bot):
        env = AmbienteComBot(DB_URL, ESTUDANTE_TESTE_ID)
        env.bot = construir_bot(env, classe_bot)
        return env

    perfis = [
        ("Modelo interno (treino)", lambda: StudentEnvironment(DB_URL, ESTUDANTE_TESTE_ID)),
        ("Aluno consistente", lambda: _construir_ambiente_bot(ConsistentStudentBot)),
        ("Aluno chutador", lambda: _construir_ambiente_bot(GuessingStudentBot)),
    ]

    print(f"{'Perfil':28s} | {'recompensa (méd±dp)':22s} | {'nível avançado':14s} | dominados")
    print("-" * 86)
    for rotulo, fabrica in perfis:
        env = fabrica()
        try:
            metricas = avaliar(agente, env, n_episodios=n_episodios)
        finally:
            env.close()
        resultados[rotulo] = metricas

        rec = metricas["recompensas"]
        dom = metricas["dominados"]
        print(
            f"{rotulo:28s} | {rec.mean():+8.2f} ± {rec.std():5.2f}      | "
            f"{metricas['taxa_sucesso']:11.0%}    | "
            f"{dom.mean():.1f}/{n_conceitos}"
        )

    salvar_graficos(resultados, n_conceitos=n_conceitos)
    return resultados


if __name__ == "__main__":
    avaliar_robustez()
