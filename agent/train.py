"""
train.py
========
Loop principal de treinamento do agente DQN do Tutor Inteligente (ITS).

Conecta os dois domínios do projeto:

    StudentEnvironment (env/) ---- (estado, recompensa) ----> DQNAgent (agent/)
            ^                                                       |
            |------------------- ação pedagógica -------------------|

A cada episódio o agente interage com o aluno simulado por vários passos,
armazenando as transições no Experience Replay e otimizando a Q-Network. O
epsilon (taxa de exploração) decai ao longo do tempo: o agente começa
explorando bastante e gradualmente passa a confiar na política aprendida.

Execução:
    python -m agent.train
"""

from __future__ import annotations

import os

import matplotlib

# Backend "Agg": renderiza para arquivo sem precisar de interface gráfica
# (essencial para rodar em servidores/containers sem display).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import após set do backend)
import numpy as np

from agent.dqn_agent import ACOES, DQNAgent
from data.database_setup import (
    DB_PATH,
    DB_URL,
    ESTUDANTE_TESTE_ID,
    criar_banco_e_popular,
)
from env.student_env import StudentEnvironment


NUM_EPISODIOS = 500          # Quantidade de episódios de treinamento.
MAX_PASSOS_POR_EPISODIO = 50  # Teto de passos (o env também encerra por fadiga).
BATCH_SIZE = 64              # Tamanho do mini-batch do Experience Replay.

# Decaimento exponencial do epsilon (exploração -> explotação).
EPSILON_INICIAL = 1.0       # Início: 100% exploração.
EPSILON_FINAL = 0.05        # Piso: mantém um mínimo de exploração.
EPSILON_DECAIMENTO = 0.99   # Fator multiplicativo por episódio.

# Frequência (em episódios) dos logs de progresso no console.
LOG_A_CADA = 20

# Caminho de saída dos pesos treinados (alinha com data/weights/ do README).
DIR_PESOS = os.path.join(os.path.dirname(DB_PATH), "weights")
CAMINHO_PESOS = os.path.join(DIR_PESOS, "dqn_policy.pt")

# Caminho de saída do gráfico da curva de aprendizado.
CAMINHO_GRAFICO = os.path.join(DIR_PESOS, "recompensa_vs_episodios.png")

# Janela da média móvel usada para suavizar a curva de recompensa.
JANELA_MEDIA_MOVEL = 20


def salvar_grafico_recompensa(
    historico_recompensas: list[float],
    caminho: str = CAMINHO_GRAFICO,
) -> None:
    """
    Salva o gráfico "Recompensa vs. Episódios" destacando o melhor desempenho.

    Plota a recompensa por episódio, uma média móvel para revelar a tendência de
    aprendizado e marca o episódio de MELHOR desempenho (maior recompensa).

    Parâmetros
    ----------
    historico_recompensas : list[float]
        Recompensa acumulada de cada episódio.
    caminho : str
        Arquivo de imagem (.png) onde o gráfico será salvo.
    """
    if not historico_recompensas:
        print("Sem histórico de recompensas: gráfico não gerado.")
        return

    recompensas = np.asarray(historico_recompensas, dtype=np.float32)
    episodios = np.arange(1, len(recompensas) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Recompensa bruta por episódio (linha tênue ao fundo).
    ax.plot(
        episodios,
        recompensas,
        color="#9ec5fe",
        linewidth=1.0,
        alpha=0.8,
        label="Recompensa por episódio",
    )

    # Média móvel: tendência suavizada do aprendizado.
    # O "melhor desempenho" é o PICO da MÉDIA MÓVEL (tendência sustentada),
    # e não um pico isolado da recompensa bruta (que pode vir de exploração).
    tem_media_movel = len(recompensas) >= JANELA_MEDIA_MOVEL
    if tem_media_movel:
        media_movel = np.convolve(
            recompensas,
            np.ones(JANELA_MEDIA_MOVEL) / JANELA_MEDIA_MOVEL,
            mode="valid",
        )
        eixo_media = np.arange(JANELA_MEDIA_MOVEL, len(recompensas) + 1)
        ax.plot(
            eixo_media,
            media_movel,
            color="#1f77b4",
            linewidth=2.2,
            label=f"Média móvel ({JANELA_MEDIA_MOVEL} episódios)",
        )

        idx_melhor = int(np.argmax(media_movel))
        melhor_episodio = int(eixo_media[idx_melhor])
        melhor_valor = float(media_movel[idx_melhor])
        rotulo_metrica = "média móvel"
    else:
        # Poucos episódios para média móvel: usa a recompensa bruta como fallback.
        idx_melhor = int(np.argmax(recompensas))
        melhor_episodio = idx_melhor + 1
        melhor_valor = float(recompensas[idx_melhor])
        rotulo_metrica = "recompensa"

    # Marca o melhor desempenho (pico sustentado da tendência de aprendizado).
    ax.scatter(
        melhor_episodio,
        melhor_valor,
        color="#d62728",
        s=80,
        zorder=5,
        label=f"Melhor ({rotulo_metrica}): ep. {melhor_episodio} ({melhor_valor:+.2f})",
    )
    ax.annotate(
        f"melhor desempenho ({rotulo_metrica})\nep. {melhor_episodio} | {melhor_valor:+.2f}",
        xy=(melhor_episodio, melhor_valor),
        xytext=(0.98, 0.05),
        textcoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    ax.set_title("Recompensa vs. Episódios — Treinamento do Agente DQN")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa acumulada")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)  # libera memória da figura.
    print(f"Gráfico de aprendizado salvo em: {caminho}")


def garantir_banco() -> None:
    """Cria e popula o banco caso o arquivo SQLite ainda não exista."""
    if not os.path.exists(DB_PATH):
        print("Banco não encontrado. Criando e populando do zero...")
        criar_banco_e_popular(reset=True)


def treinar() -> DQNAgent:
    """
    Executa o loop de episódios e retorna o agente treinado.

    Retorna
    -------
    DQNAgent
        Agente com a policy_net já treinada.
    """
    garantir_banco()

    # Ambiente simulador (aluno fictício) e agente DQN.
    env = StudentEnvironment(DB_URL, ESTUDANTE_TESTE_ID)
    agente = DQNAgent(dim_estado=env.dim_estado, dim_acoes=len(ACOES))

    print(
        f"Iniciando treino | device={agente.device} | "
        f"dim_estado={env.dim_estado} | acoes={ACOES}"
    )

    epsilon = EPSILON_INICIAL
    historico_recompensas: list[float] = []

    try:
        for episodio in range(1, NUM_EPISODIOS + 1):
            estado = env.reset()
            recompensa_acumulada = 0.0
            perdas: list[float] = []

            for _ in range(MAX_PASSOS_POR_EPISODIO):
                # 1. Política epsilon-greedy escolhe o índice da ação.
                acao_idx = agente.select_action(estado, epsilon)
                # 2. Traduz o índice para a string esperada pelo ambiente.
                acao_str = ACOES[acao_idx]

                # 3. Ambiente executa a ação e devolve a transição.
                proximo_estado, recompensa, done, _info = env.step(acao_str)

                # 4. Armazena a experiência no Replay Buffer.
                agente.store_transition(
                    estado, acao_idx, recompensa, proximo_estado, done
                )

                # 5. Um passo de otimização do DQN (se houver amostras).
                loss = agente.optimize_model(BATCH_SIZE)
                if loss is not None:
                    perdas.append(loss)

                estado = proximo_estado
                recompensa_acumulada += recompensa

                if done:
                    break

            historico_recompensas.append(recompensa_acumulada)

            # Decaimento do epsilon ao fim de cada episódio (com piso).
            epsilon = max(EPSILON_FINAL, epsilon * EPSILON_DECAIMENTO)

            # Log periódico de progresso.
            if episodio % LOG_A_CADA == 0:
                media_recompensa = np.mean(historico_recompensas[-LOG_A_CADA:])
                media_perda = np.mean(perdas) if perdas else float("nan")
                print(
                    f"Episódio {episodio:4d}/{NUM_EPISODIOS} | "
                    f"epsilon={epsilon:.3f} | "
                    f"recompensa_média={media_recompensa:+.3f} | "
                    f"perda_média={media_perda:.4f}"
                )
    finally:
        env.close()  # garante fechamento da sessão do SQLAlchemy.

    # Persiste os pesos treinados.
    os.makedirs(DIR_PESOS, exist_ok=True)
    agente.salvar(CAMINHO_PESOS)
    print(f"Treino concluído. Pesos salvos em: {CAMINHO_PESOS}")

    # Gera e salva o gráfico da curva de aprendizado (Recompensa vs. Episódios).
    salvar_grafico_recompensa(historico_recompensas)

    return agente


if __name__ == "__main__":
    treinar()
