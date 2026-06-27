"""
Agente Deep Q-Network (DQN) do Sistema Tutor Inteligente (ITS).

    - policy_net : Q-Network treinada a cada passo (gera as ações e os gradientes).
    - target_net : cópia "congelada" da policy_net, atualizada periodicamente.
                   Estabiliza o alvo de Bellman (evita perseguir um alvo móvel).
    - ReplayBuffer : memória de experiências amostrada em mini-batches.

A política de seleção é epsilon-greedy: com probabilidade epsilon o agente
EXPLORA (ação aleatória) e, caso contrário, EXPLOTA (melhor ação segundo a rede).
A perda é a Huber Loss entre os Q-values previstos e o alvo de Bellman:

    alvo = r + gamma * max_a' Q_target(s', a') * (1 - done)
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from agent.model import DQN
from agent.replay_buffer import ReplayBuffer

# Espaço de ações: espelha ACOES_VALIDAS de env/student_env.py (mesma ordem).
# Índice -> nome da Ação Pedagógica.
ACOES: tuple[str, ...] = ("Avançar", "Reforçar", "Remediar")


class DQNAgent:
    """
    Agente DQN: encapsula as redes, o otimizador, o buffer e a lógica de treino.
    """

    def __init__(
        self,
        dim_estado: int,
        dim_acoes: int = 3,
        dim_oculta: int = 128,
        gamma: float = 0.99,
        lr: float = 1e-3,
        capacidade_buffer: int = 10_000,
        tau: float = 0.005,
        device: torch.device | str | None = None,
    ) -> None:
        """
        Parâmetros
        ----------
        dim_estado : int
            Dimensão do vetor de Estado (= `env.dim_estado`).
        dim_acoes : int
            Tamanho do espaço de ações (3).
        dim_oculta : int
            Neurônios por camada oculta da Q-Network.
        gamma : float
            Fator de desconto da recompensa futura (equação de Bellman).
        lr : float
            Taxa de aprendizado do otimizador Adam.
        capacidade_buffer : int
            Capacidade máxima do Experience Replay.
        tau : float
            Coeficiente do soft update (Polyak) da target_net a cada passo:
            θ_target <- τ·θ_policy + (1-τ)·θ_target. Valores pequenos (~0.005)
            fazem a target_net seguir a policy_net de forma suave e contínua,
            eliminando os saltos do "hard update" (causa das curvas serrilhadas).
        device : torch.device | str | None
            Dispositivo de execução. Se None, escolhe CUDA se disponível.
        """
        # Seleção automática de dispositivo (GPU se houver, senão CPU).
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dim_estado = dim_estado
        self.dim_acoes = dim_acoes
        self.gamma = gamma
        self.tau = tau

        # policy_net: rede treinada (gera ações e recebe os gradientes).
        self.policy_net = DQN(dim_estado, dim_acoes, dim_oculta).to(self.device)
        # target_net: cópia estável usada para calcular o alvo de Bellman.
        self.target_net = DQN(dim_estado, dim_acoes, dim_oculta).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # nunca treina diretamente (sem dropout/grad).

        # Otimizador Adam sobre os parâmetros da policy_net.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # Huber Loss (SmoothL1): robusta a outliers, padrão em DQN.
        self.criterion = nn.SmoothL1Loss()

        # Memória de Experience Replay.
        self.buffer = ReplayBuffer(capacidade_buffer)

        # Contador de passos de otimização (controla a sincronização do target).
        self.passos_otimizacao = 0

    # ------------------------------------------------------------------ #
    # Seleção de ação (política epsilon-greedy)
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Escolhe uma ação via política epsilon-greedy.

        Parâmetros
        ----------
        state : np.ndarray
            Estado atual S (vetor de proficiências).
        epsilon : float
            Probabilidade de EXPLORAR (ação aleatória). Em (1 - epsilon) o
            agente EXPLOTA a melhor ação segundo a policy_net.

        Retorna
        -------
        int
            Índice da ação escolhida (0=Avançar, 1=Reforçar, 2=Remediar).
        """
        # EXPLORAÇÃO: com probabilidade epsilon, ação totalmente aleatória.
        if random.random() < epsilon:
            return random.randrange(self.dim_acoes)

        # EXPLOTAÇÃO: melhor ação segundo a rede (argmax dos Q-values).
        # Sem gradiente aqui (inferência apenas).
        with torch.no_grad():
            estado_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # (1, dim_estado)
            q_values = self.policy_net(estado_t)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # Armazenamento de transições
    # ------------------------------------------------------------------ #
    def store_transition(
        self,
        estado: np.ndarray,
        acao: int,
        recompensa: float,
        proximo_estado: np.ndarray,
        done: bool,
    ) -> None:
        """Adiciona uma transição (s, a, r, s', done) ao Experience Replay."""
        self.buffer.push(estado, acao, recompensa, proximo_estado, done)

    # ------------------------------------------------------------------ #
    # Otimização (um passo de treino do DQN)
    # ------------------------------------------------------------------ #
    def optimize_model(self, batch_size: int) -> float | None:
        """
        Executa um passo de otimização do DQN.

        Etapas:
          1. Amostra um mini-batch do buffer (None se ainda não há amostras).
          2. Q atuais: Q_policy(s, a) para as ações realmente tomadas.
          3. Alvo de Bellman: r + gamma * max_a' Q_target(s', a') * (1 - done).
          4. Loss Huber entre Q atual e alvo.
          5. Backpropagation + passo do Adam (com clip de gradiente).
          6. Soft update (Polyak) da target_net a cada passo.

        Retorna
        -------
        float | None
            Valor da perda (para logging), ou None se não houve amostras.
        """
        # 1. Sem amostras suficientes ainda: nada a otimizar.
        if len(self.buffer) < batch_size:
            return None

        lote = self.buffer.sample(batch_size, device=self.device)

        # 2. Q-values atuais: seleciona, via gather, o Q da ação tomada.
        #    policy_net(estados) -> (batch, dim_acoes); gather -> (batch, 1).
        q_atual = self.policy_net(lote.estados).gather(1, lote.acoes)

        # 3. Alvo de Bellman usando a target_net (sem gradiente).
        with torch.no_grad():
            # max_a' Q_target(s', a') -> (batch, 1).
            q_proximo_max = self.target_net(lote.proximos_estados).max(
                dim=1, keepdim=True
            ).values
            # Em estados terminais (done=1), não há valor futuro -> (1 - done).
            q_alvo = lote.recompensas + self.gamma * q_proximo_max * (
                1.0 - lote.dones
            )

        # 4. Perda Huber entre previsão (policy) e alvo (Bellman).
        loss = self.criterion(q_atual, q_alvo)

        # 5. Backpropagation e atualização dos pesos da policy_net.
        self.optimizer.zero_grad()
        loss.backward()
        # Clip de gradiente para evitar explosão de gradientes (estabilidade).
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 6. Soft update (Polyak) da target_net a cada passo.
        self.passos_otimizacao += 1
        self.update_target()

        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Sincronização da target_net
    # ------------------------------------------------------------------ #
    def update_target(self) -> None:
        """Soft update (Polyak): θ_target <- τ·θ_policy + (1-τ)·θ_target."""
        with torch.no_grad():
            for p_target, p_policy in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                p_target.mul_(1.0 - self.tau).add_(p_policy, alpha=self.tau)

    # ------------------------------------------------------------------ #
    # Persistência dos pesos
    # ------------------------------------------------------------------ #
    def salvar(self, caminho: str) -> None:
        """Salva os pesos da policy_net e o estado do otimizador em `.pt`."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "passos_otimizacao": self.passos_otimizacao,
            },
            caminho,
        )

    def carregar(self, caminho: str) -> None:
        """Carrega os pesos previamente salvos para retomar treino/inferência."""
        checkpoint = torch.load(caminho, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.passos_otimizacao = checkpoint.get("passos_otimizacao", 0)
