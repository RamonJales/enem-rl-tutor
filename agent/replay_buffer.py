from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class LoteTransicoes(NamedTuple):
    """Mini-batch já convertido para tensores torch, pronto para o treino."""
    estados: torch.Tensor       # (batch, dim_estado) float32
    acoes: torch.Tensor         # (batch, 1)          long
    recompensas: torch.Tensor   # (batch, 1)          float32
    proximos_estados: torch.Tensor  # (batch, dim_estado) float32
    dones: torch.Tensor         # (batch, 1)          float32


class ReplayBuffer:
    """
    Memória circular de transições para amostragem de mini-batches.

    Usa um `deque` com `maxlen`: quando cheio, as experiências mais antigas são
    descartadas automaticamente (janela deslizante de memória).
    """

    def __init__(self, capacidade: int = 10_000) -> None:
        """
        Parâmetros
        ----------
        capacidade : int
            Número máximo de transições retidas simultaneamente.
        """
        self.memoria: deque[tuple] = deque(maxlen=capacidade)

    def push(
        self,
        estado: np.ndarray,
        acao: int,
        recompensa: float,
        proximo_estado: np.ndarray,
        done: bool,
    ) -> None:
        """
        Armazena uma transição no buffer.

        Parâmetros
        ----------
        estado : np.ndarray
            Estado S (vetor de proficiências).
        acao : int
            Índice da ação tomada (0=Avançar, 1=Reforçar, 2=Remediar).
        recompensa : float
            Recompensa R_t = y - ŷ devolvida pelo ambiente.
        proximo_estado : np.ndarray
            Estado S' resultante da transição.
        done : bool
            True se o episódio terminou após esta transição.
        """
        # Copiamos os estados como float32 para garantir consistência de tipo
        # e evitar referências mutáveis ao array original do ambiente.
        self.memoria.append(
            (
                np.asarray(estado, dtype=np.float32),
                int(acao),
                float(recompensa),
                np.asarray(proximo_estado, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> LoteTransicoes:
        """
        Sorteia um mini-batch aleatório e o converte para tensores torch.

        Amostrar aleatoriamente (sem reposição) descorrelaciona as experiências,
        estabilizando o gradiente do DQN.

        Parâmetros
        ----------
        batch_size : int
            Tamanho do mini-batch.
        device : torch.device | str
            Dispositivo de destino dos tensores ('cpu' ou 'cuda').

        Retorna
        -------
        LoteTransicoes
            Tupla nomeada com os tensores empilhados.
        """
        transicoes = random.sample(self.memoria, batch_size)
        # Desempacota a lista de tuplas em colunas (transpõe as transições).
        estados, acoes, recompensas, proximos, dones = zip(*transicoes)

        # Empilha em arrays/tensores. Ações, recompensas e dones viram colunas
        # (batch, 1) para casar com a saída gather/Bellman no agente.
        return LoteTransicoes(
            estados=torch.from_numpy(np.stack(estados)).to(device),
            acoes=torch.tensor(acoes, dtype=torch.long, device=device).unsqueeze(1),
            recompensas=torch.tensor(
                recompensas, dtype=torch.float32, device=device
            ).unsqueeze(1),
            proximos_estados=torch.from_numpy(np.stack(proximos)).to(device),
            dones=torch.tensor(
                dones, dtype=torch.float32, device=device
            ).unsqueeze(1),
        )

    def __len__(self) -> int:
        """Quantidade atual de transições armazenadas."""
        return len(self.memoria)
