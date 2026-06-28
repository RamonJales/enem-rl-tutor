"""
Q-Network (rede neural) do agente DQN do Sistema Tutor Inteligente (ITS).

Esta rede aproxima a função de valor-ação Q(s, a): dado o vetor de Estado S do
aluno (proficiências por conceito), estima o "valor esperado" de cada uma das 3
Ações Pedagógicas possíveis ("Avançar", "Reforçar", "Remediar").

Arquitetura: uma MLP (Perceptron Multicamadas) simples com duas camadas ocultas
e ativação ReLU. A camada de saída NÃO tem ativação, pois os Q-values são
escalares reais (positivos ou negativos), não probabilidades.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network: MLP que mapeia Estado (S) -> Q-values das ações (A).

    Entrada : vetor de Estado com `dim_estado` features (proficiências).
    Saída   : `dim_acoes` Q-values (um por Ação Pedagógica).
    """

    def __init__(
        self,
        dim_estado: int,
        dim_acoes: int = 3,
        dim_oculta: int = 128,
    ) -> None:
        """
        Parâmetros
        ----------
        dim_estado : int
            Dimensão do vetor de Estado (número de conceitos = `env.dim_estado`).
        dim_acoes : int
            Tamanho do espaço de ações (3: Avançar, Reforçar, Remediar).
        dim_oculta : int
            Número de neurônios em cada camada oculta.
        """
        super().__init__()

        # MLP: entrada -> oculta1 -> ReLU -> oculta2 -> ReLU -> saída (Q-values).
        # A saída fica "crua" (sem ativação) porque representa valores Q reais.
        self.rede = nn.Sequential(
            nn.Linear(dim_estado, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_acoes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta (forward pass).

        Parâmetros
        ----------
        x : torch.Tensor
            Tensor de Estado(s) com shape (batch, dim_estado) ou (dim_estado,).

        Retorna
        -------
        torch.Tensor
            Q-values com shape (batch, dim_acoes).
        """
        return self.rede(x)
