"""
knowledge_graph.py
==================
Modelo de Domínio (regras do DAG de Matemática) do Tutor Inteligente.

Este módulo expressa as REGRAS PEDAGÓGICAS do grafo de pré-requisitos de forma
pura e reutilizável: navegação e PROPAGAÇÃO de conhecimento (uma resposta certa
reforça os pré-requisitos; uma errada penaliza os dependentes).

Posicionamento na arquitetura
------------------------------
Pertence ao domínio do Ambiente (`env/`), ao lado de `student_env.py` e
`bots.py`. É um utilitário de domínio independente e SEM estado de treino: NÃO
está no caminho crítico do DQN. O `StudentEnvironment` que treina o agente
mantém sua própria dinâmica (logística estilo TRI + crença Bayesiana) lida do
banco — intocada para preservar a reprodutibilidade e o desempenho do modelo já
treinado. Esta classe é consumida por testes, pela API e por notebooks que
precisam raciocinar sobre o DAG (ex.: propagar uma atualização de proficiência
para conceitos vizinhos) sem reabrir o simulador.
"""

from typing import Dict

class ConceptNode:
    """Represents a mathematical concept in the curriculum."""
    def __init__(self, node_id: str, name: str):
        self.node_id = node_id
        self.name = name

class KnowledgeGraph:
    """
    Directed Acyclic Graph (DAG) representing the domain model.
    Handles the knowledge flow (propagation) independent of the student's current state.
    """
    def __init__(self, decay_rate: float = 0.5):
        self.nodes: Dict[str, ConceptNode] = {}
        # Dependencies: target_id -> dict(pre_req_id -> weight)
        self.edges_backward: Dict[str, Dict[str, float]] = {}
        # Dependencies: pre_req_id -> dict(target_id -> weight)
        self.edges_forward: Dict[str, Dict[str, float]] = {}
        self.decay_rate = decay_rate

    def add_node(self, node_id: str, name: str) -> None:
        """Registers a new concept in the domain."""
        self.nodes[node_id] = ConceptNode(node_id, name)
        self.edges_backward[node_id] = {}
        self.edges_forward[node_id] = {}

    def add_edge(self, pre_req_id: str, target_id: str, weight: float) -> None:
        """Creates a dependency link: pre_req -> target."""
        if pre_req_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")
        
        self.edges_backward[target_id][pre_req_id] = weight
        self.edges_forward[pre_req_id][target_id] = weight

    def propagate_update(self, 
                         student_state: Dict[str, float], 
                         target_node_id: str, 
                         delta_p: float) -> Dict[str, float]:
        """
        Applies local update and propagates knowledge through the graph.
        Returns a completely new state dictionary (pure function approach).
        """
        if target_node_id not in self.nodes:
            raise KeyError(f"Node '{target_node_id}' does not exist in the graph.")

        # Create a copy to avoid mutating the original state directly
        new_state = student_state.copy()
        
        # 1. Local Update
        current_p = new_state.get(target_node_id, 0.0)
        new_state[target_node_id] = self._clamp(current_p + delta_p)
        
        # 2. Network Propagation
        if delta_p > 0:
            # Success: Backward Inference (infer mastery of prerequisites)
            self._propagate_backward(new_state, target_node_id, delta_p)
        else:
            # Failure: Forward Penalty (cascade penalty to dependent concepts)
            self._propagate_forward(new_state, target_node_id, delta_p)
            
        return new_state

    def _propagate_backward(self, state: Dict[str, float], current_id: str, delta: float) -> None:
        for pre_req_id, weight in self.edges_backward.get(current_id, {}).items():
            propagation_val = delta * weight * self.decay_rate
            current_p = state.get(pre_req_id, 0.0)
            state[pre_req_id] = self._clamp(current_p + propagation_val)
            self._propagate_backward(state, pre_req_id, propagation_val)

    def _propagate_forward(self, state: Dict[str, float], current_id: str, delta: float) -> None:
        for child_id, weight in self.edges_forward.get(current_id, {}).items():
            propagation_val = delta * weight * self.decay_rate
            current_p = state.get(child_id, 0.0)
            state[child_id] = self._clamp(current_p + propagation_val)
            self._propagate_forward(state, child_id, propagation_val)

    @staticmethod
    def _clamp(value: float) -> float:
        """Restricts probability values between 0.0 and 1.0."""
        return max(0.0, min(1.0, value))
