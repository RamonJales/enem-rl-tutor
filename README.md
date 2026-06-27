# enem-rl-tutor

The project abstracts learning into a dynamic process based on the student's interaction with mathematical concepts. Theoretical inspirations include the Zone of Proximal Development (Vygotsky), Item Response Theory (IRT), and the principles of Metacognition (Open Learner Models).

## Visão geral

Sistema Tutor Inteligente (ITS) que usa Aprendizado por Reforço Profundo (DQN) para
escolher a melhor Ação Pedagógica para um aluno a cada passo:

- `Avançar` — apresenta um conceito sucessor no grafo de pré-requisitos (DAG).
- `Reforçar` — mantém o aluno no mesmo conceito (mais prática).
- `Remediar` — retrocede para um pré-requisito do conceito atual.

A recompensa é dinâmica: `R_t = y - ŷ` (acerto observado menos acerto esperado),
incentivando o tutor a manter o aluno na Zona de Desenvolvimento Proximal.

## Requisitos

- Python 3.10+ (usa sintaxe de tipos como `float | None`).
- Dependências em [requirements.txt](requirements.txt): `torch`, `numpy`, `sqlalchemy`, `fastapi`, `uvicorn`, `pydantic`.

## Instalação

```bash
# (opcional) crie e ative um ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# instale as dependências
pip install -r requirements.txt
```

## Como rodar

Sempre execute a partir da raiz do projeto (`enem_rl_tutor/`) usando a flag `-m`,
para que os imports de pacote (`agent`, `env`, `data`) funcionem corretamente.

### 1. Criar e popular o banco de dados

Gera o SQLite `data/enem_tutor.db` com o grafo de conceitos, questões e o estado
inicial do aluno de teste:

```bash
python -m data.database_setup
```

### 2. Treinar o agente DQN

Roda o loop de episódios (o ambiente conecta ao banco; se ele não existir, é
criado automaticamente). Os pesos treinados são salvos em `data/weights/dqn_policy.pt`:

```bash
python -m agent.train
```

Hiperparâmetros (número de episódios, `epsilon`, `batch_size`, etc.) ficam no topo
de [agent/train.py](agent/train.py) e podem ser ajustados conforme necessário.

Ao final do treino, além dos pesos, é gerado automaticamente o gráfico da curva de
aprendizado em `data/weights/recompensa_vs_episodios.png` (Recompensa vs. Episódios,
com média móvel e o melhor desempenho destacado) para análise posterior.

### 3. Avaliar a robustez da política (após o treino)

Com os pesos já treinados em `data/weights/dqn_policy.pt`, esta etapa roda a
política **gulosa** (sem explorar e sem treinar) contra diferentes **perfis de
aluno** para medir a generalização *out-of-distribution* (o agente nunca viu
esses perfis no treino):

A habilidade-base do aluno (com aprendizado e pré-requisitos) é sempre a do
simulador; cada bot aplica apenas sua **distorção de comportamento** por cima:

- **Modelo interno (treino)** — o próprio `StudentEnvironment` (baseline).
- **Aluno consistente** — `ConsistentStudentBot` (sem distorção; serve de
  verificação de sanidade, deve acompanhar o baseline).
- **Aluno chutador** — `GuessingStudentBot` (chuta nas questões difíceis e tem
  desatenção nas demais — alta variância).

```bash
python -m agent.avaliar_robustez
```

É um passo de **inferência puro**: não altera `student_env.py`, não otimiza a
rede e não sobrescreve os pesos — apenas LÊ o `.pt` treinado. Ao final, imprime
um resumo no console e salva o painel comparativo (recompensa, taxa de nível
avançado e conceitos dominados por perfil) em
`docs/figuras/avaliacao_robustez.png` para a documentação.

### 4. Rodar os testes

A suíte cobre o Modelo de Domínio (`env/knowledge_graph.py`), os perfis de aluno
(`env/bots.py`) e um teste de integração do fluxo real (`StudentEnvironment` +
`DQNAgent`) executado contra um banco SQLite temporário — sem tocar no banco de
produção nem na dinâmica de treino:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```
enem_rl_tutor/
├── agent/                       # Domínio de Reinforcement Learning (RL)
│   ├── __init__.py
│   ├── model.py                 # Classe da Rede Neural em PyTorch (nn.Module)
│   ├── replay_buffer.py         # Lógica de armazenamento de memória (Experience Replay)
│   ├── dqn_agent.py             # Política do Agente (escolha da ação, cálculo de perda)
│   ├── train.py                 # Loop principal dos episódios de treinamento
│   └── avaliar_robustez.py      # Avaliação da política treinada contra perfis de aluno (bots)
├── env/                         # Domínio do Ambiente Simulador
│   ├── __init__.py              # Expõe StudentEnvironment, KnowledgeGraph e os bots
│   ├── student_env.py           # Simulador: recebe ação, atualiza proficiência, devolve recompensa (caminho do DQN)
│   ├── knowledge_graph.py       # Regras do DAG de Matemática (propagação de pré-requisitos)
│   └── bots.py                  # Perfis simulados de alunos (ex: chutador, consistente)
├── api/                         # Domínio do Backend
│   ├── __init__.py
│   ├── main.py                  # Rotas FastAPI (receber estado e retornar recomendação)
│   └── schemas.py               # Modelos de validação de dados (Pydantic)
├── data/                        # Persistência
│   ├── raw/                     # Banco de dados de questões e desempenho
│   └── weights/                 # Pesos salvos do modelo PyTorch (.pt)
├── notebooks/                   # Experimentação e Análise
│   └── avaliacao_agente.ipynb   # Geração de gráficos de curva de aprendizado
├── tests/                       # Testes (unittest)
│   ├── __init__.py
│   └── test_env.py              # Domínio (KnowledgeGraph/bots) + integração do fluxo real
├── Dockerfile                   # Receita de containerização da aplicação
├── docker-compose.yml           # Orquestração para deploy da API e dependências
├── .gitignore                   # Exclusão de arquivos sensíveis e pesados
└── requirements.txt             # Dependências Python (torch, numpy, fastapi, etc.)
```
