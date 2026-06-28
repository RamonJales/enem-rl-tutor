# enem-rl-tutor

Sistema Tutor Inteligente (ITS) com Aprendizado por Reforço Profundo (DQN) para trilha adaptativa de Matemática do ENEM. Inspirações teóricas: Zona de Desenvolvimento Proximal (Vygotsky), Teoria de Resposta ao Item (TRI) e Metacognição (Modelo Aberto do Aluno).

## Visão geral

O agente DQN escolhe a melhor **Ação Pedagógica** a cada passo:

- `Avançar` — apresenta um conceito sucessor no grafo de pré-requisitos (DAG).
- `Reforçar` — mantém o aluno no mesmo conceito (mais prática).
- `Remediar` — retrocede para um pré-requisito do conceito atual.

A recompensa é orientada à meta: premia ganho de proficiência, domínio de novos conceitos e sondagem eficiente (redução do erro de crença Bayesiana).

## Requisitos

- Python 3.10+
- Dependências listadas em [requirements.txt](requirements.txt): `torch`, `numpy`, `matplotlib`, `sqlalchemy`, `fastapi`, `uvicorn`, `pydantic`

## Instalação

```bash
# (opcional) crie e ative um ambiente virtual
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# instale as dependências
pip install -r requirements.txt

# instale o servidor ASGI com suporte a recarga automática
pip install "uvicorn[standard]"
```

## Como rodar

Execute sempre a partir da raiz do projeto (`enem_rl_tutor/`) usando a flag `-m`.

---

### 1. Criar e popular o banco de dados

Gera `data/enem_tutor.db` com o grafo de 12 conceitos, **180 questões objetivas** (A/B/C/D) e o estado inicial de proficiência do aluno:

```bash
python -m data.database_setup
```

> ⚠️ Se o banco já existir, ele é **recriado do zero** (seed fixo — reprodutível).

---

### 2. Treinar o agente DQN

```bash
python -m agent.train
```

Roda 500 episódios com o `StudentEnvironment`. Salva em `data/weights/`:
- `dqn_policy.pt` — pesos da política treinada
- `recompensa_vs_episodios.png` — curva de aprendizado

> Os hiperparâmetros (episódios, `epsilon`, `batch_size`, etc.) ficam no topo de [`agent/train.py`](agent/train.py).

---

### 3. Rodar a plataforma web (frontend)

A plataforma é uma SPA (Single-Page Application) servida pela própria API FastAPI.
O agente DQN carregado recomenda questões em tempo real para o aluno real.

#### Iniciar o servidor

```bash
python -m uvicorn api.main:app --reload --port 8000
```

#### Acessar no navegador

```
http://localhost:8000
```

#### Credenciais de acesso (demo)

| Usuário | Senha     |
|---------|-----------|
| `aluno` | `enem2024`|
| `demo`  | `demo`    |
| `admin` | `admin`   |

#### Rotas da API (OpenAPI disponível em `/docs`)

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/auth/login` | Autenticação |
| `POST` | `/api/auth/logout` | Encerrar sessão |
| `GET`  | `/api/questao/proxima` | DQN seleciona próxima questão (A/B/C/D) |
| `POST` | `/api/questao/responder` | Registra resposta e atualiza crença Bayesiana |
| `GET`  | `/api/desempenho` | Dados de desempenho para os gráficos |
| `GET`  | `/api/conceitos` | Proficiências por conceito |
| `GET`  | `/api/health` | Status do servidor |

> O modelo DQN (`data/weights/dqn_policy.pt`) é carregado automaticamente. Se não existir, o servidor opera com uma heurística de fallback (sem necessidade de treino prévio).

---

### 4. Avaliar a robustez da política

```bash
python -m agent.avaliar_robustez
```

Avalia a política gulosa contra três perfis de aluno (50 episódios cada). Gera `docs/figuras/avaliacao_robustez.png`.

---

### 5. Rodar os testes

```bash
python -m unittest discover -s tests -v
```

---

## Estrutura do projeto

```
enem_rl_tutor/
├── agent/
│   ├── model.py                 # Rede neural DQN (PyTorch)
│   ├── replay_buffer.py         # Experience Replay
│   ├── dqn_agent.py             # Política do agente (select_action, optimize)
│   ├── train.py                 # Loop de treinamento (500 episódios)
│   └── avaliar_robustez.py      # Avaliação out-of-distribution
├── env/
│   ├── student_env.py           # Simulador do aluno (estilo Gym)
│   ├── knowledge_graph.py       # Grafo DAG de pré-requisitos
│   └── bots.py                  # Perfis de aluno (consistente, chutador)
├── api/
│   ├── main.py                  # Backend FastAPI (serve frontend + API REST)
│   └── schemas.py               # Schemas Pydantic
├── frontend/
│   └── index.html               # SPA (login · dashboard · trilha adaptativa)
├── data/
│   ├── database_setup.py        # Schema SQLAlchemy + seed (180 questões objetivas)
│   ├── enem_tutor.db            # SQLite (gerado pelo passo 1)
│   └── weights/
│       └── dqn_policy.pt        # Pesos treinados (gerado pelo passo 2)
├── docs/
│   ├── RELATORIO_TECNICO.md
│   └── figuras/
├── tests/
│   └── test_env.py
└── requirements.txt
```

## Fluxo resumido

```
[1] database_setup  →  enem_tutor.db  (grafo + 180 questões A/B/C/D)
[2] agent.train     →  dqn_policy.pt  (política treinada)
[3] uvicorn         →  localhost:8000 (plataforma web + API)
```
