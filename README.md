# enem-rl-tutor

The project abstracts learning into a dynamic process based on the student's interaction with mathematical concepts. Theoretical inspirations include the Zone of Proximal Development (Vygotsky), Item Response Theory (IRT), and the principles of Metacognition (Open Learner Models).

## Project structure

```
enem_rl_tutor/
├── agent/                       # Domínio de Reinforcement Learning (RL)
│   ├── __init__.py
│   ├── model.py                 # Classe da Rede Neural em PyTorch (nn.Module)
│   ├── replay_buffer.py         # Lógica de armazenamento de memória (Experience Replay)
│   ├── dqn_agent.py             # Política do Agente (escolha da ação, cálculo de perda)
│   └── train.py                 # Loop principal dos episódios de treinamento
├── env/                         # Domínio do Ambiente Simulador
│   ├── __init__.py
│   ├── student_env.py           # Simulador: recebe ação, atualiza proficiência, devolve recompensa
│   ├── knowledge_graph.py       # Regras do DAG de Matemática (pré-requisitos)
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
├── Dockerfile                   # Receita de containerização da aplicação
├── docker-compose.yml           # Orquestração para deploy da API e dependências
├── .gitignore                   # Exclusão de arquivos sensíveis e pesados
└── requirements.txt             # Dependências Python (torch, numpy, fastapi, etc.)
```
