# chatbot-llm

## Variáveis de ambiente (.env)
HF_TOKEN=[token do hugging face para download dos modelos]
MODEL_REPO=[nome do repositório no hugging face]
MODEL_NAME=[nome do modelo no hugging face]

## Virtual python env
python3 -m venv venv

## install dependencies through requirements

1. Defina venv como source

Linux:
`source ./venv/bin/activate`

Windows:
`venv\Scripts\activate`

2. Execute o comando `pip install`:

`source venv/bin/activate && pip install -r requirements.txt`

## Executar

1. Defina venv como source

Linux:
`source ./venv/bin/activate`

Windows:
`venv\Scripts\activate`

2. Execute o comando:
`source venv/bin/activate && python ./index.py`

Nota: Na primeira execução, pode demorar devido ao download do modelo no HF