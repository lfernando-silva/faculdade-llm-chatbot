import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

load_dotenv()

## utilizar variaveis de ambiente, veja .env.example
MODEL_REPO = os.getenv("MODEL_REPO") ## repositório no hugging face
MODEL_NAME = os.getenv("MODEL_NAME") ## nome do arquivo no hf
HF_TOKEN = os.getenv("HF_TOKEN") ## token do hf, não obrigatório mas acelera o download do modelo

## Limite de mensagens no historico, para não estourar o contexto
MAX_MSGS = 20

prompt_base = (
    "Você é um assistente útil e educado. "
    "Sempre responda em portugês do Brasil, de forma clara e objetiva.\n\n"
    ## f"O seguinte conteúdo vem de um livro-texto relevante:\n{texto_pdf}\n\n"
)

historico = [
    {
        "role": "system",
        "content": prompt_base
    }
]

if not MODEL_REPO or not MODEL_NAME:
    print("Por favor, defina as variáveis de ambiente MODEL_REPO e MODEL_NAME no arquivo .env")
    exit(1)

## download do modelo no hugging face
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_NAME,
    ## essa pasta não vai ser comitada, serve para guardar o modelo baixado
    local_dir="models",
    token=HF_TOKEN,
)

print(f"Modelo baixado em: {model_path}")

modelo = Path(model_path).resolve()

# Carrega o modelo
llm = Llama(
    model_path=str(modelo),
    n_ctx=4096,
    n_threads=os.cpu_count(),
    verbose=False,
)

print(f"Modelo {modelo.name} carregado com sucesso!\n")

print("Chatbot iniciado! Digite 'sair' para encerrar.\n")
while True:
    print("----------------------------------------------------")
    pergunta = input("Você: ")

    if pergunta.strip().lower() == "sair":
        print("Chat encerrado.")
        break

    historico.append({ "role": "user",   "content": pergunta})

    print("Bot está pensando...")

    resposta = llm.create_chat_completion(
        messages=historico,
        temperature=0.7,
        max_tokens=512,
    )

    ## A reposta será sempre a primeira escolha
    texto = resposta["choices"][0]["message"]["content"]

    historico.append({ "role": "assistant", "content": texto})

    if len(historico) > MAX_MSGS:
        historico = [historico[0]] + historico[-(MAX_MSGS - 1):]

    print(f"Bot: {texto}\n")