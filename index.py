import os
from pathlib import Path

import faiss
import pandas as pd

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

load_dotenv()

## utilizar variaveis de ambiente, veja .env.example
MODEL_REPO = os.getenv("MODEL_REPO") ## repositório no hugging face
MODEL_NAME = os.getenv("MODEL_NAME") ## nome do arquivo no hf
HF_TOKEN = os.getenv("HF_TOKEN") ## token do hf, não obrigatório mas acelera o download do modelo

## Limite de mensagens no historico, para não estourar o contexto
MAX_MSGS = 20

if not MODEL_REPO or not MODEL_NAME:
    print(
        "Por favor, defina as variáveis de ambiente "
        "MODEL_REPO e MODEL_NAME no arquivo .env"
    )
    exit(1)

print("Carregando base de conhecimento...")

## Carrega CSV
df = pd.read_csv("./dataset/archive/vgsales-small.csv")

documentos = []

for _, row in df.iterrows():
    documentos.append(
        f"""
Jogo: {row['Name']}
Plataforma: {row['Platform']}
Ano: {row['Year']}
Gênero: {row['Genre']}
Publicadora: {row['Publisher']}
Vendas América do Norte: {row['NA_Sales']} milhões
Vendas Europa: {row['EU_Sales']} milhões
Vendas Japão: {row['JP_Sales']} milhões
Vendas Outras Regiões: {row['Other_Sales']} milhões
Vendas Globais: {row['Global_Sales']} milhões
"""
    )

## Embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embeddings = embedding_model.encode(
    documentos,
    convert_to_numpy=True,
    show_progress_bar=True
)

## Índice vetorial

dim = embeddings.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(embeddings)

print(f"{len(documentos)} registros indexados.")

## Download modelo
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
    n_ctx=262144,
    n_threads=os.cpu_count(),
    verbose=False,
)

print(f"Modelo {modelo.name} carregado com sucesso!\n")

system_prompt = (
    "Você é um especialista em videogames. "
    "Responda sempre em português do Brasil. "
    "Utilize as informações fornecidas no contexto quando disponíveis."
    "Considere que o rank se refere ao campo 'Rank' do arquivo CSV"
)

historico = [
    {
        "role": "system",
        "content": system_prompt
    }
]

print("Chatbot iniciado! Digite 'sair' para encerrar.\n")
while True:
    print("----------------------------------------------------")
    pergunta = input("Você: ")

    if pergunta.strip().lower() == "sair":
        print("Chat encerrado.")
        break

    print("Buscando informações relevantes...")

    ## Busca vetorial
    query_embedding = embedding_model.encode(
        [pergunta],
        convert_to_numpy=True
    )

    distancias, indices = index.search(
        query_embedding,
        k=5
    )

    contexto = "\n\n".join(
        documentos[i]
        for i in indices[0]
    )

    # Guarda apenas a pergunta original
    historico.append({
        "role": "user",
        "content": pergunta
    })

    # Cria uma cópia temporária para esta consulta
    mensagens = historico.copy()

    # Injeta o contexto apenas nesta chamada
    mensagens.append({
        "role": "system",
        "content": f"""
    Informações recuperadas da base de conhecimento:

    {contexto}

    Utilize essas informações para responder à próxima pergunta.
    Caso o contexto não contenha a resposta, informe isso explicitamente.
    """
    })

    print("Bot está pensando...")

    resposta = llm.create_chat_completion(
        messages=mensagens,
        temperature=0.7,
        max_tokens=512,
    )

    ## A reposta será sempre a primeira escolha
    texto = resposta["choices"][0]["message"]["content"]

    historico.append({ "role": "assistant", "content": texto})

    if len(historico) > MAX_MSGS:
        historico = [historico[0]] + historico[-(MAX_MSGS - 1):]

    print(f"Bot: {texto}\n")