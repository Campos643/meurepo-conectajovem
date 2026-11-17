import math
import os
import requests

CHAVE = os.environ["AI_GATEWAY_API_KEY"]
BASE = "https://ai-gateway.vercel.sh/v1"
CABECALHOS = {"Authorization": f"Bearer {CHAVE}"}

# A "base de conhecimento": trechos baseados no edital do Conecta Jovem
DOCUMENTOS = [
    "A bolsa auxílio do Conecta Jovem é de R$ 500,00 por mês durante o curso.",
    "O aluno perde a bolsa se tiver 3 faltas sem justificativa, seguidas ou não.",
    "As aulas são presenciais, das 7h30 às 11h30, no Yumit Hub e na UFMT.",
    "Há vale-transporte (se necessário), auxílio alimentação e material gratuito.",
    "O programa atende jovens de 15 a 24 anos que moram na Baixada Cuiabana.",
    "No Projeto Integrador, cada aluno apresenta um MVP hospedado na nuvem.",
]


def embeddings(textos):
    r = requests.post(f"{BASE}/embeddings", headers=CABECALHOS, json={
        "model": "openai/text-embedding-3-small",
        "input": textos,
    }, timeout=60)
    r.raise_for_status()
    return [item["embedding"] for item in r.json()["data"]]


def similaridade(a, b):
    produto = sum(x * y for x, y in zip(a, b))
    tamanho_a = math.sqrt(sum(x * x for x in a))
    tamanho_b = math.sqrt(sum(y * y for y in b))
    return produto / (tamanho_a * tamanho_b)


vetores = embeddings(DOCUMENTOS)                 # 1. documentos viram vetores
pergunta = input("Pergunte sobre o Conecta Jovem: ")
v_pergunta = embeddings([pergunta])[0]           # 2. a pergunta vira vetor

# 3. ordena os documentos do mais parecido para o menos parecido
ranking = sorted(zip(DOCUMENTOS, vetores),
                 key=lambda par: similaridade(v_pergunta, par[1]),
                 reverse=True)
trechos = [texto for texto, _ in ranking[:2]]   # fica com os 2 melhores

# 4. monta o prompt com os trechos e pergunta ao modelo
contexto = "\n".join(f"- {t}" for t in trechos)
r = requests.post(f"{BASE}/chat/completions", headers=CABECALHOS, json={
    "model": "google/gemini-3.5-flash-lite",
    "messages": [
        {"role": "system", "content":
            "Responda usando SOMENTE as informações fornecidas. Se a resposta "
            "não estiver nelas, diga: 'Não encontrei essa informação.'"},
        {"role": "user", "content": f"Informações:\n{contexto}\n\nPergunta: {pergunta}"},
    ],
    "temperature": 0.2,
}, timeout=60)
r.raise_for_status()

print("\nResposta:", r.json()["choices"][0]["message"]["content"])
print("\nTrechos consultados:")
for t in trechos:
    print(" -", t)