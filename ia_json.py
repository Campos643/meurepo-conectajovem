import json
import os
import requests

CHAVE = os.environ["AI_GATEWAY_API_KEY"]
URL = "https://ai-gateway.vercel.sh/v1/chat/completions"

INSTRUCAO = """Você é um assistente de estudos.
Leia o texto do usuário e responda SOMENTE com um JSON neste formato:
{"resumo": "resumo em até 2 frases",
 "palavras_chave": ["palavra1", "palavra2", "palavra3"],
 "nivel": "básico, intermediário ou avançado"}"""

texto = input("Cole um texto de estudo: ")

resposta = requests.post(
    URL,
    headers={"Authorization": f"Bearer {CHAVE}"},
    json={
        "model": "google/gemini-3.5-flash-lite",
        "messages": [
            {"role": "system", "content": INSTRUCAO},
            {"role": "user", "content": texto},
        ],
        "temperature": 0.2,
    },
    timeout=60,
)
resposta.raise_for_status()

conteudo = resposta.json()["choices"][0]["message"]["content"]
conteudo = conteudo.replace("```json", "").replace("```", "").strip()

dados = json.loads(conteudo)          # texto JSON vira dicionário Python
print("Resumo:", dados["resumo"])
print("Palavras-chave:", ", ".join(dados["palavras_chave"]))
print("Nível:", dados["nivel"])