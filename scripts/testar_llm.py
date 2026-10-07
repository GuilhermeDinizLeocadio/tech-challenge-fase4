"""Etapa 0 — testa a conectividade com os provedores de LLM via cliente compatível com OpenAI.

Uso:
    python scripts/testar_llm.py groq              -> lista modelos disponíveis
    python scripts/testar_llm.py groq <modelo>     -> lista e testa geração
"""
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # carrega as variáveis do arquivo .env para a memória do programa

# provedor -> (endereço da API, nome da variável com a chave no .env)
PROVEDORES = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), None),
}


def testar(provedor: str, modelo: str | None = None) -> None:
    base_url, var_chave = PROVEDORES[provedor]
    chave = os.getenv(var_chave) if var_chave else "ollama"  # Ollama não exige chave
    if not chave:
        print(f"[{provedor}] {var_chave} vazio no .env")
        return

    cliente = OpenAI(base_url=base_url, api_key=chave)

    try:
        modelos = sorted(m.id for m in cliente.models.list())
        print(f"[{provedor}] {len(modelos)} modelos disponíveis:")
        for m in modelos:
            print("   ", m)
    except Exception as erro:  # listar não é essencial; segue para o teste
        print(f"[{provedor}] não foi possível listar modelos: {erro}")

    if not modelo:
        return

    inicio = time.perf_counter()
    resposta = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": "Em uma frase: o que é uma avaliação de cliente em e-commerce?"}],
        temperature=0,   # 0 = respostas mais estáveis, menos "criativas"
        max_tokens=300,  # folga para modelos que "pensam" antes de responder
    )
    duracao = time.perf_counter() - inicio
    texto = (resposta.choices[0].message.content or "").strip()
    print(f"\n[{provedor}] {modelo} em {duracao:.1f}s -> {texto}")
    if resposta.usage:
        print(f"    tokens: entrada {resposta.usage.prompt_tokens} | saída {resposta.usage.completion_tokens}")


if __name__ == "__main__":
    testar(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)