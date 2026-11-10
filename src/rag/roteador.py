"""Etapa 6 — Roteador: decide se a pergunta pede uma visão do conjunto (agregada) ou é sobre um
assunto específico. Na dúvida, ou se a chamada falhar, a rota é "específica" (o fluxo validado
na Etapa 5).
"""
import json
import re

from src.rag.geracao import criar_cliente

MODELO = "qwen/qwen3.8-27b"
GRUPOS = {"problema", "elogio", "ambos"}

PROMPT_ROTEADOR = """Classifique a pergunta de um gestor sobre avaliações de clientes de um
e-commerce brasileiro.
- "agregada": pede uma VISÃO DO CONJUNTO das avaliações: principais problemas ou elogios, temas,
  padrões, o que mais aparece, ranking, visão geral de um recorte.
  Ex.: "Quais são os principais problemas relatados?", "Quais padrões aparecem nas avaliações
  de clientes insatisfeitos?", "O que os clientes mais elogiam nesta categoria?"
- "especifica": pergunta sobre um ASSUNTO DETERMINADO, mesmo que peça "o que os clientes
  relatam". Ex.: "O que os clientes relatam sobre atrasos?", "Como os clientes descrevem a
  embalagem?", "O que dizem sobre a loja Lannister?", "Qual o faturamento da empresa?"
Se for agregada, informe o grupo: "problema" (reclamações, insatisfação, notas baixas),
"elogio" (elogios, satisfação) ou "ambos" (quando a pergunta não define).
Responda SOMENTE com JSON: {"tipo": "agregada", "grupo": "problema"}"""


def classificar_pergunta(pergunta: str, cliente=None, modelo: str = MODELO) -> dict:
    """Devolve {"tipo": "agregada" | "especifica", "grupo": "problema" | "elogio" | "ambos" | None}."""
    cliente = cliente or criar_cliente("groq")
    resposta = cliente.chat.completions.create(
        model=modelo, temperature=0, max_tokens=300,
        messages=[{"role": "system", "content": PROMPT_ROTEADOR},
                  {"role": "user", "content": pergunta}])
    texto = re.sub(r"<think>.*?</think>", "", resposta.choices[0].message.content or "",
                   flags=re.S)
    try:
        dados = json.loads(texto[texto.find("{"):texto.rfind("}") + 1])
    except ValueError:
        dados = {}
    if dados.get("tipo") != "agregada":
        return {"tipo": "especifica", "grupo": None}
    grupo = dados.get("grupo")
    return {"tipo": "agregada", "grupo": grupo if grupo in GRUPOS else "ambos"}