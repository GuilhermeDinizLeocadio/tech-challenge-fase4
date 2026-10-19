"""Etapa 3 — Geração: monta o contexto numerado e pede ao LLM uma resposta citada.

O LLM recebe só as evidências recuperadas, numeradas [R1], [R2]..., o recorte consultado
(filtros e tamanho) e regras que o obrigam a: usar apenas esse contexto, citar cada ponto,
não inventar frequências nem causas, separar relato do cliente de registro do sistema e
recusar quando as evidências não bastam. Depois, o código confere automaticamente:
citações inexistentes, trechos entre aspas que não existem nas avaliações, termos de
frequência proibidos e se houve recusa.

Prompt v2: ajustes feitos após a comparação gpt-oss-120b x qwen3.8-27b (Etapa 3).
"""
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from src.ingestao.construir_corpus import gerar_chave_texto
from src.rag.recuperacao import _condicoes

load_dotenv()

PROVEDORES = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "ollama": (os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), None),
}

# Ajustes específicos por modelo (o gpt-oss "pensa" antes de responder; "low" reduz o gasto)
PARAMETROS_EXTRAS = {
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}

FRASE_RECUSA = "As avaliações recuperadas não são suficientes para responder a esta pergunta."

# Termos que indicam frequência na base inteira, proibidos pela regra 4 (verificados no código)
TERMOS_FREQUENCIA = re.compile(
    r"\b(maioria|principalmente|frequentemente|frequente|recorrentes?|recorrência|sistêmic[oa]s?)\b",
    flags=re.IGNORECASE,
)

# Trecho entre aspas (retas ou curvas), opcionalmente seguido de uma citação [R#]
PADRAO_TRECHO = re.compile(r'[“"]([^”"]{3,}?)[”"]\s*(\[[^\]]+\])?')

PROMPT_SISTEMA = f"""Você é um analista de Voice of Customer de um e-commerce brasileiro.
Responda à pergunta do gestor usando SOMENTE as avaliações de clientes fornecidas no contexto.

Regras obrigatórias:
1. Use apenas as avaliações do contexto. Não use conhecimento externo nem suponha fatos.
2. Cada ponto deve conter UMA ideia e citar SOMENTE as avaliações que mencionam essa ideia,
   no formato [R1] ou [R2, R5]. Não repita a mesma ideia em pontos diferentes.
   Cite apenas referências que existem no contexto.
3. Sempre que possível, inclua um trecho curto e literal da avaliação entre aspas
   (até 12 palavras) como prova do ponto.
4. Frequência: você vê apenas uma amostra de avaliações escolhidas por semelhança com a
   pergunta, não a base inteira. NÃO use "a maioria", "principalmente", "frequentemente",
   "recorrente", "sistêmico" nem porcentagens. Prefira "N das avaliações recuperadas"
   (contando as que você cita) ou o campo "aparece Nx" do contexto.
5. Causas: só mencione a causa de um problema se a própria avaliação citada mencionar essa
   causa, e atribua ao cliente ("os clientes atribuem a quebra à embalagem [R3]").
   Nunca deduza causas por conta própria.
6. Relato x registro: o texto entre aspas no contexto é o RELATO do cliente; os campos do
   cabeçalho (nota, entrega, dias de atraso, categoria, UF, data) são o REGISTRO do sistema.
   Ao usar um dado do cabeçalho, diga que é registro ("segundo o registro, o pedido atrasou
   6 dias [R2]"). Se relato e registro divergirem, mostre os dois sem corrigir o cliente.
7. Os nomes Stark, Lannister, Targaryen e Baratheon são nomes de empresas anonimizados pela
   base (lojas, parceiros ou transportadoras). Trate-os como empresas, nunca como personagens.
8. Recorte: o contexto informa os filtros e a quantidade de avaliações consultadas. Quando
   houver filtros, comece a Resposta dizendo a que recorte ela se refere.
9. Interpretação: apenas uma hipótese analítica sobre o que já foi citado ("isso sugere...").
   Não introduza fatos novos, números novos nem causas não relatadas pelos clientes.
10. Escopo e evidência: se as avaliações do contexto não forem suficientes para responder, ou
    se a pergunta estiver fora do escopo (experiência de compra relatada por clientes:
    produto, entrega, atendimento, loja), use o FORMATO DE RECUSA e nada mais.

FORMATO NORMAL (Markdown, português do Brasil):
## Resposta
2 a 4 frases diretas, com citações. Mencione o recorte quando houver filtros.
## O que os clientes relatam
- **Rótulo curto:** uma única ideia, com trecho literal entre aspas quando possível [R#]
(3 a 6 pontos)
## Interpretação
1 a 3 frases de hipótese analítica, sem fatos novos.
## Limitações
Uma frase informando que a resposta usa N avaliações recuperadas de um recorte de M
avaliações com texto e que não mede frequências na base.

FORMATO DE RECUSA (use somente este, sem as outras seções):
## Resposta
{FRASE_RECUSA}
**Motivo:** uma frase explicando por quê.
**O que esta base consegue responder:** uma frase sugerindo perguntas sobre experiências de
compra (produto, entrega, atendimento, loja)."""

NOMES_OPERADORES = {"$eq": "=", "$ne": "≠", "$gt": ">", "$gte": "≥",
                    "$lt": "<", "$lte": "≤", "$in": "em"}


def descrever_filtros(filtros: dict | None) -> str:
    """Filtros em texto legível, ex.: 'categoria_principal = cama mesa banho; nota ≤ 2'."""
    condicoes = _condicoes(filtros)
    if not condicoes:
        return "sem filtros (todas as avaliações com texto)"
    return "; ".join(f"{campo} {NOMES_OPERADORES.get(op, op)} {valor}"
                     for campo, op, valor in condicoes)


def _descrever_entrega(linha: pd.Series) -> str:
    situacao = linha.get("situacao_entrega")
    dias = linha.get("dias_atraso")
    if situacao == "atrasado" and pd.notna(dias):
        return f"atrasado ({int(dias)} dias após o prazo)"
    return str(situacao)


def formatar_contexto(evidencias: pd.DataFrame) -> str:
    """Transforma as evidências em blocos numerados: cabeçalho (registro) + texto (relato)."""
    blocos = []
    for _, e in evidencias.iterrows():
        data = str(e["data_compra"])
        cabecalho = (f"[{e['ref']}] nota {e['nota']} | categoria: {e['categoria_principal']} | "
                     f"entrega: {_descrever_entrega(e)} | compra: {data[:4]}-{data[4:6]} | "
                     f"UF: {e['uf']}")
        if e["repeticoes"] > 1:
            cabecalho += f" | aparece {e['repeticoes']}x"
        blocos.append(f'{cabecalho}\n"{e["texto"]}"')
    return "\n\n".join(blocos)


def montar_mensagem_usuario(pergunta: str, evidencias: pd.DataFrame) -> str:
    total = evidencias.attrs.get("documentos_no_filtro")
    filtros = evidencias.attrs.get("filtros")
    tamanho = f"{total:,} avaliações com texto" if total is not None else "tamanho não informado"
    return (f"Pergunta do gestor: {pergunta}\n\n"
            f"Recorte consultado: {descrever_filtros(filtros)} ({tamanho}).\n"
            f"Evidências: {len(evidencias)} avaliações recuperadas, as mais parecidas com a "
            f"pergunta dentro do recorte.\n\n"
            f"{formatar_contexto(evidencias)}")


def extrair_citacoes(texto: str) -> list[str]:
    """Todas as referências citadas entre colchetes, ex.: [R1, R3] -> ['R1', 'R3']."""
    citacoes = []
    for grupo in re.findall(r"\[([^\]]+)\]", texto):
        citacoes.extend(re.findall(r"R\d+", grupo))
    return citacoes


def verificar_trechos(texto: str, evidencias: pd.DataFrame) -> list[str]:
    """Confere se cada trecho entre aspas existe de fato no texto das avaliações.

    - Trecho seguido de citação ("..." [R12]): procura nas avaliações citadas, o que pega
      inclusive trecho atribuído à avaliação errada.
    - Trecho sem citação: procura em todo o contexto, incluindo nomes de categoria.
    A comparação usa o texto padronizado (minúsculas, sem acento, sem pontuação), e trechos
    com reticências são conferidos por partes.
    """
    chaves = dict(zip(evidencias["ref"], evidencias["chave_texto"]))
    categorias = [gerar_chave_texto(c) for c in evidencias["categoria_principal"].unique()]
    contexto_todo = " | ".join(list(chaves.values()) + categorias)

    nao_encontrados = []
    for trecho, citacao in PADRAO_TRECHO.findall(texto):
        refs = [r for r in re.findall(r"R\d+", citacao or "") if r in chaves]
        alvo = " | ".join(chaves[r] for r in refs) if refs else contexto_todo
        partes = [gerar_chave_texto(p) for p in re.split(r"\.\.\.|…", trecho)]
        if any(len(p) >= 3 and p not in alvo for p in partes):
            nao_encontrados.append(trecho.strip())
    return nao_encontrados


def criar_cliente(provedor: str) -> OpenAI:
    base_url, var_chave = PROVEDORES[provedor]
    chave = os.getenv(var_chave) if var_chave else "ollama"
    if not chave:
        raise RuntimeError(f"{var_chave} não está definido no .env")
    return OpenAI(base_url=base_url, api_key=chave)


def gerar_resposta(pergunta: str, evidencias: pd.DataFrame, modelo: str | None = None,
                   provedor: str = "groq", tentativas: int = 3) -> dict:
    """Chama o LLM e devolve a resposta com métricas e verificações automáticas."""
    modelo = modelo or os.getenv("LLM_MODELO")
    cliente = criar_cliente(provedor)
    mensagens = [{"role": "system", "content": PROMPT_SISTEMA},
                 {"role": "user", "content": montar_mensagem_usuario(pergunta, evidencias)}]

    for tentativa in range(1, tentativas + 1):
        try:
            inicio = time.perf_counter()
            resposta = cliente.chat.completions.create(
                model=modelo, messages=mensagens, temperature=0, max_tokens=1500,
                extra_body=PARAMETROS_EXTRAS.get(modelo, {}),
            )
            duracao = time.perf_counter() - inicio
            break
        except RateLimitError:
            # Camada gratuita: ao estourar o limite por minuto, espera e tenta de novo
            espera = 20 * tentativa
            print(f"Limite de uso atingido; aguardando {espera}s (tentativa {tentativa}/{tentativas})")
            time.sleep(espera)
    else:
        raise RuntimeError("Limite de uso persistente; tente novamente mais tarde.")

    texto = (resposta.choices[0].message.content or "").strip()
    citacoes = extrair_citacoes(texto)
    validas = set(evidencias["ref"])
    return {
        "modelo": modelo,
        "resposta": texto,
        "recusou": FRASE_RECUSA in texto,
        "citacoes": sorted(set(citacoes), key=lambda r: int(r[1:])),
        "citacoes_invalidas": sorted({c for c in citacoes if c not in validas}),
        "trechos_nao_encontrados": verificar_trechos(texto, evidencias),
        "termos_frequencia": sorted({t.lower() for t in TERMOS_FREQUENCIA.findall(texto)}),
        "tokens_entrada": resposta.usage.prompt_tokens if resposta.usage else None,
        "tokens_saida": resposta.usage.completion_tokens if resposta.usage else None,
        "segundos": round(duracao, 1),
    }