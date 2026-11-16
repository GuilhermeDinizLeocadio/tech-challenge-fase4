"""Etapas 3 e 5 — Geração: monta o contexto numerado e pede ao LLM uma resposta citada.

O LLM recebe só as evidências recuperadas, numeradas [R1], [R2]..., o recorte consultado
(filtros e tamanho) e regras que o obrigam a: usar apenas esse contexto, citar cada ponto,
não inventar frequências nem causas, separar relato do cliente de registro do sistema e
recusar quando as evidências não bastam. Depois, o código confere automaticamente:
citações inexistentes, trechos entre aspas, termos de frequência, contagens e se houve recusa.

Versões do prompt e do contexto:
- v2 (Etapa 3): ajustes feitos após a comparação gpt-oss-120b x qwen3.8-27b.
- v3 (Etapa 5): o contexto rotula "Registro do sistema" e "Texto do cliente" e o prompt restringe
  as aspas ao texto do cliente, sem costurar frases (erro dominante encontrado na Etapa 4).
- v4 (Etapa 5, padrão): igual ao v3, mas proíbe contar avaliações ou clientes da amostra
  recuperada ("8 das avaliações"): com a busca melhor, as contagens cresceram, ficaram
  inverificáveis e, mesmo quando certas, eram frequência disfarçada.

Verificador de trechos:
- v1 (verificar_trechos, Etapa 3): mantido sem mudanças, para comparação.
- v2 (classificar_trechos, Etapa 5): ignora inserções editoriais [entre colchetes], separa
  trechos que são termos da própria pergunta e trechos do REGISTRO apresentados como fala do
  cliente.

Correção guiada pelo verificador (Etapa 5): se a resposta tiver termos de frequência ou
contagens, uma chamada extra pede ao modelo que reescreva SÓ essas expressões. O código aceita a
correção apenas se ela não piorar as citações nem os trechos e reduzir os problemas.
"""
import os
import re
import time
from difflib import SequenceMatcher

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
VERSAO_PADRAO = "v4"  # configuração final da Etapa 5

# Termos que indicam frequência na base inteira, proibidos pela regra 4 (verificados no código)
TERMOS_FREQUENCIA = re.compile(
    r"\b(maioria|principalmente|frequentemente|frequente|recorrentes?|recorrência|sistêmic[oa]s?)\b",
    flags=re.IGNORECASE,
)

# Contagens de avaliações ou clientes ("8 das avaliações", "três clientes") e porcentagens,
# proibidas no prompt v4: na amostra recuperada, contar é frequência disfarçada
PADRAO_CONTAGEM = re.compile(
    r"\b(?:\d+|dois|duas|três|tres|quatro|cinco|seis|sete|oito|nove|dez|onze|doze|treze|"
    r"quatorze|catorze|quinze|dezesseis|dezessete|dezoito|dezenove|vinte)\s+"
    r"(?:(?:das|dos|de)\s+)?(?:\d+\s+)?(?:avalia\w*|clientes|relatos|compradores)\b"
    r"|\b\d+(?:[.,]\d+)?\s*%",
    flags=re.IGNORECASE,
)

# Trecho entre aspas (retas ou curvas), opcionalmente seguido de uma citação [R#]
PADRAO_TRECHO = re.compile(r'[“"]([^”"]{3,}?)[”"]\s*(\[[^\]]+\])?')

# Inserção editorial dentro de uma citação, ex.: "produtos [são] originais"
PADRAO_INSERCAO = re.compile(r"\[[^\]]*\]")

# Semelhança mínima (0 a 1) para um trecho citado contar como literal:
# aceita erro de digitação corrigido ou palavra cortada completada; rejeita paráfrase
LIMIAR_SEMELHANCA_TRECHO = 0.85

REGRA_3 = {
    "v2": """3. Sempre que possível, inclua um trecho curto e literal da avaliação entre aspas
   (até 12 palavras) como prova do ponto.""",
    "v3": """3. Sempre que possível, inclua um trecho curto e literal do TEXTO DO CLIENTE entre aspas
   (até 12 palavras), copiado como está, sem juntar pedaços de frases diferentes.
   Use aspas SOMENTE para trechos do texto do cliente.""",
}
REGRA_3["v4"] = REGRA_3["v3"]

REGRA_4 = {
    "v2": """4. Frequência: você vê apenas uma amostra de avaliações escolhidas por semelhança com a
   pergunta, não a base inteira. NÃO use "a maioria", "principalmente", "frequentemente",
   "recorrente", "sistêmico" nem porcentagens. Prefira "N das avaliações recuperadas"
   (contando as que você cita) ou o campo "aparece Nx" do contexto.""",
    "v4": """4. Frequência e contagens: você vê apenas uma amostra de avaliações escolhidas por
   semelhança com a pergunta, não a base inteira; por isso, contar avaliações dessa amostra não
   mede nada. NÃO conte avaliações nem clientes ("8 das avaliações", "três clientes"), NÃO use
   porcentagens e NÃO use "a maioria", "principalmente", "frequentemente", "recorrente" nem
   "sistêmico". Descreva o que os clientes relatam e cite as avaliações. A única quantidade
   permitida é o campo "aparece Nx" do contexto, que conta textos idênticos no recorte.""",
}
REGRA_4["v3"] = REGRA_4["v2"]

REGRA_6 = {
    "v2": """6. Relato x registro: o texto entre aspas no contexto é o RELATO do cliente; os campos do
   cabeçalho (nota, entrega, dias de atraso, categoria, UF, data) são o REGISTRO do sistema.
   Ao usar um dado do cabeçalho, diga que é registro ("segundo o registro, o pedido atrasou
   6 dias [R2]"). Se relato e registro divergirem, mostre os dois sem corrigir o cliente.""",
    "v3": """6. Relato x registro: cada avaliação traz o "Registro do sistema" (nota, entrega, dias de
   atraso, categoria, UF, data) e o "Texto do cliente" (o relato). Nunca coloque dados do
   registro entre aspas nem os apresente como fala do cliente; ao usá-los, diga que são
   registro ("segundo o registro, o pedido atrasou 6 dias [R2]"). Se relato e registro
   divergirem, mostre os dois sem corrigir o cliente.""",
}
REGRA_6["v4"] = REGRA_6["v3"]


def _montar_prompt(versao: str) -> str:
    return f"""Você é um analista de Voice of Customer de um e-commerce brasileiro.
Responda à pergunta do gestor usando SOMENTE as avaliações de clientes fornecidas no contexto.

Regras obrigatórias:
1. Use apenas as avaliações do contexto. Não use conhecimento externo nem suponha fatos.
2. Cada ponto deve conter UMA ideia e citar SOMENTE as avaliações que mencionam essa ideia,
   no formato [R1] ou [R2, R5]. Não repita a mesma ideia em pontos diferentes.
   Cite apenas referências que existem no contexto.
{REGRA_3[versao]}
{REGRA_4[versao]}
5. Causas: só mencione a causa de um problema se a própria avaliação citada mencionar essa
   causa, e atribua ao cliente ("os clientes atribuem a quebra à embalagem [R3]").
   Nunca deduza causas por conta própria.
{REGRA_6[versao]}
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


PROMPTS = {versao: _montar_prompt(versao) for versao in ("v2", "v3", "v4")}
PROMPT_SISTEMA = PROMPTS["v2"]  # compatibilidade com o código das etapas anteriores

PROMPT_CORRECAO = """Você revisa respostas de um analista de Voice of Customer.
A resposta abaixo usa expressões de frequência ou contagens que não podem ser sustentadas, porque
as avaliações foram escolhidas por semelhança com a pergunta e não representam a base inteira.
Reescreva a resposta substituindo APENAS estas expressões: {lista}.
Como substituir: troque a afirmação de frequência por uma afirmação de EXISTÊNCIA, que diz que os
relatos existem sem dizer quantos. Nunca transforme a frase numa afirmação sobre todos.
Exemplos:
- "os clientes elogiam principalmente X" → "há clientes que elogiam X"
- "há queixas recorrentes sobre X" → "há queixas sobre X"
- "a maioria das entregas ocorreu no prazo" → "há relatos de entregas no prazo"
- "frequentemente associam X a Y" → "há clientes que associam X a Y"
- "8 das avaliações relatam X" → "há relatos de X"
Regras: mantenha exatamente as mesmas seções, os mesmos pontos, as mesmas citações [R#] e os
mesmos trechos entre aspas; não acrescente fatos, citações nem trechos novos.
Devolva somente a resposta reescrita, em Markdown."""

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


def _registro(e: pd.Series) -> str:
    """Os campos do REGISTRO do sistema de uma avaliação, em uma linha."""
    data = str(e["data_compra"])
    return (f"nota {e['nota']} | categoria: {e['categoria_principal']} | "
            f"entrega: {_descrever_entrega(e)} | compra: {data[:4]}-{data[4:6]} | UF: {e['uf']}")


def formatar_contexto(evidencias: pd.DataFrame, versao: str = VERSAO_PADRAO) -> str:
    """Transforma as evidências em blocos numerados: registro do sistema + relato do cliente.

    v2: cabeçalho + texto entre aspas. v3 e v4: os dois rotulados explicitamente.
    """
    blocos = []
    for _, e in evidencias.iterrows():
        repeticoes = f" | aparece {e['repeticoes']}x" if e["repeticoes"] > 1 else ""
        if versao in ("v3", "v4"):
            blocos.append(f"[{e['ref']}] Registro do sistema: {_registro(e)}{repeticoes}\n"
                          f'Texto do cliente: "{e["texto"]}"')
        else:
            blocos.append(f'[{e["ref"]}] {_registro(e)}{repeticoes}\n"{e["texto"]}"')
    return "\n\n".join(blocos)


def montar_mensagem_usuario(pergunta: str, evidencias: pd.DataFrame,
                            versao: str = VERSAO_PADRAO) -> str:
    total = evidencias.attrs.get("documentos_no_filtro")
    filtros = evidencias.attrs.get("filtros")
    tamanho = f"{total:,} avaliações com texto" if total is not None else "tamanho não informado"
    return (f"Pergunta do gestor: {pergunta}\n\n"
            f"Recorte consultado: {descrever_filtros(filtros)} ({tamanho}).\n"
            f"Evidências: {len(evidencias)} avaliações recuperadas, as mais parecidas com a "
            f"pergunta dentro do recorte.\n\n"
            f"{formatar_contexto(evidencias, versao)}")


def extrair_citacoes(texto: str) -> list[str]:
    """Todas as referências citadas entre colchetes, ex.: [R1, R3] -> ['R1', 'R3']."""
    citacoes = []
    for grupo in re.findall(r"\[([^\]]+)\]", texto):
        citacoes.extend(re.findall(r"R\d+", grupo))
    return citacoes


def _trecho_presente(parte: str, alvo: str) -> bool:
    """True se a parte aparece no alvo, exatamente ou com pequenas diferenças de grafia.

    Desliza uma "janela" do tamanho do trecho sobre o texto e mede a semelhança (0 a 1).
    """
    if parte in alvo:
        return True
    n = len(parte)
    melhor = max((SequenceMatcher(None, parte, alvo[i:i + n]).ratio()
                  for i in range(max(1, len(alvo) - n + 1))), default=0.0)
    return melhor >= LIMIAR_SEMELHANCA_TRECHO


def verificar_trechos(texto: str, evidencias: pd.DataFrame) -> list[str]:
    """VERIFICADOR v1 (Etapa 3), mantido sem mudanças para comparação.

    Confere se cada trecho entre aspas existe de fato no texto das avaliações.
    - Trecho seguido de citação ("..." [R12]): procura nas avaliações citadas, o que pega
      inclusive trecho atribuído à avaliação errada.
    - Trecho sem citação: procura em todo o contexto, incluindo nomes de categoria.
    A comparação usa o texto padronizado (minúsculas, sem acento, sem pontuação) e tolera
    pequenas diferenças de grafia (erro de digitação corrigido, palavra cortada completada),
    mas não paráfrases. Trechos com reticências são conferidos por partes.
    """
    chaves = dict(zip(evidencias["ref"], evidencias["chave_texto"]))
    categorias = [gerar_chave_texto(c) for c in evidencias["categoria_principal"].unique()]
    contexto_todo = " | ".join(list(chaves.values()) + categorias)

    nao_encontrados = []
    for trecho, citacao in PADRAO_TRECHO.findall(texto):
        refs = [r for r in re.findall(r"R\d+", citacao or "") if r in chaves]
        alvo = " | ".join(chaves[r] for r in refs) if refs else contexto_todo
        partes = [gerar_chave_texto(p) for p in re.split(r"\.\.\.|…", trecho)]
        if any(len(p) >= 3 and not _trecho_presente(p, alvo) for p in partes):
            nao_encontrados.append(trecho.strip())
    return nao_encontrados


def classificar_trechos(texto: str, evidencias: pd.DataFrame, pergunta: str = "") -> dict:
    """VERIFICADOR v2 (Etapa 5): classifica cada trecho entre aspas que não está no relato.

    Mesma comparação do v1 (texto padronizado, tolerância a grafia, partes separadas por
    reticências), com três refinamentos:
    - inserções editoriais [entre colchetes] são removidas antes de comparar;
    - trecho que é termo da própria pergunta (ex.: numa recusa) vai para "trechos_da_pergunta";
    - trecho que está no REGISTRO do sistema (cabeçalho) vai para "registro_como_relato".
    O que sobra são os "trechos_nao_encontrados" de fato.
    """
    chaves = dict(zip(evidencias["ref"], evidencias["chave_texto"]))
    registros = {e["ref"]: gerar_chave_texto(_registro(e)) for _, e in evidencias.iterrows()}
    categorias = [gerar_chave_texto(c) for c in evidencias["categoria_principal"].unique()]
    contexto_todo = " | ".join(list(chaves.values()) + categorias)
    registro_todo = " | ".join(registros.values())
    chave_pergunta = gerar_chave_texto(pergunta) if pergunta else ""

    resultado = {"trechos_nao_encontrados": [], "registro_como_relato": [],
                 "trechos_da_pergunta": []}
    for trecho, citacao in PADRAO_TRECHO.findall(texto):
        limpo = PADRAO_INSERCAO.sub(" ", trecho)
        partes = [gerar_chave_texto(p) for p in re.split(r"\.\.\.|…", limpo)]
        partes = [p for p in partes if len(p) >= 3]
        if not partes:
            continue
        refs = [r for r in re.findall(r"R\d+", citacao or "") if r in chaves]
        alvo = " | ".join(chaves[r] for r in refs) if refs else contexto_todo
        if all(_trecho_presente(p, alvo) for p in partes):
            continue  # trecho literal do relato: correto
        if chave_pergunta and all(p in chave_pergunta for p in partes):
            resultado["trechos_da_pergunta"].append(trecho.strip())
            continue
        alvo_registro = " | ".join(registros[r] for r in refs) if refs else registro_todo
        if all(_trecho_presente(p, alvo_registro) for p in partes):
            resultado["registro_como_relato"].append(trecho.strip())
        else:
            resultado["trechos_nao_encontrados"].append(trecho.strip())
    return resultado


def _texto_proprio(texto: str) -> str:
    """O texto escrito pelo modelo: sem a seção Limitações e sem os trechos entre aspas
    (a fala do cliente pode conter "100%" ou "frequente" sem que isso seja um erro)."""
    return PADRAO_TRECHO.sub(" ", texto.split("## Limitações")[0])


def encontrar_termos_frequencia(texto: str) -> list[str]:
    """Termos de frequência escritos pelo modelo (fora das aspas e das Limitações)."""
    return sorted({t.lower() for t in TERMOS_FREQUENCIA.findall(_texto_proprio(texto))})


def encontrar_contagens(texto: str) -> list[str]:
    """Contagens de avaliações/clientes e porcentagens escritas pelo modelo.

    Ignora recusas ("nenhuma das 20 avaliações menciona..." é o que uma recusa deve dizer),
    os trechos entre aspas e a seção Limitações.
    """
    if FRASE_RECUSA in texto:
        return []
    return [m.group(0) for m in PADRAO_CONTAGEM.finditer(_texto_proprio(texto))]


def problemas_de_frequencia(texto: str) -> list[str]:
    """Tudo o que a correção deve remover: termos de frequência e contagens."""
    return sorted(set(encontrar_termos_frequencia(texto)) | set(encontrar_contagens(texto)))


def criar_cliente(provedor: str) -> OpenAI:
    base_url, var_chave = PROVEDORES[provedor]
    chave = os.getenv(var_chave) if var_chave else "ollama"
    if not chave:
        raise RuntimeError(f"{var_chave} não está definido no .env")
    return OpenAI(base_url=base_url, api_key=chave)


def _chamar(cliente: OpenAI, modelo: str, mensagens: list[dict], tentativas: int = 3):
    """Uma chamada ao LLM, esperando e tentando de novo se o limite gratuito for atingido."""
    for tentativa in range(1, tentativas + 1):
        try:
            inicio = time.perf_counter()
            resposta = cliente.chat.completions.create(
                model=modelo, messages=mensagens, temperature=0, max_tokens=1500,
                extra_body=PARAMETROS_EXTRAS.get(modelo, {}),
            )
            return resposta, time.perf_counter() - inicio
        except RateLimitError:
            espera = 20 * tentativa
            print(f"Limite de uso atingido; aguardando {espera}s (tentativa {tentativa}/{tentativas})")
            time.sleep(espera)
    raise RuntimeError("Limite de uso persistente; tente novamente mais tarde.")


def _verificacoes(texto: str, evidencias: pd.DataFrame, pergunta: str) -> dict:
    """Todas as verificações automáticas de uma resposta."""
    citacoes = extrair_citacoes(texto)
    validas = set(evidencias["ref"])
    return {
        "recusou": FRASE_RECUSA in texto,
        "citacoes": sorted(set(citacoes), key=lambda r: int(r[1:])),
        "citacoes_invalidas": sorted({c for c in citacoes if c not in validas}),
        **classificar_trechos(texto, evidencias, pergunta),
        "termos_frequencia": encontrar_termos_frequencia(texto),
        "contagens": encontrar_contagens(texto),
    }


def gerar_resposta(pergunta: str, evidencias: pd.DataFrame, modelo: str | None = None,
                   provedor: str = "groq", versao_prompt: str = VERSAO_PADRAO) -> dict:
    """Chama o LLM e devolve a resposta com métricas e verificações automáticas."""
    modelo = modelo or os.getenv("LLM_MODELO")
    cliente = criar_cliente(provedor)
    mensagens = [{"role": "system", "content": PROMPTS[versao_prompt]},
                 {"role": "user",
                  "content": montar_mensagem_usuario(pergunta, evidencias, versao_prompt)}]
    resposta, duracao = _chamar(cliente, modelo, mensagens)
    texto = (resposta.choices[0].message.content or "").strip()
    return {
        "modelo": modelo,
        "versao_prompt": versao_prompt,
        "resposta": texto,
        **_verificacoes(texto, evidencias, pergunta),
        "tokens_entrada": resposta.usage.prompt_tokens if resposta.usage else None,
        "tokens_saida": resposta.usage.completion_tokens if resposta.usage else None,
        "segundos": round(duracao, 1),
    }


def corrigir_resposta(pergunta: str, evidencias: pd.DataFrame, resultado: dict,
                      modelo: str | None = None, provedor: str = "groq") -> dict:
    """Correção guiada pelo verificador: uma reescrita só das expressões de frequência.

    A correção só é aceita se, pelos critérios do código: as citações não aumentarem nem caírem
    mais de 20%; não surgir trecho inexistente nem registro citado como relato a mais; e o
    número de problemas de frequência diminuir. Caso contrário, fica a resposta original.
    """
    problemas = problemas_de_frequencia(resultado["resposta"])
    if resultado["recusou"] or not problemas:
        return {**resultado, "correcao": "não necessária"}

    modelo = modelo or resultado["modelo"]
    lista = ", ".join(f'"{p}"' for p in problemas)
    mensagens = [{"role": "system", "content": PROMPT_CORRECAO.format(lista=lista)},
                 {"role": "user", "content": resultado["resposta"]}]
    resposta, duracao = _chamar(criar_cliente(provedor), modelo, mensagens)
    novo = (resposta.choices[0].message.content or "").strip()
    novo = re.sub(r"<think>.*?</think>", "", novo, flags=re.S).strip()
    v = _verificacoes(novo, evidencias, pergunta)

    antes, depois = set(resultado["citacoes"]), set(v["citacoes"])
    aceita = (bool(novo) and "## Resposta" in novo and not v["recusou"]
              and depois <= antes and len(depois) >= 0.8 * len(antes)
              and len(v["trechos_nao_encontrados"]) <= len(resultado["trechos_nao_encontrados"])
              and len(v["registro_como_relato"]) <= len(resultado["registro_como_relato"])
              and len(problemas_de_frequencia(novo)) < len(problemas))
    if not aceita:
        return {**resultado, "correcao": "rejeitada", "problemas_antes": problemas}
    return {**resultado, **v, "resposta": novo, "resposta_original": resultado["resposta"],
            "correcao": "aplicada", "problemas_antes": problemas,
            "segundos_correcao": round(duracao, 1),
            "tokens_correcao": resposta.usage.total_tokens if resposta.usage else None}