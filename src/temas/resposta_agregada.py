"""Etapa 6 — Resposta para perguntas agregadas: panorama com contagens reais + exemplos citados.

O modelo recebe a contagem REAL de avaliações por macrotema no recorte e exemplos numerados [R#]
dos maiores temas. Ele deve copiar os números do panorama e ilustrar cada tema com os exemplos.
O código confere: todo número seguido de "%" ou "avaliações" na resposta precisa existir no
panorama ("numeros_fora_do_panorama"), além das verificações de citações e trechos de sempre.
"""
import os
import re

import pandas as pd

from src.rag.geracao import (FRASE_RECUSA, _chamar, _registro, classificar_trechos,
                             criar_cliente, descrever_filtros, encontrar_termos_frequencia,
                             extrair_citacoes)
from src.temas.panorama import Panorama

VERSAO = "agregada_v1"

PROMPT_AGREGADO = """Você é um analista de Voice of Customer de um e-commerce brasileiro.
A pergunta do gestor pede uma VISÃO DO CONJUNTO das avaliações. Você recebe:
- o PANORAMA: a contagem REAL de avaliações por tema no recorte. Os temas foram atribuídos
  automaticamente a TODAS as avaliações com texto do recorte (agrupamento por assunto),
  separadas em problemas (notas 1 a 3) e elogios (notas 4 e 5);
- EXEMPLOS numerados [R#] dos principais temas.

Regras obrigatórias:
1. Números: use SOMENTE os números do panorama, copiados exatamente como estão
   (ex.: "3.255 avaliações (22,5%)"). Não faça outras contagens, somas nem estimativas.
2. Apresente os temas do maior para o menor. Temas marcados "(fora do ranking)" reúnem
   avaliações mistas ou genéricas: informe a parcela deles, mas não os trate como um problema
   ou elogio.
3. Para cada tema principal, descreva o que os clientes relatam usando os exemplos, com trechos
   curtos e literais do texto do cliente entre aspas e a citação [R#]. Os exemplos ilustram o
   tema; não diga que representam todas as avaliações dele.
4. Relato x registro: o "Registro do sistema" (nota, entrega, dias de atraso...) não é fala do
   cliente; nunca o coloque entre aspas; ao usá-lo, diga "segundo o registro".
5. Causas: só mencione causas que os próprios clientes relatam nos exemplos.
6. Interpretação: apenas uma hipótese sobre o que foi apresentado, sem fatos ou números novos.
7. Os nomes Stark, Lannister, Targaryen e Baratheon são empresas anonimizadas.

FORMATO (Markdown, português do Brasil):
## Resposta
2 a 3 frases com os principais temas e seus números. Mencione o recorte quando houver filtros.
## Panorama
Tabela com as colunas Tema | Avaliações | %, na ordem do panorama, incluindo as linhas fora do
ranking.
## O que os clientes relatam
- **Nome do tema:** o que os clientes relatam, com trecho literal entre aspas [R#]
## Interpretação
1 a 2 frases de hipótese.
## Limitações
Uma frase dizendo que os temas foram atribuídos automaticamente (podem conter erros de
classificação), que a separação entre problemas e elogios usa a nota e que só entram avaliações
com texto."""

# Número seguido de "%" ou de "avaliações": tem de existir no panorama
NUMERO_COM_UNIDADE = re.compile(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)\s*(%|avalia)",
                                flags=re.IGNORECASE)


def _inteiro(n: int) -> str:
    return f"{n:,}".replace(",", ".")  # 3255 -> "3.255"


def _porcentagem(p: float) -> str:
    return f"{p:.1f}".replace(".", ",") + "%"  # 22.5 -> "22,5%"


def numeros_permitidos(panoramas: list[dict]) -> set[str]:
    permitidos = set()
    for p in panoramas:
        permitidos.add(_inteiro(p["total"]))
        for t in p["temas"]:
            permitidos |= {_inteiro(t["n"]), _porcentagem(t["pct"])}
            if t["pct"] == int(t["pct"]):
                permitidos.add(f"{int(t['pct'])}%")  # "7,0%" escrito como "7%"
    return permitidos


def numeros_fora_do_panorama(texto: str, permitidos: set[str]) -> list[str]:
    corpo = texto.split("## Limitações")[0]
    fora = []
    for numero, unidade in NUMERO_COM_UNIDADE.findall(corpo):
        token = numero + ("%" if unidade == "%" else "")
        if token not in permitidos:
            fora.append(token)
    return fora


def _bloco_panorama(p: dict) -> str:
    nome = ("PROBLEMAS (avaliações com nota 1 a 3)" if p["grupo"] == "problema"
            else "ELOGIOS (avaliações com nota 4 e 5)")
    linhas = [f"{nome}: {_inteiro(p['total'])} avaliações com texto no recorte"]
    for t in p["temas"]:
        marca = " (fora do ranking)" if t["mistos"] else ""
        linhas.append(f"- {t['nome']}{marca}: {_inteiro(t['n'])} avaliações "
                      f"({_porcentagem(t['pct'])})")
    return "\n".join(linhas)


def _bloco_exemplos(exemplos: pd.DataFrame) -> str:
    return "\n\n".join(f"[{e['ref']}] Tema: {e['tema_nome']} | Registro do sistema: "
                       f'{_registro(e)}\nTexto do cliente: "{e["texto"]}"'
                       for _, e in exemplos.iterrows())


def gerar_resposta_agregada(pergunta: str, filtros: dict | None, grupo: str,
                            panorama: Panorama, modelo: str | None = None
                            ) -> tuple[dict | None, pd.DataFrame]:
    """Calcula o(s) panorama(s), chama o LLM e devolve (resultado, exemplos citáveis)."""
    grupos = ["problema", "elogio"] if grupo == "ambos" else [grupo]
    panoramas = [p for p in (panorama.calcular(filtros, g) for g in grupos) if p["total"] > 0]
    if not panoramas:
        return None, pd.DataFrame()

    exemplos = pd.concat([p["exemplos"] for p in panoramas], ignore_index=True)
    exemplos.insert(0, "ref", [f"R{i + 1}" for i in range(len(exemplos))])
    exemplos["repeticoes"] = 1
    exemplos.attrs["documentos_no_filtro"] = sum(p["total"] for p in panoramas)
    exemplos.attrs["filtros"] = filtros or {}

    mensagem = (f"Pergunta do gestor: {pergunta}\n\n"
                f"Recorte consultado: {descrever_filtros(filtros)}.\n\n"
                "PANORAMA (contagem real de todas as avaliações com texto do recorte):\n"
                + "\n\n".join(_bloco_panorama(p) for p in panoramas)
                + "\n\nEXEMPLOS:\n" + _bloco_exemplos(exemplos))
    modelo = modelo or os.getenv("LLM_MODELO")
    resposta, duracao = _chamar(criar_cliente("groq"), modelo,
                                [{"role": "system", "content": PROMPT_AGREGADO},
                                 {"role": "user", "content": mensagem}])
    texto = (resposta.choices[0].message.content or "").strip()
    citacoes = extrair_citacoes(texto)
    validas = set(exemplos["ref"])
    resultado = {
        "modelo": modelo, "versao_prompt": VERSAO, "modo": "agregada", "grupo": grupo,
        "resposta": texto, "recusou": FRASE_RECUSA in texto,
        "citacoes": sorted(set(citacoes), key=lambda r: int(r[1:])),
        "citacoes_invalidas": sorted({c for c in citacoes if c not in validas}),
        **classificar_trechos(texto, exemplos, pergunta),
        "termos_frequencia": encontrar_termos_frequencia(texto),
        "contagens": [],  # aqui os números vêm do panorama; quem confere é a linha abaixo
        "numeros_fora_do_panorama": numeros_fora_do_panorama(texto, numeros_permitidos(panoramas)),
        "panorama": [{k: p[k] for k in ("grupo", "total", "temas")} for p in panoramas],
        "tokens_entrada": resposta.usage.prompt_tokens if resposta.usage else None,
        "tokens_saida": resposta.usage.completion_tokens if resposta.usage else None,
        "segundos": round(duracao, 1),
    }
    return resultado, exemplos