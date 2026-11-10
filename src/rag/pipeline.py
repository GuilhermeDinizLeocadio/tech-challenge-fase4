"""Etapa 5 — O fluxo completo de uma pergunta: reescrita → busca → reranker → resposta → correção.

Função única usada pela linha de comando e, na Etapa 8, pela interface, para que todos os
caminhos usem exatamente o mesmo fluxo. Configuração padrão = a final da Etapa 5: busca densa,
consulta reescrita na linguagem dos clientes, reranker, prompt v4 e correção guiada pelo
verificador.

Se a reescrita falhar (limite de uso, servidor indisponível), a busca continua só com a pergunta
original; se a correção falhar, fica a resposta original. O sistema não para.
"""
import pandas as pd

from src.rag.geracao import VERSAO_PADRAO, corrigir_resposta, gerar_resposta
from src.rag.recuperacao import Buscador
from src.rag.reescrita import montar_consulta, reescrever_pergunta


def responder(buscador: Buscador, pergunta: str, filtros: dict | None = None, k: int = 20,
              reescrever: bool = True, reranquear: bool = True, modo: str = "densa",
              modelo: str | None = None, versao_prompt: str = VERSAO_PADRAO,
              corrigir: bool = True) -> dict:
    """Devolve {"reescrita", "evidencias", "resultado"} (resultado = None se o recorte for vazio)."""
    reescrita = ""
    if reescrever:
        try:
            reescrita = reescrever_pergunta(pergunta)
        except Exception as erro:  # degradação elegante: segue sem reescrita
            print(f"Aviso: reescrita indisponível ({erro}); buscando só pela pergunta.",
                  flush=True)

    evidencias: pd.DataFrame = buscador.buscar(
        pergunta, filtros=filtros, k_final=k, modo=modo, reranquear=reranquear,
        consulta=montar_consulta(pergunta, reescrita))
    if evidencias.empty:
        return {"reescrita": reescrita, "evidencias": evidencias, "resultado": None}

    resultado = gerar_resposta(pergunta, evidencias, modelo=modelo, versao_prompt=versao_prompt)
    if corrigir:
        try:
            resultado = corrigir_resposta(pergunta, evidencias, resultado, modelo=modelo)
        except Exception as erro:  # degradação elegante: fica a resposta original
            print(f"Aviso: correção indisponível ({erro}); mantendo a resposta original.",
                  flush=True)
    return {"reescrita": reescrita, "evidencias": evidencias, "resultado": resultado}