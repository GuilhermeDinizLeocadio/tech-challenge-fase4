"""Etapas 5 e 6 — O fluxo completo de uma pergunta.

1. Roteador (Etapa 6): a pergunta pede uma visão do conjunto (agregada) ou é específica?
2a. Agregada: panorama com a contagem real dos macrotemas no recorte + exemplos → resposta.
2b. Específica: reescrita → busca densa → reranker → resposta (prompt v4) → correção guiada.

Função única usada pela linha de comando e, na Etapa 8, pela interface. Cada etapa auxiliar
degrada com elegância: sem roteador, a rota é "específica"; sem reescrita, a busca usa só a
pergunta; sem correção, fica a resposta original. O sistema não para.
"""
import pandas as pd

from src.rag.geracao import VERSAO_PADRAO, corrigir_resposta, gerar_resposta
from src.rag.recuperacao import Buscador
from src.rag.reescrita import montar_consulta, reescrever_pergunta
from src.rag.roteador import classificar_pergunta


def responder(buscador: Buscador, pergunta: str, filtros: dict | None = None, k: int = 20,
              reescrever: bool = True, reranquear: bool = True, modo: str = "densa",
              modelo: str | None = None, versao_prompt: str = VERSAO_PADRAO,
              corrigir: bool = True, rotear: bool = True, panorama=None) -> dict:
    """Devolve {"rota", "reescrita", "evidencias", "resultado"} (resultado = None se o recorte
    for vazio). `panorama` pode ser passado já carregado, para não recarregar a cada pergunta."""
    rota = {"tipo": "especifica", "grupo": None}
    if rotear:
        try:
            rota = classificar_pergunta(pergunta)
        except Exception as erro:  # degradação elegante: segue pela rota específica
            print(f"Aviso: roteador indisponível ({erro}); usando a rota específica.", flush=True)

    if rota["tipo"] == "agregada":
        from src.temas.panorama import Panorama
        from src.temas.resposta_agregada import gerar_resposta_agregada
        panorama = panorama or Panorama()
        resultado, exemplos = gerar_resposta_agregada(pergunta, filtros, rota["grupo"],
                                                      panorama, modelo=modelo)
        return {"rota": rota, "reescrita": "", "evidencias": exemplos, "resultado": resultado}

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
        return {"rota": rota, "reescrita": reescrita, "evidencias": evidencias, "resultado": None}

    resultado = gerar_resposta(pergunta, evidencias, modelo=modelo, versao_prompt=versao_prompt)
    if corrigir:
        try:
            resultado = corrigir_resposta(pergunta, evidencias, resultado, modelo=modelo)
        except Exception as erro:  # degradação elegante: fica a resposta original
            print(f"Aviso: correção indisponível ({erro}); mantendo a resposta original.",
                  flush=True)
    return {"rota": rota, "reescrita": reescrita, "evidencias": evidencias, "resultado": resultado}