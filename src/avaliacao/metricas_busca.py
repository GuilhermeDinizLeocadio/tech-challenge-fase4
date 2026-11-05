"""Métricas da busca, calculadas com os rótulos de relevância do juiz (Etapas 4 e 5).

Para cada pergunta e cada configuração de busca, pega as K avaliações que seriam enviadas ao LLM
e calcula (relevante = nota 2 do juiz):
  precisao@K  fração das K avaliações que são relevantes
  hit@5       se há ao menos uma relevante entre as 5 primeiras
  mrr         1 / posição da primeira relevante (0 se nenhuma)
  recall@K    relevantes nas K / relevantes entre todos os candidatos rotulados (relativo ao pool)
Para perguntas que devem ser recusadas, interessam as "iscas": avaliações de nota 1 (mesmo
assunto, mas não respondem). Nas configurações com reranker, também registra a nota do reranker
na 1ª avaliação (para calibrar a recusa) e o tempo de cada busca.

Uso (na raiz do projeto):
    python -m src.avaliacao.metricas_busca
"""
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.avaliacao.rotular_relevancia import carregar_rotulos
from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.recuperacao import Buscador

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "metricas_busca.csv"
K = 20
CONFIGURACOES = {
    "densa": {"modo": "densa"},
    "hibrida_ingenua": {"modo": "hibrida"},
    "densa_rerank": {"modo": "densa", "reranquear": True},
    "hibrida_rerank": {"modo": "hibrida", "reranquear": True},
}


def calcular(notas: list[int], relevantes_no_pool: int) -> dict:
    relevantes = [n == 2 for n in notas]
    primeira = next((i + 1 for i, r in enumerate(relevantes) if r), None)
    return {
        "precisao@20": sum(relevantes) / len(notas) if notas else 0.0,
        "hit@5": float(any(relevantes[:5])),
        "mrr": 1 / primeira if primeira else 0.0,
        "recall@20": sum(relevantes) / relevantes_no_pool if relevantes_no_pool else float("nan"),
        "relevantes_top20": sum(relevantes),
        "iscas_top20": sum(n == 1 for n in notas),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = carregar_perguntas()
    rotulos = carregar_rotulos()

    por_pergunta = defaultdict(Counter)
    for (id_pergunta, _), nota in rotulos.items():
        por_pergunta[id_pergunta][nota] += 1

    print("Carregando o buscador (o reranker é carregado na primeira busca que o usar)...",
          flush=True)
    buscador = Buscador()
    linhas, sem_rotulo = [], 0
    for p in perguntas:
        relevantes_no_pool = por_pergunta[p["id"]][2]
        for nome, opcoes in CONFIGURACOES.items():
            inicio = time.perf_counter()
            evid = buscador.buscar(p["pergunta"], filtros=p.get("filtros"), k_final=K, **opcoes)
            segundos = time.perf_counter() - inicio
            notas = []
            for chave in evid["chave_texto"]:
                nota = rotulos.get((p["id"], chave))
                if nota is None:
                    sem_rotulo += 1
                    nota = 0
                notas.append(nota)
            score_top1 = (float(evid["score_rerank"].iloc[0])
                          if "score_rerank" in evid.columns and len(evid) else float("nan"))
            linhas.append({"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                           "agregada": p.get("agregada", False), "config": nome,
                           "segundos_busca": segundos, "score_top1": score_top1,
                           **calcular(notas, relevantes_no_pool)})
        print(f"[{p['id']}] ok", flush=True)

    tabela = pd.DataFrame(linhas)
    SAIDA.parent.mkdir(exist_ok=True)
    tabela.to_csv(SAIDA, index=False, encoding="utf-8")
    if sem_rotulo:
        print(f"\nAviso: {sem_rotulo} avaliações sem rótulo foram tratadas como irrelevantes.")

    metricas = ["precisao@20", "hit@5", "mrr", "recall@20"]
    responder = tabela[tabela["esperado"] != "recusar"]
    print("\nPERGUNTAS A RESPONDER (média por configuração):")
    print(responder.groupby("config", sort=False)[metricas].mean().round(3).to_string())

    print("\nPrecisão@20 por tipo de pergunta:")
    print(responder.pivot_table(index="tipo", columns="config", values="precisao@20",
                                aggfunc="mean").round(3).to_string())

    print("\nPERGUNTAS A RECUSAR (média por pergunta nas 20 enviadas ao LLM):")
    recusar = tabela[tabela["esperado"] == "recusar"]
    print(recusar.groupby("config", sort=False)[["relevantes_top20", "iscas_top20"]]
          .mean().round(1).to_string())

    print("\nTempo médio por busca (segundos):")
    print(tabela.groupby("config", sort=False)["segundos_busca"].mean().round(2).to_string())

    com_rerank = tabela[tabela["config"].str.contains("rerank")]
    print("\nNota do reranker na 1ª avaliação, por comportamento esperado:")
    print(com_rerank.groupby(["config", "esperado"])["score_top1"]
          .describe()[["count", "min", "mean", "max"]].round(3).to_string())

    print(f"\nDetalhe por pergunta salvo em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()