"""Etapa 4 — Métricas da busca, calculadas com os rótulos de relevância do juiz.

Para cada pergunta e cada configuração de busca, pega as K avaliações que seriam enviadas ao LLM
e calcula (relevante = nota 2 do juiz):
  precisao@K  fração das K avaliações que são relevantes
  hit@5       se há ao menos uma relevante entre as 5 primeiras
  mrr         1 / posição da primeira relevante (0 se nenhuma)
  recall@K    relevantes nas K / relevantes entre todos os candidatos rotulados (relativo ao pool)
Para perguntas que devem ser recusadas, interessam as "iscas": avaliações de nota 1 (mesmo
assunto, mas não respondem) e eventuais notas 2 (que indicariam juiz generoso ou gabarito errado).

Uso (na raiz do projeto):
    python -m src.avaliacao.metricas_busca
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.avaliacao.rotular_relevancia import carregar_rotulos
from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.recuperacao import Buscador

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "metricas_busca.csv"
K = 20
CONFIGURACOES = {"densa": "densa", "hibrida_ingenua": "hibrida"}


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

    # 1) Conferência dos rótulos: perguntas que devem ser recusadas não deveriam ter notas 2
    por_pergunta = defaultdict(Counter)
    for (id_pergunta, _), nota in rotulos.items():
        por_pergunta[id_pergunta][nota] += 1
    print("Rótulos das perguntas que NÃO devem ser respondidas normalmente:")
    for p in perguntas:
        if p["esperado"] != "responder":
            c = por_pergunta[p["id"]]
            print(f"  [{p['id']}] {p['esperado']:9s} nota 2: {c[2]:3d} | nota 1: {c[1]:3d} | "
                  f"nota 0: {c[0]:3d} | {p['pergunta'][:60]}")

    # 2) Métricas por pergunta e configuração
    print("\nCarregando o buscador...", flush=True)
    buscador = Buscador()
    linhas, sem_rotulo = [], 0
    for p in perguntas:
        relevantes_no_pool = por_pergunta[p["id"]][2]
        for nome, modo in CONFIGURACOES.items():
            evid = buscador.buscar(p["pergunta"], filtros=p.get("filtros"), k_final=K, modo=modo)
            notas = []
            for chave in evid["chave_texto"]:
                nota = rotulos.get((p["id"], chave))
                if nota is None:
                    sem_rotulo += 1
                    nota = 0  # não deveria acontecer: as 20 primeiras estão entre os 50 rotulados
                notas.append(nota)
            linhas.append({"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                           "agregada": p.get("agregada", False), "config": nome,
                           **calcular(notas, relevantes_no_pool)})

    tabela = pd.DataFrame(linhas)
    SAIDA.parent.mkdir(exist_ok=True)
    tabela.to_csv(SAIDA, index=False, encoding="utf-8")
    if sem_rotulo:
        print(f"Aviso: {sem_rotulo} avaliações sem rótulo foram tratadas como irrelevantes.")

    metricas = ["precisao@20", "hit@5", "mrr", "recall@20"]
    responder = tabela[tabela["esperado"] != "recusar"]
    print("\nPERGUNTAS A RESPONDER (média por configuração):")
    print(responder.groupby("config")[metricas].mean().round(3).to_string())

    print("\nPrecisão@20 por tipo de pergunta:")
    print(responder.pivot_table(index="tipo", columns="config", values="precisao@20",
                                aggfunc="mean").round(3).to_string())

    print("\nPERGUNTAS A RECUSAR (média de avaliações por pergunta nas 20 enviadas ao LLM):")
    recusar = tabela[tabela["esperado"] == "recusar"]
    print(recusar.groupby("config")[["relevantes_top20", "iscas_top20"]].mean().round(1).to_string())

    print(f"\nDetalhe por pergunta salvo em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()