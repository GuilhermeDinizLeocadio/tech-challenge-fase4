"""Etapa 4 — Valida o conjunto de perguntas de teste antes de usá-lo na avaliação.

Confere: ids únicos, tipos e gabaritos válidos, campos e categorias dos filtros, tamanho de cada
recorte e similaridade da avaliação mais parecida. Para as perguntas que devem ser RECUSADAS ou
RESSALVADAS, mostra as 3 avaliações mais parecidas, para conferência humana do gabarito.

Uso (na raiz do projeto):
    python -m src.avaliacao.validar_perguntas
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.rag.recuperacao import Buscador, filtro_para_mascara

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO = RAIZ / "data" / "avaliacao" / "perguntas.json"
TIPOS = {"enunciado", "especifica", "filtro", "fora_escopo", "sem_evidencia", "evidencia_escassa"}
ESPERADOS = {"responder", "recusar", "ressalvar"}


def carregar_perguntas() -> list[dict]:
    return json.loads(ARQUIVO.read_text(encoding="utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = carregar_perguntas()
    print(f"{len(perguntas)} perguntas em {ARQUIVO.name}. Carregando o buscador...\n", flush=True)
    buscador = Buscador()
    corpus = buscador.corpus
    categorias = set(corpus["categoria_principal"])

    problemas = []
    ids = [p["id"] for p in perguntas]
    if len(ids) != len(set(ids)):
        problemas.append("Há ids repetidos.")

    linhas = []
    for p in perguntas:
        filtros = p.get("filtros") or {}
        if p["tipo"] not in TIPOS:
            problemas.append(f"{p['id']}: tipo inválido '{p['tipo']}'")
        if p["esperado"] not in ESPERADOS:
            problemas.append(f"{p['id']}: esperado inválido '{p['esperado']}'")
        campos_invalidos = [c for c in filtros if c not in corpus.columns]
        for c in campos_invalidos:
            problemas.append(f"{p['id']}: campo de filtro inexistente '{c}'")
        categoria = filtros.get("categoria_principal")
        if isinstance(categoria, str) and categoria not in categorias:
            problemas.append(f"{p['id']}: categoria inexistente '{categoria}'")

        recorte, sim_top1 = None, None
        if not campos_invalidos:
            recorte = int(filtro_para_mascara(corpus, filtros).sum())
            if recorte > 0:
                evid = buscador.buscar(p["pergunta"], filtros=filtros, k_final=3)
                sim_top1 = round(float(evid["sim_densa"].iloc[0]), 3)
        linhas.append({"id": p["id"], "tipo": p["tipo"], "agregada": p.get("agregada", False),
                       "esperado": p["esperado"], "recorte": recorte, "sim_top1": sim_top1})

    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))

    print("\nSimilaridade da 1ª avaliação, por comportamento esperado:")
    print(tabela.groupby("esperado")["sim_top1"].describe()[["count", "min", "mean", "max"]].round(3))

    print("\nPROBLEMAS ENCONTRADOS:" if problemas else "\nNenhum problema de estrutura encontrado.")
    for problema in problemas:
        print(f"  - {problema}")

    print("\n" + "=" * 80)
    print("CONFERÊNCIA HUMANA: perguntas que devem ser RECUSADAS ou RESSALVADAS.")
    print("Confira se as avaliações mais parecidas confirmam o comportamento esperado.")
    print("=" * 80)
    for p in perguntas:
        if p["esperado"] == "responder":
            continue
        filtros = p.get("filtros") or {}
        print(f"\n[{p['id']}] {p['pergunta']}  | filtros: {filtros or '—'}")
        if int(filtro_para_mascara(corpus, filtros).sum()) == 0:
            print("    (recorte vazio: nenhuma avaliação com texto)")
            continue
        for _, e in buscador.buscar(p["pergunta"], filtros=filtros, k_final=3).iterrows():
            print(f"    sim {e['sim_densa']:.3f} | nota {e['nota']} | {e['texto'][:110]}")


if __name__ == "__main__":
    main()