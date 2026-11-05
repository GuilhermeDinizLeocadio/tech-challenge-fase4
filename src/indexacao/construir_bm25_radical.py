"""Etapa 5 — Índice BM25 com radicais (stemming em português) para a busca híbrida corrigida.

Usa a mesma ordem de documentos do índice BM25 original (indice/bm25_ids.parquet), para que as
posições continuem apontando para os mesmos review_id. Leva poucos segundos.

Uso (na raiz do projeto):
    python -m src.indexacao.construir_bm25_radical
"""
import sys
import time

import bm25s
import pandas as pd
import Stemmer

from src.rag.recuperacao import CORPUS, INDICE

IDIOMA = "portuguese"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    ids = pd.read_parquet(INDICE / "bm25_ids.parquet")["review_id"]
    corpus = pd.read_parquet(CORPUS).set_index("review_id").loc[ids]
    stemmer = Stemmer.Stemmer(IDIOMA)

    print("Exemplos de radicais:",
          dict(zip(["quebrado", "quebrados", "quebrou", "correio", "correios", "atrasou"],
                   stemmer.stemWords(["quebrado", "quebrados", "quebrou",
                                      "correio", "correios", "atrasou"]))))

    tokens = bm25s.tokenize(corpus["chave_texto"].tolist(), stopwords=None,
                            stemmer=stemmer, show_progress=False)
    indice = bm25s.BM25()
    indice.index(tokens, show_progress=False)
    indice.save(str(INDICE / "bm25_radical"))
    print(f"Índice com radicais salvo em indice/bm25_radical: {len(corpus):,} documentos, "
          f"{time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()