"""Etapa 6 — Panorama de temas de um recorte: contagem REAL de avaliações por macrotema e exemplos.

Diferente da busca (que traz as avaliações mais parecidas com a pergunta), o panorama conta TODAS
as avaliações com texto do recorte, pelo macrotema atribuído na construção dos temas. Por isso as
frequências daqui são reais e podem ser informadas ao gestor.
"""
import json
from pathlib import Path

import pandas as pd

from src.rag.recuperacao import CORPUS, filtro_para_mascara

RAIZ = Path(__file__).resolve().parents[2]
TEMAS = RAIZ / "data" / "processed" / "temas.parquet"
CATALOGO = RAIZ / "resultados" / "temas.json"
MIN_PALAVRAS_EXEMPLO = 5  # exemplos curtos demais ("não recebi") ilustram pouco


class Panorama:
    """Carrega o catálogo e o tema de cada avaliação UMA vez; depois calcula quantos recortes quiser."""

    def __init__(self) -> None:
        catalogo = json.loads(CATALOGO.read_text(encoding="utf-8"))
        self.macros = {m["id"]: m for m in catalogo["macrotemas"]}
        temas = pd.read_parquet(TEMAS, columns=["review_id", "grupo", "macrotema", "similaridade"])
        self.base = pd.read_parquet(CORPUS).merge(temas, on="review_id", how="inner")

    def calcular(self, filtros: dict | None = None, grupo: str = "problema",
                 n_principais: int = 5, exemplos_por_tema: int = 3) -> dict:
        """Contagem por macrotema no recorte + exemplos típicos dos maiores temas."""
        recorte = self.base[filtro_para_mascara(self.base, filtros)]
        recorte = recorte[recorte["grupo"] == grupo]
        total = len(recorte)

        temas = []
        for id_macro, n in recorte["macrotema"].value_counts().items():
            m = self.macros[id_macro]
            temas.append({"id": id_macro, "nome": m["nome"], "mistos": bool(m["mistos"]),
                          "n": int(n), "pct": round(100 * n / total, 1)})
        principais = [t for t in temas if not t["mistos"]][:n_principais]

        exemplos = []
        for t in principais:
            do_tema = recorte[recorte["macrotema"] == t["id"]].drop_duplicates("chave_texto")
            longos = do_tema[do_tema["n_palavras"] >= MIN_PALAVRAS_EXEMPLO]
            fonte = longos if len(longos) >= exemplos_por_tema else do_tema
            for _, e in fonte.nlargest(exemplos_por_tema, "similaridade").iterrows():
                exemplos.append({**e.to_dict(), "tema_nome": t["nome"]})
        return {"grupo": grupo, "total": total, "temas": temas, "principais": principais,
                "exemplos": pd.DataFrame(exemplos)}