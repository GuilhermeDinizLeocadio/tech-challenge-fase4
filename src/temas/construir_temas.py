"""Etapa 6 — Camada de temas: agrupa as avaliações por assunto e dá nome a cada grupo.

Os temas vêm dos DADOS, não de uma lista definida antes: as avaliações são agrupadas pela
semelhança do que dizem, e só depois o LLM dá nome aos grupos. Assim, um assunto que ninguém
imaginou também pode virar tema (evita o viés de confirmação visto na reescrita da Etapa 5).

1. Vetores: os mesmos embeddings da busca, lidos do índice (um por avaliação).
2. Dois grupos pela nota: "problema" (nota 1 a 3) e "elogio" (nota 4 e 5). O embedding capta
   assunto, não polaridade (Etapa 2); separar pela nota evita juntar "chegou antes do prazo" com
   "atrasou muito".
3. Agrupamento: K-means com K = 20 por grupo (definido antes de ver os resultados), sobre os
   textos distintos, com peso igual ao número de repetições.
4. Nome: o juiz (Gemini) lê as 15 avaliações mais típicas e 10 sorteadas de cada grupo e
   devolve nome, descrição e se o grupo é coerente.

Saídas: data/processed/temas.parquet (tema de cada avaliação) e resultados/temas.json (catálogo).

Uso (na raiz do projeto):
    python -m src.temas.construir_temas
"""
import json
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.avaliacao.juiz import Juiz
from src.rag.recuperacao import CORPUS, INDICE

RAIZ = Path(__file__).resolve().parents[2]
SAIDA_TEMAS = RAIZ / "data" / "processed" / "temas.parquet"
CATALOGO = RAIZ / "resultados" / "temas.json"
K_POR_GRUPO = 20
SEMENTE = 42
N_TIPICOS, N_SORTEADOS = 15, 10
GRUPOS = {
    "problema": (lambda nota: nota <= 3, "notas 1 a 3 (problemas e insatisfações)"),
    "elogio": (lambda nota: nota >= 4, "notas 4 e 5 (elogios e satisfação)"),
}

PROMPT_NOME = """Você nomeia temas de avaliações de clientes de um e-commerce brasileiro.
Você recebe avaliações de um mesmo grupo (as mais típicas e algumas sorteadas) e devolve:
- "nome": 2 a 5 palavras em português, descrevendo o assunto comum (ex.: "Atraso na entrega",
  "Produto diferente do anunciado", "Elogio à entrega rápida");
- "descricao": uma frase sobre o que os clientes relatam neste grupo;
- "coerente": true se a maior parte das avaliações (típicas e sorteadas) trata do mesmo assunto;
  false se o grupo mistura assuntos diferentes.
Use apenas o que está nas avaliações. Os nomes Stark, Lannister, Targaryen e Baratheon são
empresas anonimizadas.
Responda SOMENTE com JSON: {"nome": "...", "descricao": "...", "coerente": true}"""


def carregar_vetores(ids: list[str]) -> np.ndarray:
    """Lê do Chroma o vetor de cada avaliação, na ordem de `ids`."""
    colecao = chromadb.PersistentClient(path=str(INDICE / "chroma")).get_collection("avaliacoes")
    vetores = {}
    total = colecao.count()
    for inicio in range(0, total, 5000):
        lote = colecao.get(limit=5000, offset=inicio, include=["embeddings"])
        vetores.update(zip(lote["ids"], lote["embeddings"]))
    faltando = [i for i in ids if i not in vetores]
    if faltando:
        raise RuntimeError(f"{len(faltando)} avaliações sem vetor no índice")
    return np.asarray([vetores[i] for i in ids], dtype=np.float32)


def nomear(juiz: Juiz, descricao_grupo: str, tipicos: list[str], sorteados: list[str]) -> dict:
    corpo = ("Mais típicas:\n" + "\n".join(f"- {t}" for t in tipicos)
             + "\n\nSorteadas:\n" + "\n".join(f"- {t}" for t in sorteados))
    r = juiz.avaliar_json(PROMPT_NOME, f"Grupo: avaliações com {descricao_grupo}\n\n{corpo}")
    return {"nome": str(r.get("nome", "")).strip() or "Sem nome",
            "descricao": str(r.get("descricao", "")).strip(),
            "coerente": bool(r.get("coerente", False))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    inicio = time.perf_counter()
    corpus = pd.read_parquet(CORPUS, columns=["review_id", "chave_texto", "texto", "nota",
                                              "situacao_entrega"])
    print(f"Lendo {len(corpus):,} vetores do índice...", flush=True)
    vetores = carregar_vetores(corpus["review_id"].tolist())
    corpus["linha"] = np.arange(len(corpus))
    juiz = Juiz()

    atribuicoes, catalogo = [], []
    for grupo, (filtro, descricao_grupo) in GRUPOS.items():
        sub = corpus[filtro(corpus["nota"])]
        distintos = (sub.groupby("chave_texto")
                        .agg(texto=("texto", "first"), linha=("linha", "first"),
                             n=("review_id", "size"))
                        .reset_index())
        X = vetores[distintos["linha"].to_numpy()]
        print(f"\n{grupo.upper()}: {len(sub):,} avaliações, {len(distintos):,} textos distintos. "
              f"Agrupando em {K_POR_GRUPO} temas...", flush=True)

        modelo = MiniBatchKMeans(n_clusters=K_POR_GRUPO, random_state=SEMENTE, n_init=5,
                                 batch_size=4096)
        modelo.fit(X, sample_weight=distintos["n"].to_numpy())
        rotulos = modelo.predict(X)
        centros = modelo.cluster_centers_ / np.linalg.norm(modelo.cluster_centers_, axis=1,
                                                           keepdims=True)
        distintos["cluster"] = rotulos
        distintos["similaridade"] = (X * centros[rotulos]).sum(axis=1)  # proximidade ao centro

        # Temas numerados do maior para o menor (em avaliações, contando repetições)
        tamanhos = distintos.groupby("cluster")["n"].sum().sort_values(ascending=False)
        prefixo = "P" if grupo == "problema" else "E"
        ids_tema = {c: f"{prefixo}{i + 1:02d}" for i, c in enumerate(tamanhos.index)}
        distintos["tema"] = distintos["cluster"].map(ids_tema)

        mapa = distintos.set_index("chave_texto")[["tema", "similaridade"]]
        atrib = sub[["review_id", "chave_texto", "nota", "situacao_entrega"]].join(
            mapa, on="chave_texto")
        atrib["grupo"] = grupo
        atribuicoes.append(atrib)

        for c, id_tema in ids_tema.items():
            membros = distintos[distintos["cluster"] == c]
            mais_tipicos = membros.nlargest(N_TIPICOS, "similaridade")
            resto = membros.drop(mais_tipicos.index)
            sorteados = (resto.sample(min(N_SORTEADOS, len(resto)), random_state=SEMENTE)
                         ["texto"].tolist() if len(resto) else [])
            avaliacoes = atrib[atrib["tema"] == id_tema]
            item = {
                "id": id_tema, "grupo": grupo,
                **nomear(juiz, descricao_grupo, mais_tipicos["texto"].tolist(), sorteados),
                "n_avaliacoes": int(len(avaliacoes)),
                "pct_do_grupo": round(100 * len(avaliacoes) / len(sub), 1),
                "n_textos_distintos": int(len(membros)),
                "nota_media": round(float(avaliacoes["nota"].mean()), 2),
                "pct_entrega_com_problema": round(100 * avaliacoes["situacao_entrega"]
                                                  .isin(["atrasado", "não entregue"]).mean(), 1),
                "exemplos": mais_tipicos["texto"].head(5).tolist(),
            }
            catalogo.append(item)
            print(f"[{id_tema}] {item['nome'][:42]:<42} {item['n_avaliacoes']:>6,} "
                  f"({item['pct_do_grupo']:>4}%) | nota {item['nota_media']:.2f} | "
                  f"entrega c/ problema {item['pct_entrega_com_problema']:>5}%"
                  f"{'' if item['coerente'] else ' | MISTO'}", flush=True)

    SAIDA_TEMAS.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(atribuicoes).drop(columns=["chave_texto"]).to_parquet(SAIDA_TEMAS, index=False)
    CATALOGO.write_text(json.dumps({"k_por_grupo": K_POR_GRUPO, "semente": SEMENTE,
                                    "temas": catalogo}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n{len(catalogo)} temas em {time.perf_counter() - inicio:.0f} s. "
          f"Catálogo: {CATALOGO.relative_to(RAIZ)} | atribuições: "
          f"{SAIDA_TEMAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()