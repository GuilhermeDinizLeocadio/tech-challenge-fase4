"""Etapa 2 — Constrói os índices de busca a partir de data/processed/corpus.parquet.

Gera em indice/:
- embeddings.npy + textos_unicos.parquet : vetores (float32, normalizados) de cada texto distinto
- chroma/                                 : banco vetorial, 1 documento por avaliação + metadados
- bm25/ + bm25_ids.parquet                : índice de palavras exatas (BM25)
- manifesto.json                          : modelo, quantidades, data e tempos da construção

O cálculo dos embeddings salva o progresso em partes: se for interrompido,
basta rodar de novo que ele continua de onde parou.

Uso (na raiz do projeto):
    python -m src.indexacao.construir_indice
"""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import bm25s
import chromadb
import numpy as np
import pandas as pd

from src.indexacao.embeddings import MODELO, carregar_modelo, preparar_pergunta

RAIZ = Path(__file__).resolve().parents[2]
CORPUS = RAIZ / "data" / "processed" / "corpus.parquet"
INDICE = RAIZ / "indice"
PARTES = INDICE / "embeddings_partes"

TAMANHO_PARTE = 2000        # textos por parte salva (~2 min cada na CPU)
TAMANHO_LOTE_CHROMA = 4000  # documentos enviados ao Chroma por vez
NOME_COLECAO = "avaliacoes"

METADADOS = ["nota", "situacao_entrega", "dias_atraso", "status_pedido", "categoria_principal",
             "categorias", "data_compra", "data_avaliacao", "uf", "regiao", "empresas_citadas",
             "n_palavras", "texto_curto", "chave_texto", "pedidos"]


def calcular_embeddings(unicos: pd.DataFrame) -> np.ndarray:
    """Embeddings dos textos distintos, salvando o progresso parte a parte."""
    PARTES.mkdir(parents=True, exist_ok=True)
    arquivo_ordem = PARTES / "ordem.parquet"

    # Se a lista de textos mudou desde a última execução, as partes salvas não valem mais
    if arquivo_ordem.exists():
        ordem_anterior = pd.read_parquet(arquivo_ordem)["chave_texto"]
        if not ordem_anterior.equals(unicos["chave_texto"]):
            print("A lista de textos mudou desde a última execução: recomeçando do zero.")
            shutil.rmtree(PARTES)
            PARTES.mkdir(parents=True)
    unicos[["chave_texto"]].to_parquet(arquivo_ordem, index=False)

    modelo = None  # só carrega o modelo se houver parte a calcular
    n_partes = int(np.ceil(len(unicos) / TAMANHO_PARTE))
    vetores = []
    for i in range(n_partes):
        arquivo = PARTES / f"parte_{i:03d}.npy"
        if arquivo.exists():  # parte já calculada numa execução anterior
            vetores.append(np.load(arquivo))
            continue
        if modelo is None:
            modelo = carregar_modelo()
        textos = unicos["texto"].iloc[i * TAMANHO_PARTE:(i + 1) * TAMANHO_PARTE].tolist()
        print(f"Parte {i + 1}/{n_partes} ({len(textos)} textos)")
        v = modelo.encode(textos, batch_size=64, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        np.save(arquivo, v)
        vetores.append(v)
    return np.vstack(vetores)


def _valor_nativo(valor):
    """Converte tipos do numpy (ex.: numpy.bool_) para tipos Python, que o Chroma aceita."""
    return valor.item() if hasattr(valor, "item") else valor


def montar_metadados(registro: dict) -> dict:
    """O Chroma não aceita campos vazios: campos ausentes (ex.: dias_atraso de pedido
    não entregue) simplesmente não entram nos metadados daquele documento."""
    return {campo: _valor_nativo(v) for campo, v in registro.items() if not pd.isna(v)}


def construir_chroma(corpus: pd.DataFrame, vetores_por_doc: np.ndarray):
    """Banco vetorial persistente: 1 documento por avaliação, com vetor e metadados."""
    cliente = chromadb.PersistentClient(path=str(INDICE / "chroma"))
    try:
        cliente.delete_collection(NOME_COLECAO)  # recria do zero: rodar 2x dá o mesmo resultado
    except Exception:
        pass  # na primeira execução a coleção ainda não existe
    colecao = cliente.create_collection(NOME_COLECAO)

    metadados = [montar_metadados(r) for r in corpus[METADADOS].to_dict("records")]
    for inicio in range(0, len(corpus), TAMANHO_LOTE_CHROMA):
        fim = inicio + TAMANHO_LOTE_CHROMA
        colecao.add(
            ids=corpus["review_id"].iloc[inicio:fim].tolist(),
            documents=corpus["texto"].iloc[inicio:fim].tolist(),
            embeddings=vetores_por_doc[inicio:fim],
            metadatas=metadados[inicio:fim],
        )
        print(f"Chroma: {min(fim, len(corpus)):,}/{len(corpus):,} documentos")
    return colecao


def construir_bm25(corpus: pd.DataFrame) -> None:
    """Índice de palavras exatas sobre a chave_texto (minúsculas, sem acento, sem pontuação).

    Sem remover stopwords: o próprio BM25 já dá peso baixo a palavras muito comuns,
    e listas de stopwords em português incluem "não", que muda o sentido do texto.
    """
    tokens = bm25s.tokenize(corpus["chave_texto"].tolist(), stopwords=None, show_progress=False)
    bm25 = bm25s.BM25()
    bm25.index(tokens, show_progress=False)
    bm25.save(str(INDICE / "bm25"))
    # O BM25 identifica documentos pela posição; esta tabela traduz posição -> review_id
    corpus[["review_id"]].to_parquet(INDICE / "bm25_ids.parquet", index=False)


def teste_rapido(colecao) -> None:
    """Uma busca de verificação, para ver o índice respondendo."""
    modelo = carregar_modelo()
    pergunta = "O produto chegou quebrado ou danificado?"
    vetor = modelo.encode([preparar_pergunta(pergunta)], normalize_embeddings=True)
    resultado = colecao.query(query_embeddings=vetor, n_results=5)
    print(f"\nTeste rápido — pergunta: '{pergunta}'")
    for texto, meta, distancia in zip(resultado["documents"][0], resultado["metadatas"][0],
                                      resultado["distances"][0]):
        # Vetores normalizados: similaridade de cosseno = 1 - distância/2
        print(f"  [sim {1 - distancia / 2:.3f}] nota {meta['nota']} | {texto[:90]}")


def main() -> None:
    inicio_total = time.perf_counter()
    INDICE.mkdir(exist_ok=True)

    corpus = pd.read_parquet(CORPUS)
    unicos = (corpus.drop_duplicates("chave_texto")[["chave_texto", "texto"]]
              .reset_index(drop=True))
    print(f"{len(corpus):,} documentos | {len(unicos):,} textos distintos para os embeddings")

    t0 = time.perf_counter()
    vetores_unicos = calcular_embeddings(unicos)
    tempo_embeddings = time.perf_counter() - t0
    np.save(INDICE / "embeddings.npy", vetores_unicos)
    unicos.to_parquet(INDICE / "textos_unicos.parquet", index=False)

    # Cada documento recebe o vetor do seu texto distinto (textos idênticos = mesmo vetor)
    posicao = pd.Series(np.arange(len(unicos)), index=unicos["chave_texto"])
    vetores_por_doc = vetores_unicos[posicao.loc[corpus["chave_texto"]].to_numpy()]

    t0 = time.perf_counter()
    colecao = construir_chroma(corpus, vetores_por_doc)
    tempo_chroma = time.perf_counter() - t0

    t0 = time.perf_counter()
    construir_bm25(corpus)
    tempo_bm25 = time.perf_counter() - t0

    assert colecao.count() == len(corpus), "Chroma não recebeu todos os documentos"

    manifesto = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo_embedding": MODELO,
        "dimensao": int(vetores_unicos.shape[1]),
        "precisao": "float32",
        "documentos": len(corpus),
        "textos_distintos": len(unicos),
        "distancia_chroma": "l2 em vetores normalizados (similaridade = 1 - distancia/2)",
        "bm25": "sobre chave_texto (minúsculas, sem acento, sem pontuação), sem stopwords",
        "tempo_segundos": {"embeddings": round(tempo_embeddings), "chroma": round(tempo_chroma),
                           "bm25": round(tempo_bm25),
                           "total": round(time.perf_counter() - inicio_total)},
    }
    (INDICE / "manifesto.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\nManifesto:")
    print(json.dumps(manifesto, ensure_ascii=False, indent=2))

    teste_rapido(colecao)


if __name__ == "__main__":
    main()