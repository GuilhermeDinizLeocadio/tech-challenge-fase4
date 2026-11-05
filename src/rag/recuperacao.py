"""Recuperação: busca por sentido (Chroma), por palavras (BM25) e reranker (cross-encoder).

Modos de busca:
- "densa"   (padrão): só busca por sentido. Linha de base da Etapa 3.
- "hibrida": sentido + BM25 fundidos por RRF (Reciprocal Rank Fusion), que combina POSIÇÕES em
             vez de notas, porque cada busca tem uma escala própria.

Reranker (opcional, Etapa 5): um cross-encoder lê a pergunta JUNTO com cada um dos 50 primeiros
candidatos e dá uma nota de relevância; os candidatos são reordenados por essa nota antes da
escolha dos K finais. É mais preciso que a comparação de vetores, mas lento demais para a base
inteira, por isso atua só sobre os finalistas.

Em todos os casos, textos idênticos são colapsados num só (com a contagem de repetições).

Filtros usam uma linguagem única, traduzida para o Chroma e para o pandas (BM25):
    {"nota": {"$lte": 2}, "categoria_principal": "cama mesa banho"}
    {"data_compra": {"$gte": 20180101, "$lte": 20180331}}
Operadores aceitos: $eq, $ne, $gt, $gte, $lt, $lte, $in. Valor sem operador = $eq.
"""
import re
from pathlib import Path

import bm25s
import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.indexacao.embeddings import carregar_modelo, preparar_pergunta
from src.ingestao.construir_corpus import gerar_chave_texto

RAIZ = Path(__file__).resolve().parents[2]
INDICE = RAIZ / "indice"
CORPUS = RAIZ / "data" / "processed" / "corpus.parquet"

K_CANDIDATOS = 50  # quantos resultados cada busca traz antes da fusão (e quantos o reranker lê)
K_RRF = 60         # constante padrão do RRF (Cormack et al., 2009)
PADRAO_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # mesmo critério de "palavra" usado pelo bm25s

MODELO_RERANKER = "BAAI/bge-reranker-v2-m3"
LOTE_RERANKER = 16  # pares (pergunta, avaliação) processados por vez

COLUNAS_EVIDENCIA = ["texto", "nota", "categoria_principal", "situacao_entrega", "dias_atraso",
                     "data_compra", "uf", "empresas_citadas", "chave_texto"]


def _condicoes(filtros: dict | None) -> list[tuple[str, str, object]]:
    """Transforma o filtro numa lista de (campo, operador, valor)."""
    condicoes = []
    for campo, regra in (filtros or {}).items():
        if not isinstance(regra, dict):
            regra = {"$eq": regra}
        for operador, valor in regra.items():
            condicoes.append((campo, operador, valor))
    return condicoes


def filtro_para_chroma(filtros: dict | None) -> dict | None:
    """Traduz o filtro para o formato do Chroma (uma condição por item, unidas por $and)."""
    condicoes = [{campo: {op: valor}} for campo, op, valor in _condicoes(filtros)]
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


def filtro_para_mascara(df: pd.DataFrame, filtros: dict | None) -> pd.Series:
    """Traduz o filtro para uma máscara True/False do pandas (usada no BM25 e nas contagens)."""
    mascara = pd.Series(True, index=df.index)
    comparadores = {
        "$eq": lambda c, v: c == v, "$ne": lambda c, v: c != v,
        "$gt": lambda c, v: c > v, "$gte": lambda c, v: c >= v,
        "$lt": lambda c, v: c < v, "$lte": lambda c, v: c <= v,
        "$in": lambda c, v: c.isin(v),
    }
    for campo, op, valor in _condicoes(filtros):
        if op not in comparadores:
            raise ValueError(f"Operador de filtro não suportado: {op}")
        # Campos vazios (ex.: dias_atraso de pedido não entregue) não passam no filtro,
        # o mesmo comportamento do Chroma
        mascara &= comparadores[op](df[campo], valor).fillna(False).astype(bool)
    return mascara


class Buscador:
    """Carrega modelo e índices UMA vez; depois responde a quantas buscas forem necessárias."""

    def __init__(self) -> None:
        self.modelo = carregar_modelo()
        cliente = chromadb.PersistentClient(path=str(INDICE / "chroma"))
        self.colecao = cliente.get_collection("avaliacoes")
        self.bm25 = bm25s.BM25.load(str(INDICE / "bm25"))
        ids_bm25 = pd.read_parquet(INDICE / "bm25_ids.parquet")["review_id"]
        # Corpus na MESMA ordem do índice BM25 (o BM25 identifica documentos pela posição)
        self.corpus = pd.read_parquet(CORPUS).set_index("review_id").loc[ids_bm25]
        self._reranker = None  # carregado só no primeiro uso (~2 GB)

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            modelo = CrossEncoder(MODELO_RERANKER, device="cpu")
            getattr(modelo, "model", modelo).float()  # garante float32 na CPU
            self._reranker = modelo
        return self._reranker

    def pontuar_reranker(self, pergunta: str, candidatos: pd.DataFrame) -> pd.Series:
        """Nota do cross-encoder para cada par (pergunta, avaliação)."""
        pares = [(pergunta, texto) for texto in candidatos["texto"]]
        notas = self.reranker.predict(pares, batch_size=LOTE_RERANKER, show_progress_bar=False)
        return pd.Series(np.asarray(notas, dtype=float), index=candidatos.index)

    def busca_densa(self, pergunta: str, filtros: dict | None = None,
                    k: int = K_CANDIDATOS) -> pd.DataFrame:
        """Busca por SENTIDO no Chroma (pergunta com a instrução do e5-instruct)."""
        vetor = self.modelo.encode([preparar_pergunta(pergunta)], normalize_embeddings=True)
        res = self.colecao.query(query_embeddings=vetor, n_results=k,
                                 where=filtro_para_chroma(filtros), include=["distances"])
        ids = res["ids"][0]
        return pd.DataFrame({
            "review_id": ids,
            "sim_densa": [1 - d / 2 for d in res["distances"][0]],  # vetores normalizados
            "pos_densa": range(1, len(ids) + 1),
        })

    def busca_bm25(self, pergunta: str, filtros: dict | None = None,
                   k: int = K_CANDIDATOS) -> pd.DataFrame:
        """Busca por PALAVRAS: a pergunta passa pela mesma padronização dos documentos."""
        tokens = PADRAO_TOKEN.findall(gerar_chave_texto(pergunta))
        vocabulario = getattr(self.bm25, "vocab_dict", None)
        if vocabulario is not None:  # ignora palavras que não existem em nenhuma avaliação
            tokens = [t for t in tokens if t in vocabulario]
        vazio = pd.DataFrame(columns=["review_id", "score_bm25", "pos_bm25"])
        if not tokens:
            return vazio

        pontuacoes = self.bm25.get_scores(tokens)  # nota para TODOS os documentos
        mascara = filtro_para_mascara(self.corpus, filtros).to_numpy()
        pontuacoes = np.where(mascara, pontuacoes, -np.inf)  # fora do filtro = nunca escolhido
        melhores = np.argsort(-pontuacoes)[:k]
        melhores = melhores[pontuacoes[melhores] > 0]  # precisa conter ao menos uma palavra
        return pd.DataFrame({
            "review_id": self.corpus.index[melhores],
            "score_bm25": pontuacoes[melhores],
            "pos_bm25": range(1, len(melhores) + 1),
        })

    def buscar(self, pergunta: str, filtros: dict | None = None, k_final: int = 20,
               modo: str = "densa", reranquear: bool = False) -> pd.DataFrame:
        """Recupera evidências, colapsa textos idênticos e, se pedido, reordena pelo reranker.

        modo="densa"   -> só busca por sentido (padrão)
        modo="hibrida" -> sentido + BM25 fundidos por RRF
        reranquear     -> os 50 primeiros candidatos são reordenados pelo cross-encoder
        """
        if modo not in ("densa", "hibrida"):
            raise ValueError(f"modo inválido: {modo}")
        densa = self.busca_densa(pergunta, filtros)
        if modo == "hibrida":
            lexical = self.busca_bm25(pergunta, filtros)
        else:
            lexical = pd.DataFrame(columns=["review_id", "score_bm25", "pos_bm25"])

        fundido = densa.merge(lexical, on="review_id", how="outer")
        fundido["rrf"] = ((1 / (K_RRF + fundido["pos_densa"])).fillna(0)
                          + (1 / (K_RRF + fundido["pos_bm25"])).fillna(0))
        fundido = (fundido.sort_values("rrf", ascending=False)
                          .join(self.corpus[COLUNAS_EVIDENCIA], on="review_id"))

        # Textos idênticos viram UMA evidência: fica a mais bem posicionada
        fundido = fundido.drop_duplicates("chave_texto", keep="first")

        # Reranker: lê pergunta + avaliação juntas nos 50 primeiros e reordena
        if reranquear and not fundido.empty:
            fundido = fundido.head(K_CANDIDATOS).copy()
            fundido["score_rerank"] = self.pontuar_reranker(pergunta, fundido)
            fundido = fundido.sort_values("score_rerank", ascending=False)

        # Quantas vezes cada texto aparece no universo filtrado ("aparece N vezes")
        universo = self.corpus[filtro_para_mascara(self.corpus, filtros)]
        contagem = universo["chave_texto"].value_counts()
        fundido["repeticoes"] = fundido["chave_texto"].map(contagem).fillna(1).astype(int)

        fundido = fundido.head(k_final).reset_index(drop=True)
        fundido.insert(0, "ref", [f"R{i + 1}" for i in range(len(fundido))])
        fundido.attrs["documentos_no_filtro"] = len(universo)  # volume de evidência disponível
        fundido.attrs["filtros"] = filtros or {}  # recorte usado, para a resposta declarar
        return fundido