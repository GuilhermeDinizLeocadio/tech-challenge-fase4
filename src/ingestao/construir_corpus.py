"""Etapa 1 — Constrói a base de conhecimento a partir dos CSVs da Olist.

Gera dois arquivos em data/processed/:
- avaliacoes.parquet: TODAS as avaliações únicas (com e sem texto) + metadados -> estatísticas
- corpus.parquet:     só avaliações com texto válido -> documentos da busca (RAG)

Uso (na raiz do projeto):
    python -m src.ingestao.construir_corpus
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]  # src/ingestao/arquivo.py -> raiz do projeto
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"

# Nomes de empresas anonimizados pela Olist (casas de Game of Thrones)
CASAS = ["stark", "lannister", "targaryen", "baratheon", "greyjoy", "tyrell",
         "martell", "arryn", "tully", "bolton", "frey", "mormont"]
PADRAO_CASAS = r"\b(" + "|".join(CASAS) + r")\b"

LIMITE_TEXTO_CURTO = 3  # até 3 palavras = texto curto

REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_PARA_REGIAO = {uf: regiao for regiao, ufs in REGIOES.items() for uf in ufs}

# Quando uma avaliação liga vários pedidos, vale o PIOR caso de entrega
GRAVIDADE = {"no prazo": 0, "atrasado": 1, "não entregue": 2}

COLUNAS_FINAIS = [
    "review_id", "texto", "nota", "situacao_entrega", "dias_atraso", "status_pedido",
    "categoria_principal", "categorias", "data_compra", "data_avaliacao", "uf", "regiao",
    "empresas_citadas", "n_palavras", "texto_curto", "chave_texto", "pedidos", "tem_texto",
]


def ler_csv(nome: str, **kwargs) -> pd.DataFrame:
    # utf-8-sig remove o BOM (caractere invisível) do cabeçalho de alguns CSVs
    return pd.read_csv(RAW / nome, encoding="utf-8-sig", **kwargs)


def limpar_texto(serie: pd.Series) -> pd.Series:
    """Limpeza LEVE, que preserva o sentido: sem remover stopwords, sem stemming, sem corretor.

    Modelos de embedding usam a frase inteira; tirar palavras como "não" mudaria o sentido.
    """
    return (serie.fillna("")
                 .str.replace(r"[\r\n\t]+", " ", regex=True)  # quebras de linha -> espaço
                 .str.replace(r"\s{2,}", " ", regex=True)       # espaços repetidos -> um só
                 .str.strip())


def gerar_chave_texto(texto: str) -> str:
    """Versão padronizada para achar textos idênticos: minúsculas, sem acento, sem pontuação."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    so_letras = re.sub(r"[^a-z0-9 ]", " ", sem_acento.lower())
    return re.sub(r"\s+", " ", so_letras).strip()


def montar_textos(reviews: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por avaliação, com o texto final e os campos derivados dele."""
    # As linhas repetidas de um review_id são idênticas, exceto pelo order_id (EDA, célula 3)
    av = reviews.drop_duplicates("review_id").set_index("review_id")

    titulo = limpar_texto(av["review_comment_title"])
    mensagem = limpar_texto(av["review_comment_message"])

    # Junta título e mensagem; se o título já termina em pontuação, separa só com espaço
    separador = pd.Series(np.where(titulo.str.contains(r"[.!?…]$", regex=True), " ", ". "),
                          index=av.index)
    ambos = (titulo != "") & (mensagem != "")
    av["texto"] = pd.Series(np.where(ambos, titulo + separador + mensagem, titulo + mensagem),
                            index=av.index, dtype="str")

    # Texto válido = pelo menos uma letra ou número (descarta "...", "!!!" e vazios)
    av["tem_texto"] = av["texto"].str.contains(r"[^\W_]", regex=True)
    av["n_palavras"] = av["texto"].str.split().str.len().fillna(0).astype(int)
    av["texto_curto"] = av["n_palavras"] <= LIMITE_TEXTO_CURTO
    av["chave_texto"] = av["texto"].map(gerar_chave_texto)
    av["empresas_citadas"] = (av["texto"].str.lower()
                              .str.findall(PADRAO_CASAS)
                              .map(lambda achadas: " | ".join(sorted(set(achadas)))))

    av["nota"] = av["review_score"].astype(int)
    av["data_avaliacao"] = (pd.to_datetime(av["review_creation_date"])
                            .dt.strftime("%Y%m%d").astype(int))  # 2018-03-07 -> 20180307
    return av


def montar_metadados_pedido(rev_ped, orders, customers) -> pd.DataFrame:
    """Entrega, status, data da compra, UF e pedidos ligados — reunidos por avaliação."""
    orders = orders.copy()
    orders["dias_atraso"] = (orders["order_delivered_customer_date"].dt.normalize()
                             - orders["order_estimated_delivery_date"].dt.normalize()).dt.days
    orders["situacao_entrega"] = np.select(
        [orders["order_delivered_customer_date"].isna(), orders["dias_atraso"] > 0],
        ["não entregue", "atrasado"],
        default="no prazo",
    )

    ped = (rev_ped
           .merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                          "situacao_entrega", "dias_atraso"]], on="order_id", how="left")
           .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left"))
    ped["gravidade"] = ped["situacao_entrega"].map(GRAVIDADE)
    ped = ped.sort_values(["review_id", "gravidade"])  # pior caso fica por último

    meta = ped.groupby("review_id").agg(
        situacao_entrega=("situacao_entrega", "last"),
        status_pedido=("order_status", "last"),
        dias_atraso=("dias_atraso", "max"),
        data_compra=("order_purchase_timestamp", "min"),
        uf=("customer_state", "first"),
        pedidos=("order_id", lambda s: " | ".join(s)),
    )
    meta["dias_atraso"] = meta["dias_atraso"].astype("Int64")  # inteiro que aceita vazio
    meta["data_compra"] = meta["data_compra"].dt.strftime("%Y%m%d").astype(int)
    return meta


def montar_categorias(rev_ped, items, products) -> pd.DataFrame:
    """Categoria principal (item mais caro) e lista de todas as categorias por avaliação."""
    itens = items[["order_id", "product_id", "price"]].merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left")
    itens["categoria"] = (itens["product_category_name"].fillna("sem categoria")
                          .str.replace("_", " "))
    itens["tem_categoria"] = itens["categoria"] != "sem categoria"

    cat = rev_ped.merge(itens[["order_id", "categoria", "price", "tem_categoria"]],
                        on="order_id", how="inner")
    principal = (cat.sort_values(["tem_categoria", "price"], ascending=[False, False])
                    .drop_duplicates("review_id")
                    .set_index("review_id")["categoria"])
    todas = cat.groupby("review_id")["categoria"].agg(lambda s: " | ".join(sorted(s.unique())))
    return pd.DataFrame({"categoria_principal": principal, "categorias": todas})


def main() -> None:
    reviews = ler_csv("olist_order_reviews_dataset.csv")
    orders = ler_csv("olist_orders_dataset.csv",
                     parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                  "order_estimated_delivery_date"])
    items = ler_csv("olist_order_items_dataset.csv")
    products = ler_csv("olist_products_dataset.csv")
    customers = ler_csv("olist_customers_dataset.csv")

    # Ponte avaliação -> pedidos (uma avaliação pode estar ligada a até 3 pedidos)
    rev_ped = reviews[["review_id", "order_id"]].drop_duplicates()

    av = (montar_textos(reviews)
          .join(montar_metadados_pedido(rev_ped, orders, customers))
          .join(montar_categorias(rev_ped, items, products)))

    # Tratamento explícito de ausentes
    av[["categoria_principal", "categorias"]] = av[["categoria_principal", "categorias"]].fillna(
        "sem categoria")
    av["uf"] = av["uf"].fillna("desconhecida")
    av["regiao"] = av["uf"].map(UF_PARA_REGIAO).fillna("desconhecida")

    av = av.reset_index()[COLUNAS_FINAIS]
    corpus = av[av["tem_texto"]].drop(columns="tem_texto").reset_index(drop=True)

    # Verificações de sanidade: se alguma falhar, o script para com erro
    assert av["review_id"].is_unique, "review_id repetido em avaliacoes"
    assert corpus["texto"].str.len().gt(0).all(), "documento vazio no corpus"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    av.to_parquet(PROCESSED / "avaliacoes.parquet", index=False)
    corpus.to_parquet(PROCESSED / "corpus.parquet", index=False)

    # Resumo
    print(f"Avaliações únicas (avaliacoes.parquet): {len(av):,}")
    print(f"Documentos com texto (corpus.parquet):  {len(corpus):,}")
    print(f"Textos distintos (chave_texto):          {corpus['chave_texto'].nunique():,}")
    print(f"Textos curtos (<= {LIMITE_TEXTO_CURTO} palavras):           {corpus['texto_curto'].sum():,}")
    print(f"Citam empresa anonimizada:               {(corpus['empresas_citadas'] != '').sum():,}")
    print("\nSituação da entrega no corpus (%):")
    print(corpus["situacao_entrega"].value_counts(normalize=True).mul(100).round(1).to_string())
    print(f"\nSem categoria: {(corpus['categoria_principal'] == 'sem categoria').sum():,}")
    print(f"UF desconhecida: {(corpus['uf'] == 'desconhecida').sum():,}")


if __name__ == "__main__":
    main()