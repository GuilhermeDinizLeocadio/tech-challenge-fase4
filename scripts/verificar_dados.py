"""Etapa 0 — confere se os CSVs da Olist necessários estão em data/raw e se carregam corretamente."""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# arquivo -> coluna-chave usada para juntar as tabelas
ARQUIVOS = {
    "olist_order_reviews_dataset.csv": "review_id",
    "olist_orders_dataset.csv": "order_id",
    "olist_order_items_dataset.csv": "order_id",
    "olist_products_dataset.csv": "product_id",
    "product_category_name_translation.csv": "product_category_name",
    "olist_customers_dataset.csv": "customer_id",
}

for nome, chave in ARQUIVOS.items():
    caminho = RAW / nome
    if not caminho.exists():
        print(f"FALTANDO: {nome}")
        continue
    # utf-8-sig remove um caractere invisível (BOM) que alguns CSVs da Olist trazem no cabeçalho
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    print(
        f"{nome:42s} {len(df):>7,} linhas x {df.shape[1]:>2} col | "
        f"{chave}: {df[chave].nunique():>7,} únicos | {caminho.stat().st_size / 1e6:5.1f} MB"
    )