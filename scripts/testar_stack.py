"""Etapa 0 — confirma que as bibliotecas pesadas instalam e funcionam no Python atual."""
import sys
from importlib.metadata import version

import chromadb

print("Python", sys.version.split()[0])
for pacote in ["torch", "sentence-transformers", "chromadb", "bm25s", "streamlit"]:
    print(f"{pacote:22s} {version(pacote)}")

# Mini teste do banco vetorial: guarda 1 "documento" com coordenadas inventadas e conta
cliente = chromadb.Client()  # banco temporário, só na memória (some ao fechar)
colecao = cliente.create_collection("teste")
colecao.add(ids=["1"], documents=["a entrega atrasou"], embeddings=[[0.1, 0.2, 0.3]])
print("Chroma OK:", colecao.count(), "documento salvo")