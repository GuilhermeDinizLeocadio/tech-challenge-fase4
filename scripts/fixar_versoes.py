"""Gera requirements.txt com as versões INSTALADAS dos pacotes listados em requirements.in.

Fixa apenas as dependências diretas: um `pip freeze` completo no Windows inclui
pacotes específicos da plataforma (ex.: pywin32) que quebram a instalação em Linux/Mac.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]  # pasta do projeto (um nível acima de scripts/)

# Lê requirements.in ignorando comentários (#) e linhas vazias
pacotes = [
    linha.split("#")[0].strip()
    for linha in (RAIZ / "requirements.in").read_text(encoding="utf-8").splitlines()
]
fixados = []
for pacote in filter(None, pacotes):
    try:
        fixados.append(f"{pacote}=={version(pacote)}")  # ex.: pandas==2.x.y
    except PackageNotFoundError:
        print(f"AVISO: {pacote} não está instalado")

(RAIZ / "requirements.txt").write_text("\n".join(fixados) + "\n", encoding="utf-8")
print("\n".join(fixados))