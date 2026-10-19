"""Regras de embedding compartilhadas pela indexação (Etapa 2) e pela busca (Etapa 3).

Ficam num só lugar para garantir que perguntas e documentos usem exatamente
o mesmo modelo, a mesma precisão e a mesma convenção de instrução.
"""
from sentence_transformers import SentenceTransformer

MODELO = "intfloat/multilingual-e5-large-instruct"

# Instrução que acompanha as PERGUNTAS (documentos entram puros).
# Em inglês, como no treinamento do modelo; a pergunta em si segue em português.
INSTRUCAO = ("Given a question about an online shopping experience, "
             "retrieve customer reviews that answer it")


def carregar_modelo() -> SentenceTransformer:
    """Carrega o modelo na CPU em float32.

    Os pesos vêm do Hugging Face em float16; na CPU, float32 é o formato nativo
    e garante precisão total para comparar similaridades muito próximas.
    """
    modelo = SentenceTransformer(MODELO, device="cpu")
    return modelo.float()


def preparar_pergunta(pergunta: str) -> str:
    """Formato de pergunta do e5-instruct: instrução + pergunta."""
    return f"Instruct: {INSTRUCAO}\nQuery: {pergunta}"