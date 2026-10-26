"""Etapa 4 — O juiz: um LLM avaliador, com resposta em JSON, ritmo controlado e conferência
do nome do modelo antes de começar. Usado na rotulagem de relevância e na avaliação de fidelidade.
"""
import json
import os
import time

from dotenv import load_dotenv
from openai import RateLimitError

from src.rag.geracao import criar_cliente

load_dotenv()

PROVEDOR_JUIZ = "gemini"
INTERVALO_MINIMO = 4.5  # segundos entre chamadas (~13 por minuto, abaixo do limite de 15)


def extrair_json(texto: str) -> dict:
    """Extrai o objeto JSON da resposta (o modelo às vezes o envolve em ```json ... ```)."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1:
        raise ValueError("nenhum JSON encontrado na resposta")
    return json.loads(texto[inicio:fim + 1])


class Juiz:
    def __init__(self, modelo: str | None = None) -> None:
        self.modelo = modelo or os.getenv("JUIZ_MODELO")
        if not self.modelo:
            raise RuntimeError("JUIZ_MODELO não está definido no .env")
        self.cliente = criar_cliente(PROVEDOR_JUIZ)
        self._conferir_modelo()
        self._ultima_chamada = 0.0
        self.chamadas = 0

    def _conferir_modelo(self) -> None:
        """Erra cedo e com clareza se o nome do modelo não existir no provedor."""
        disponiveis = {m.id.removeprefix("models/") for m in self.cliente.models.list()}
        if self.modelo not in disponiveis:
            raise RuntimeError(f"O modelo do juiz '{self.modelo}' não existe no provedor. "
                               "Confira a linha JUIZ_MODELO do .env.")

    def _respeitar_ritmo(self) -> None:
        espera = INTERVALO_MINIMO - (time.monotonic() - self._ultima_chamada)
        if espera > 0:
            time.sleep(espera)
        self._ultima_chamada = time.monotonic()

    def avaliar_json(self, sistema: str, usuario: str, tentativas: int = 4) -> dict:
        """Chama o juiz e devolve a resposta já convertida de JSON para dicionário."""
        for tentativa in range(1, tentativas + 1):
            self._respeitar_ritmo()
            try:
                resposta = self.cliente.chat.completions.create(
                    model=self.modelo, temperature=0,
                    messages=[{"role": "system", "content": sistema},
                              {"role": "user", "content": usuario}],
                )
                self.chamadas += 1
                return extrair_json(resposta.choices[0].message.content or "")
            except RateLimitError:
                espera = 30 * tentativa
                print(f"   limite do juiz atingido; aguardando {espera}s "
                      f"(tentativa {tentativa}/{tentativas})", flush=True)
                time.sleep(espera)
            except ValueError as erro:  # inclui JSON inválido (json.JSONDecodeError)
                print(f"   resposta do juiz fora do formato (tentativa {tentativa}): {erro}",
                      flush=True)
        raise RuntimeError("O juiz não devolveu uma resposta válida após várias tentativas.")