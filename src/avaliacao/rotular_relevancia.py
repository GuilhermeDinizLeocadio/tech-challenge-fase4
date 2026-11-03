"""Etapa 4 — Rotula a relevância das avaliações recuperadas para cada pergunta de teste.

Método de "pooling": para cada pergunta, junta os candidatos das buscas que serão comparadas
(densa e híbrida, até 50 de cada) e pede ao juiz uma nota para cada um:
  2 = ajuda a responder diretamente à pergunta
  1 = mesmo assunto geral, mas não responde
  0 = sem relação
Textos idênticos (mesma chave_texto) são rotulados uma vez só. Os rótulos são salvos em
resultados/relevancia.jsonl assim que chegam; rodar de novo só rotula os pares ainda não vistos.

Uso (na raiz do projeto):
    python -m src.avaliacao.rotular_relevancia
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.avaliacao.juiz import Juiz
from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.geracao import descrever_filtros
from src.rag.recuperacao import Buscador

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_ROTULOS = RAIZ / "resultados" / "relevancia.jsonl"
K_POOL = 50         # candidatos de cada busca (densa e híbrida) por pergunta
TAMANHO_LOTE = 25   # avaliações por chamada ao juiz

PROMPT_RELEVANCIA = """Você avalia a relevância de avaliações de clientes de um e-commerce
brasileiro para uma pergunta de um gestor. Para cada avaliação, dê uma nota:
2 = a avaliação traz informação que ajuda a responder DIRETAMENTE à pergunta (mesmo tema e,
    quando a pergunta pede, o mesmo tipo de experiência: problema, elogio etc.);
1 = a avaliação é do mesmo assunto geral, mas NÃO ajuda a responder (ex.: fala de entrega em
    geral, mas a pergunta é sobre entrega feita por drones);
0 = a avaliação não tem relação com a pergunta.
Julgue apenas o texto da avaliação. Seja exigente: na dúvida entre 2 e 1, escolha 1.
Os nomes Stark, Lannister, Targaryen e Baratheon são nomes de empresas anonimizados.
Responda SOMENTE com JSON no formato {"notas": [{"id": "A1", "nota": 2}, ...]},
com exatamente uma entrada para cada avaliação recebida."""


def carregar_rotulos() -> dict[tuple[str, str], int]:
    """Rótulos já salvos: {(id_pergunta, chave_texto): nota}."""
    rotulos = {}
    if ARQUIVO_ROTULOS.exists():
        for linha in ARQUIVO_ROTULOS.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                rotulos[(r["id_pergunta"], r["chave_texto"])] = r["nota"]
    return rotulos


def montar_pool(buscador: Buscador, pergunta: dict) -> pd.DataFrame:
    """Candidatos distintos das buscas densa e híbrida para uma pergunta."""
    filtros = pergunta.get("filtros")
    densa = buscador.buscar(pergunta["pergunta"], filtros=filtros, k_final=K_POOL, modo="densa")
    hibrida = buscador.buscar(pergunta["pergunta"], filtros=filtros, k_final=K_POOL, modo="hibrida")
    return (pd.concat([densa, hibrida])[["chave_texto", "texto"]]
              .drop_duplicates("chave_texto").reset_index(drop=True))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = carregar_perguntas()
    rotulos = carregar_rotulos()
    juiz = Juiz()
    print(f"Juiz: {juiz.modelo} | rótulos já salvos: {len(rotulos)}. Carregando o buscador...",
          flush=True)
    buscador = Buscador()

    ARQUIVO_ROTULOS.parent.mkdir(exist_ok=True)
    with ARQUIVO_ROTULOS.open("a", encoding="utf-8") as saida:
        for p in perguntas:
            pool = montar_pool(buscador, p)
            pendentes = pool[[(p["id"], c) not in rotulos for c in pool["chave_texto"]]]
            print(f"[{p['id']}] candidatos {len(pool)} | já rotulados {len(pool) - len(pendentes)}"
                  f" | a rotular {len(pendentes)}", flush=True)

            for inicio in range(0, len(pendentes), TAMANHO_LOTE):
                lote = pendentes.iloc[inicio:inicio + TAMANHO_LOTE]
                ids = [f"A{i + 1}" for i in range(len(lote))]
                corpo = "\n".join(f'[{i}] "{t}"' for i, t in zip(ids, lote["texto"]))
                usuario = (f"Pergunta do gestor: {p['pergunta']}\n"
                           f"Recorte consultado: {descrever_filtros(p.get('filtros'))}\n\n"
                           f"Avaliações:\n{corpo}")
                resposta = juiz.avaliar_json(PROMPT_RELEVANCIA, usuario)

                notas = {}
                for item in resposta.get("notas", []):
                    try:
                        notas[str(item["id"])] = int(item["nota"])
                    except (KeyError, TypeError, ValueError):
                        continue
                faltando = [i for i in ids if notas.get(i) not in (0, 1, 2)]
                if faltando:
                    print(f"   aviso: {len(faltando)} avaliações sem nota válida; "
                          "ficam para a próxima execução", flush=True)

                for i, chave in zip(ids, lote["chave_texto"]):
                    if notas.get(i) in (0, 1, 2):
                        registro = {"id_pergunta": p["id"], "chave_texto": chave,
                                    "nota": notas[i], "juiz": juiz.modelo}
                        saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                        saida.flush()  # grava na hora: se parar, nada se perde
                        rotulos[(p["id"], chave)] = notas[i]

    distribuicao = Counter(rotulos.values())
    print(f"\nTotal de rótulos: {len(rotulos):,} | chamadas ao juiz nesta execução: {juiz.chamadas}")
    print(f"Distribuição das notas: 2 = {distribuicao[2]:,} | 1 = {distribuicao[1]:,} | "
          f"0 = {distribuicao[0]:,}")


if __name__ == "__main__":
    main()