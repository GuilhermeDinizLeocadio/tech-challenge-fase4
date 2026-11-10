"""Etapa 5 — Inspeção das auditorias de fidelidade, sem API: onde e por que o juiz não deu
"sustentada".

Mostra o % de afirmações sustentadas por pergunta em cada configuração (só o Qwen) e, para as
configurações pedidas em --detalhar, cada afirmação parcial ou não sustentada com a
justificativa do juiz.

Uso (na raiz do projeto):
    python -m src.avaliacao.inspecionar_fidelidade CONFIG1 CONFIG2 ... [--detalhar CONFIG ...] [--ids P19 P06]
"""
import argparse
import sys
import textwrap

import pandas as pd

from src.avaliacao.relatorio import PASTA_FIDELIDADE, ler_jsonl

MODELO = "qwen/qwen3.8-27b"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--detalhar", nargs="*", default=[])
    p.add_argument("--ids", nargs="*", default=None)
    args = p.parse_args()

    linhas, detalhes = [], []
    for config in args.configs:
        auditorias = {r["id"]: r for r in ler_jsonl(PASTA_FIDELIDADE / f"{config}.jsonl")
                      if r["modelo"] == MODELO}
        for id_, r in auditorias.items():
            vereditos = [af["veredito"] for af in r["afirmacoes"]]
            linhas.append({"id": id_, "config": config,
                           "sustentadas_%": 100 * vereditos.count("sustentada") / len(vereditos)})
            if config in args.detalhar and (args.ids is None or id_ in args.ids):
                detalhes += [(config, id_, af) for af in r["afirmacoes"]
                             if af["veredito"] != "sustentada"]

    pd.set_option("display.width", 200)
    tabela = pd.DataFrame(linhas).pivot_table(index="id", columns="config",
                                              values="sustentadas_%")[args.configs]
    print("% de afirmações sustentadas por pergunta (Qwen):")
    print(tabela.round(0).to_string())

    for config, id_, af in detalhes:
        print(f"\n[{id_}] {config} | {af['veredito']} | citações {af['citacoes']}")
        print(textwrap.indent(textwrap.fill("AFIRMAÇÃO: " + af["afirmacao"], 100), "  "))
        print(textwrap.indent(textwrap.fill("JUIZ: " + af["justificativa"], 100), "  "))


if __name__ == "__main__":
    main()