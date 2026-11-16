"""Etapa 6 — Consolida os 20 temas de cada grupo em macrotemas (dois níveis de taxonomia).

O K-means separa também pelo jeito de escrever ("não recebi" x "até hoje não chegou"), então o
mesmo assunto aparece em vários temas. Para contar "os principais problemas", os temas com o
mesmo assunto são reunidos em macrotemas pelo juiz, com regras definidas antes: 5 a 10
macrotemas por grupo, cada tema em exatamente um (conferido pelo código) e os temas mistos num
macrotema próprio, fora do ranking. K não foi alterado: escolher outro K depois de ver os
resultados seria ajustar o número ao que se quer ver.

Saídas: coluna "macrotema" em data/processed/temas.parquet e lista "macrotemas" em
resultados/temas.json.

Uso (na raiz do projeto):
    python -m src.temas.consolidar_temas
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.avaliacao.juiz import Juiz

RAIZ = Path(__file__).resolve().parents[2]
SAIDA_TEMAS = RAIZ / "data" / "processed" / "temas.parquet"
CATALOGO = RAIZ / "resultados" / "temas.json"
NOME_MISTOS = "Avaliações mistas ou genéricas"

PROMPT_CONSOLIDAR = f"""Você organiza temas de avaliações de clientes de um e-commerce brasileiro.
Recebe os temas de um grupo (gerados por agrupamento automático, por isso alguns repetem o mesmo
assunto com palavras diferentes) e deve reuni-los em MACROTEMAS que um gestor reconheceria como
o mesmo assunto.
Regras:
- entre 5 e 10 macrotemas;
- cada tema em exatamente UM macrotema; use todos os temas recebidos;
- junte apenas temas sobre o mesmo assunto (ex.: vários temas de "não entrega" num só);
- temas marcados como MISTO, ou que não descrevem um problema ou elogio claro, vão para um
  macrotema chamado exatamente "{NOME_MISTOS}";
- nomes com 2 a 5 palavras; descrição com uma frase.
Responda SOMENTE com JSON:
{{"macrotemas": [{{"nome": "...", "descricao": "...", "temas": ["P05", "P16"]}}]}}"""


def consolidar(juiz: Juiz, grupo: str, temas: list[dict]) -> list[dict]:
    linhas = [f'{t["id"]} | {t["nome"]} | {t["descricao"]} | {t["n_avaliacoes"]} avaliações | '
              f'nota média {t["nota_media"]} | entrega com problema '
              f'{t["pct_entrega_com_problema"]}% | {"coerente" if t["coerente"] else "MISTO"} | '
              f'exemplos: ' + " / ".join(t["exemplos"][:3]) for t in temas]
    ids = sorted(t["id"] for t in temas)
    for tentativa in range(1, 4):
        r = juiz.avaliar_json(PROMPT_CONSOLIDAR, f"Grupo: {grupo}\n\n" + "\n".join(linhas))
        macros = [m for m in r.get("macrotemas", []) if m.get("temas")]
        usados = sorted(str(i) for m in macros for i in m["temas"])
        if usados == ids and 5 <= len(macros) <= 10:
            return macros
        print(f"   consolidação inválida (tentativa {tentativa}): {len(macros)} macrotemas, "
              f"{len(usados)} temas usados de {len(ids)}", flush=True)
    raise RuntimeError(f"O juiz não consolidou o grupo {grupo} de forma válida.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    catalogo = json.loads(CATALOGO.read_text(encoding="utf-8"))
    atrib = pd.read_parquet(SAIDA_TEMAS)
    juiz = Juiz()

    todos_macros, mapa = [], {}
    for grupo in ("problema", "elogio"):
        temas = [t for t in catalogo["temas"] if t["grupo"] == grupo]
        macros = consolidar(juiz, grupo, temas)
        total_grupo = int((atrib["grupo"] == grupo).sum())
        linhas = []
        for m in macros:
            avaliacoes = atrib[atrib["tema"].isin(m["temas"])]
            linhas.append({
                "grupo": grupo, "nome": str(m["nome"]).strip(),
                "descricao": str(m.get("descricao", "")).strip(),
                "temas": sorted(str(i) for i in m["temas"]),
                "mistos": str(m["nome"]).strip() == NOME_MISTOS,
                "n_avaliacoes": int(len(avaliacoes)),
                "pct_do_grupo": round(100 * len(avaliacoes) / total_grupo, 1),
                "nota_media": round(float(avaliacoes["nota"].mean()), 2),
                "pct_entrega_com_problema": round(100 * avaliacoes["situacao_entrega"]
                                                  .isin(["atrasado", "não entregue"]).mean(), 1),
            })
        # Numerados do maior para o menor; o macrotema de mistos vai por último
        linhas.sort(key=lambda m: (m["mistos"], -m["n_avaliacoes"]))
        prefixo = "MP" if grupo == "problema" else "ME"
        print(f"\n{grupo.upper()} ({total_grupo:,} avaliações):")
        for i, m in enumerate(linhas, start=1):
            m["id"] = f"{prefixo}{i:02d}"
            mapa.update({t: m["id"] for t in m["temas"]})
            print(f"[{m['id']}] {m['nome'][:38]:<38} {m['n_avaliacoes']:>6,} "
                  f"({m['pct_do_grupo']:>4}%) | nota {m['nota_media']:.2f} | entrega c/ problema "
                  f"{m['pct_entrega_com_problema']:>5}% | {', '.join(m['temas'])}", flush=True)
        todos_macros += linhas

    atrib["macrotema"] = atrib["tema"].map(mapa)
    atrib.to_parquet(SAIDA_TEMAS, index=False)
    for t in catalogo["temas"]:
        t["macrotema"] = mapa[t["id"]]
    catalogo["macrotemas"] = todos_macros
    CATALOGO.write_text(json.dumps(catalogo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMacrotemas salvos em {CATALOGO.relative_to(RAIZ)} e {SAIDA_TEMAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()