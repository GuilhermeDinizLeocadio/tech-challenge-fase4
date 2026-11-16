"""Etapa 6 — Avalia o roteador nas 31 perguntas de teste: a rota escolhida bate com a marcação
"agregada" do conjunto de teste (feita na Etapa 4, antes de o roteador existir)?

Salva cada decisão em resultados/roteador.jsonl; rodar de novo pula o que já foi avaliado.

Uso (na raiz do projeto):
    python -m src.avaliacao.avaliar_roteador
"""
import json
import sys
import time
from pathlib import Path

from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.geracao import criar_cliente
from src.rag.roteador import classificar_pergunta

RAIZ = Path(__file__).resolve().parents[2]
SAIDA = RAIZ / "resultados" / "roteador.jsonl"
INTERVALO = 4  # segundos entre chamadas (são curtas)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    feitos = {}
    if SAIDA.exists():
        for linha in SAIDA.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                feitos[r["id"]] = r
    cliente = criar_cliente("groq")
    with SAIDA.open("a", encoding="utf-8") as saida:
        for p in carregar_perguntas():
            if p["id"] in feitos:
                continue
            rota = classificar_pergunta(p["pergunta"], cliente)
            r = {"id": p["id"], "pergunta": p["pergunta"], "tipo_pergunta": p["tipo"],
                 "agregada_esperada": bool(p.get("agregada", False)),
                 "rota": rota["tipo"], "grupo": rota["grupo"]}
            saida.write(json.dumps(r, ensure_ascii=False) + "\n")
            saida.flush()
            feitos[p["id"]] = r
            print(f"[{p['id']}] {rota['tipo']:<11} {rota['grupo'] or '':<9} | {p['pergunta'][:70]}",
                  flush=True)
            time.sleep(INTERVALO)

    rs = list(feitos.values())
    acertos = [r for r in rs if (r["rota"] == "agregada") == r["agregada_esperada"]]
    vp = sum(r["rota"] == "agregada" and r["agregada_esperada"] for r in rs)
    fp = sum(r["rota"] == "agregada" and not r["agregada_esperada"] for r in rs)
    fn = sum(r["rota"] != "agregada" and r["agregada_esperada"] for r in rs)
    print(f"\nAcurácia do roteador: {len(acertos)} de {len(rs)} ({100 * len(acertos) / len(rs):.0f}%)")
    print(f"Agregadas reconhecidas: {vp} de {vp + fn} | específicas mandadas ao panorama: {fp}")
    print("\nDivergências:")
    for r in rs:
        if (r["rota"] == "agregada") != r["agregada_esperada"]:
            esperado = "agregada" if r["agregada_esperada"] else "especifica"
            print(f"  [{r['id']}] esperado {esperado} → rota {r['rota']} | {r['pergunta'][:70]}")


if __name__ == "__main__":
    main()