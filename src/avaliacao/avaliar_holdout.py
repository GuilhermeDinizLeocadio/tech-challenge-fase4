"""Etapa 7b — Teste de generalização: o sistema final, AO VIVO, em perguntas NOVAS.

As perguntas de data/avaliacao/perguntas_holdout.json foram escritas depois de todo o
desenvolvimento e nunca foram usadas para ajustar nada. Diferente das 31 perguntas de teste, aqui
tudo roda ao vivo, exatamente como na interface: roteador, reescrita, busca, reranker, resposta e
correção (ou o panorama de temas). Regra: nada é ajustado depois de ver estes resultados.

Saída: resultados/respostas/holdout.jsonl (rodar de novo pula o que já foi feito).

Uso (na raiz do projeto):
    python -m src.avaliacao.avaliar_holdout
"""
import json
import sys
import time
from pathlib import Path

from src.rag.pipeline import responder
from src.rag.recuperacao import Buscador
from src.temas.panorama import Panorama

RAIZ = Path(__file__).resolve().parents[2]
PERGUNTAS = RAIZ / "data" / "avaliacao" / "perguntas_holdout.json"
SAIDA = RAIZ / "resultados" / "respostas" / "holdout.jsonl"
INTERVALO = 45  # cada pergunta usa até 4 chamadas ao Qwen (roteador, reescrita, resposta, correção)
COLUNAS_EVIDENCIA = ["ref", "review_id", "chave_texto", "texto", "nota", "categoria_principal",
                     "situacao_entrega", "dias_atraso", "data_compra", "uf", "repeticoes",
                     "score_rerank", "tema_nome"]


def ler(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    perguntas = json.loads(PERGUNTAS.read_text(encoding="utf-8"))
    feitos = {r["id"] for r in ler(SAIDA) if "erro" not in r}
    pendentes = [p for p in perguntas if p["id"] not in feitos]
    print(f"{len(perguntas)} perguntas novas | {len(pendentes)} pendentes. Carregando...",
          flush=True)
    if pendentes:
        buscador, panorama = Buscador(), Panorama()
        ultima = 0.0
        with SAIDA.open("a", encoding="utf-8") as saida:
            for p in pendentes:
                espera = INTERVALO - (time.monotonic() - ultima)
                if espera > 0:
                    time.sleep(espera)
                ultima = time.monotonic()
                base = {"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                        "agregada": p.get("agregada", False), "pergunta": p["pergunta"],
                        "filtros": p.get("filtros"), "config": "holdout"}
                try:
                    s = responder(buscador, p["pergunta"], filtros=p.get("filtros"),
                                  panorama=panorama)
                    r, evidencias = s["resultado"], s["evidencias"]
                    if r is None:
                        raise RuntimeError("recorte sem avaliações")
                    colunas = [c for c in COLUNAS_EVIDENCIA if c in evidencias.columns]
                    registro = {**base, **r, "config": "holdout",
                                "rota": s["rota"]["tipo"], "grupo_rota": s["rota"]["grupo"],
                                "reescrita": s["reescrita"],
                                "comportamento_correto": (r["recusou"] if p["esperado"] == "recusar"
                                                          else not r["recusou"]),
                                "documentos_no_filtro": int(evidencias.attrs.get(
                                    "documentos_no_filtro", 0)),
                                "evidencias": json.loads(evidencias[colunas].to_json(
                                    orient="records", force_ascii=False))}
                    status = (f"rota {s['rota']['tipo']}"
                              f"{' (' + s['rota']['grupo'] + ')' if s['rota']['grupo'] else ''} | "
                              f"recusou={r['recusou']} | correção: {r.get('correcao', '-')}")
                except Exception as erro:  # registra e segue; a próxima execução refaz
                    registro = {**base, "erro": str(erro)}
                    status = f"ERRO: {erro}"
                saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                saida.flush()
                print(f"[{p['id']}] {status} | {p['pergunta'][:60]}", flush=True)

    rs = [r for r in ler(SAIDA) if "erro" not in r]
    certos = sum(r["comportamento_correto"] for r in rs)
    rotas = sum((r["rota"] == "agregada") == r["agregada"] for r in rs)
    print(f"\nComportamento correto: {certos} de {len(rs)} | rota como esperado: {rotas} de {len(rs)}")
    for r in rs:
        if not r["comportamento_correto"] or (r["rota"] == "agregada") != r["agregada"]:
            print(f"  [{r['id']}] esperado {r['esperado']}, agregada={r['agregada']} → "
                  f"recusou={r['recusou']}, rota {r['rota']} | {r['pergunta'][:60]}")


if __name__ == "__main__":
    main()