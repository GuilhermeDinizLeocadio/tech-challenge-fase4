"""Etapa 7 — Monta as respostas do SISTEMA FINAL para as 31 perguntas de teste.

Sistema final = roteador + rota agregada (panorama de temas) ou rota específica (busca densa +
reescrita + reranker + prompt v4 + correção v2).
- Rota: a decisão do roteador salva em resultados/roteador.jsonl (avaliada na Etapa 6).
- Específicas: reaproveita as respostas e auditorias da configuração
  reescrita_rerank_k20_prompt_v4_corrigida, que é exatamente o fluxo específico final (mesmas
  perguntas, evidências e respostas). Gerar de novo só gastaria cota e traria variação aleatória.
- Agregadas: gera agora a resposta com o panorama de temas.

Saídas: resultados/respostas/final.jsonl e resultados/fidelidade/final.jsonl (com as auditorias
copiadas das específicas; as agregadas são auditadas depois por avaliar_fidelidade).

Uso (na raiz do projeto):
    python -m src.avaliacao.gerar_final
"""
import json
import sys
import time
from pathlib import Path

from src.avaliacao.validar_perguntas import carregar_perguntas
from src.temas.panorama import Panorama
from src.temas.resposta_agregada import gerar_resposta_agregada

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RESPOSTAS = RAIZ / "resultados" / "respostas"
PASTA_FIDELIDADE = RAIZ / "resultados" / "fidelidade"
ROTAS = RAIZ / "resultados" / "roteador.jsonl"
ORIGEM = "reescrita_rerank_k20_prompt_v4_corrigida"
CONFIG = "final"
MODELO = "qwen/qwen3.8-27b"
INTERVALO = 35  # segundos entre respostas agregadas (limite de tokens por minuto do Groq)
COLUNAS_EVIDENCIA = ["ref", "review_id", "chave_texto", "texto", "nota", "categoria_principal",
                     "situacao_entrega", "dias_atraso", "data_compra", "uf", "repeticoes",
                     "tema_nome"]


def ler(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rotas = {r["id"]: r for r in ler(ROTAS)}
    origem = {r["id"]: r for r in ler(PASTA_RESPOSTAS / f"{ORIGEM}.jsonl")
              if "erro" not in r and r["modelo"] == MODELO}
    auditorias = {r["id"]: r for r in ler(PASTA_FIDELIDADE / f"{ORIGEM}.jsonl")
                  if "erro" not in r and r["modelo"] == MODELO}
    feitos = {r["id"] for r in ler(PASTA_RESPOSTAS / f"{CONFIG}.jsonl") if "erro" not in r}
    perguntas = carregar_perguntas()
    faltando = [p["id"] for p in perguntas if p["id"] not in rotas]
    if faltando:
        raise RuntimeError(f"Sem decisão do roteador para {faltando}: "
                           "rode python -m src.avaliacao.avaliar_roteador")

    panorama, ultima = None, 0.0
    with (PASTA_RESPOSTAS / f"{CONFIG}.jsonl").open("a", encoding="utf-8") as saida, \
         (PASTA_FIDELIDADE / f"{CONFIG}.jsonl").open("a", encoding="utf-8") as saida_fid:
        for p in perguntas:
            if p["id"] in feitos:
                continue
            rota = rotas[p["id"]]
            base = {"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                    "agregada": p.get("agregada", False), "pergunta": p["pergunta"],
                    "filtros": p.get("filtros"), "config": CONFIG, "modelo": MODELO,
                    "rota": rota["rota"], "grupo_rota": rota["grupo"]}

            if rota["rota"] == "especifica":  # reaproveita resposta e auditoria
                registro = {**origem[p["id"]], **base, "origem": ORIGEM}
                saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                if p["id"] in auditorias:
                    saida_fid.write(json.dumps({**auditorias[p["id"]], "config": CONFIG},
                                               ensure_ascii=False) + "\n")
                print(f"[{p['id']}] específica (reaproveitada)", flush=True)
                continue

            panorama = panorama or Panorama()
            espera = INTERVALO - (time.monotonic() - ultima)
            if espera > 0:
                time.sleep(espera)
            ultima = time.monotonic()
            try:
                resultado, exemplos = gerar_resposta_agregada(
                    p["pergunta"], p.get("filtros"), rota["grupo"], panorama, modelo=MODELO)
                if resultado is None:
                    raise RuntimeError("recorte sem avaliações")
                colunas = [c for c in COLUNAS_EVIDENCIA if c in exemplos.columns]
                registro = {**base, **resultado,
                            "comportamento_correto": (resultado["recusou"]
                                                      if p["esperado"] == "recusar"
                                                      else not resultado["recusou"]),
                            "documentos_no_filtro": exemplos.attrs["documentos_no_filtro"],
                            "evidencias": json.loads(exemplos[colunas].to_json(
                                orient="records", force_ascii=False))}
                status = (f"agregada ({rota['grupo']}) | {resultado['segundos']} s | "
                          f"números fora do panorama: "
                          f"{', '.join(resultado['numeros_fora_do_panorama']) or 'nenhum'}")
            except Exception as erro:  # registra e segue; a próxima execução refaz
                registro = {**base, "erro": str(erro)}
                status = f"ERRO: {erro}"
            saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            saida.flush()
            print(f"[{p['id']}] {status}", flush=True)


if __name__ == "__main__":
    main()