"""Etapas 4 e 5 — Relatório comparativo das configurações avaliadas.

Lê todas as configurações salvas em resultados/respostas/ (e a auditoria correspondente em
resultados/fidelidade/, quando existir) e monta uma tabela com uma linha por (configuração, modelo).

Os verificadores de trechos são RECALCULADOS a partir das respostas e evidências salvas, para que
todas as configurações sejam medidas com a mesma régua (sem chamar a API de novo):
- v1 (Etapa 3): trechos entre aspas que não estão nas avaliações;
- v2 (Etapa 5): separa trechos inexistentes, registro citado como relato e termos da pergunta.

Fidelidade ajustada: uma afirmação que o juiz considerou "sustentada", mas que contém um trecho
sinalizado pelo verificador v2 (inexistente ou registro citado como relato), é rebaixada para
"parcial". A validação manual (Etapa 4) mostrou que o juiz é leniente justamente nesses casos.

Uso (na raiz do projeto):
    python -m src.avaliacao.relatorio
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.ingestao.construir_corpus import gerar_chave_texto
from src.rag.geracao import classificar_trechos, verificar_trechos

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RESPOSTAS = RAIZ / "resultados" / "respostas"
PASTA_FIDELIDADE = RAIZ / "resultados" / "fidelidade"
SAIDA = RAIZ / "resultados" / "tabela_comparativa.csv"


def ler_jsonl(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    registros = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in registros if "erro" not in r]


def ultima_por_par(registros: list[dict]) -> dict:
    return {(r["id"], r["modelo"]): r for r in registros}  # fica a última de cada par


def contem_trecho_sinalizado(afirmacao: str, trechos: list[str]) -> bool:
    texto = gerar_chave_texto(afirmacao)
    return any(len(chave) >= 3 and chave in texto
               for chave in (gerar_chave_texto(t) for t in trechos))


def verificar(r: dict) -> dict:
    """Recalcula os verificadores v1 e v2 a partir da resposta e das evidências salvas."""
    evidencias = pd.DataFrame(r["evidencias"])
    v2 = classificar_trechos(r["resposta"], evidencias, r["pergunta"])
    return {"v1": verificar_trechos(r["resposta"], evidencias), **v2,
            "sinalizados": v2["trechos_nao_encontrados"] + v2["registro_como_relato"]}


def metricas_configuracao(config: str) -> list[dict]:
    respostas = ultima_por_par(ler_jsonl(PASTA_RESPOSTAS / f"{config}.jsonl"))
    auditorias = ultima_por_par(ler_jsonl(PASTA_FIDELIDADE / f"{config}.jsonl"))
    verificacoes = {chave: verificar(r) for chave, r in respostas.items()}
    linhas = []
    for modelo in sorted({m for _, m in respostas}):
        chaves = [c for c in respostas if c[1] == modelo]
        rs = [respostas[c] for c in chaves]
        vs = [verificacoes[c] for c in chaves]
        sem_evid = [r for r in rs if r["tipo"] == "sem_evidencia"]
        linha = {
            "config": config, "modelo": modelo.split("/")[-1], "respostas": len(rs),
            "comportamento_correto_%": 100 * sum(r["comportamento_correto"] for r in rs) / len(rs),
            "sem_evidencia_recusadas_%": (100 * sum(r["recusou"] for r in sem_evid) / len(sem_evid)
                                          if sem_evid else float("nan")),
            "citacoes_invalidas": sum(len(r["citacoes_invalidas"]) for r in rs),
            "trechos_v1": sum(len(v["v1"]) for v in vs),
            "trechos_nao_encontrados_v2": sum(len(v["trechos_nao_encontrados"]) for v in vs),
            "registro_como_relato_v2": sum(len(v["registro_como_relato"]) for v in vs),
            "trechos_da_pergunta_v2": sum(len(v["trechos_da_pergunta"]) for v in vs),
            "respostas_com_termos_freq": sum(bool(r["termos_frequencia"]) for r in rs),
            "tokens_saida_medio": sum(r["tokens_saida"] or 0 for r in rs) / len(rs),
            "segundos_mediana": pd.Series([r["segundos"] for r in rs]).median(),
        }

        auds = [(a, verificacoes[(a["id"], a["modelo"])]) for (_, m), a in auditorias.items()
                if m == modelo and (a["id"], a["modelo"]) in respostas]
        if auds:
            total = sust = sust_ajust = parc = nao = 0
            for aud, v in auds:
                for af in aud["afirmacoes"]:
                    total += 1
                    if af["veredito"] == "sustentada":
                        sust += 1
                        if not contem_trecho_sinalizado(af["afirmacao"], v["sinalizados"]):
                            sust_ajust += 1
                    elif af["veredito"] == "parcial":
                        parc += 1
                    else:
                        nao += 1
            linha.update({
                "afirmacoes_auditadas": total,
                "fidelidade_juiz_%": 100 * sust / total,
                "fidelidade_ajustada_%": 100 * sust_ajust / total,
                "nao_sustentadas_%": 100 * nao / total,
                "interpretacao_fatos_novos": sum(a["fatos_novos_na_interpretacao"] for a, _ in auds),
                "responde_a_pergunta_media": sum(a["responde_a_pergunta"] for a, _ in auds) / len(auds),
            })
        linhas.append(linha)
    return linhas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    configs = sorted(p.stem for p in PASTA_RESPOSTAS.glob("*.jsonl"))
    tabela = pd.DataFrame([linha for c in configs for linha in metricas_configuracao(c)])
    SAIDA.parent.mkdir(exist_ok=True)
    tabela.to_csv(SAIDA, index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(tabela.set_index(["config", "modelo"]).round(1).T.to_string())
    print(f"\nTabela salva em {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()