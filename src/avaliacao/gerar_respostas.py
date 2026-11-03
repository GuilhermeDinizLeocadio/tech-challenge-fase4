"""Etapa 4 — Gera as respostas dos redatores para todas as perguntas de teste e salva em arquivo.

Para cada pergunta, a busca é feita UMA vez e as mesmas evidências vão para todos os modelos
(comparação justa). As chamadas alternam entre os modelos e respeitam um intervalo mínimo por
modelo, para não estourar o limite de tokens por minuto da camada gratuita do Groq.
Cada resposta é salva assim que chega, junto com as evidências usadas; rodar de novo pula o que
já foi gerado e refaz o que terminou em erro.

Uso (na raiz do projeto):
    python -m src.avaliacao.gerar_respostas
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.geracao import criar_cliente, gerar_resposta
from src.rag.recuperacao import Buscador

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "resultados" / "respostas"
CONFIG = "densa_k20_prompt_v2"  # identifica esta configuração nas comparações futuras
MODELOS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
K = 20
INTERVALO_POR_MODELO = 30  # segundos entre duas chamadas ao mesmo modelo
COLUNAS_EVIDENCIA = ["ref", "review_id", "chave_texto", "texto", "nota", "categoria_principal",
                     "situacao_entrega", "dias_atraso", "data_compra", "uf", "repeticoes"]
CAMPOS_RESULTADO = ["resposta", "recusou", "citacoes", "citacoes_invalidas",
                    "trechos_nao_encontrados", "termos_frequencia", "tokens_entrada",
                    "tokens_saida", "segundos"]


def conferir_modelos() -> None:
    """Erra cedo e com clareza se algum modelo não existir no Groq."""
    disponiveis = {m.id for m in criar_cliente("groq").models.list()}
    faltando = [m for m in MODELOS if m not in disponiveis]
    if faltando:
        raise RuntimeError(f"Modelos inexistentes no Groq: {faltando}")


def ler_registros(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    return [json.loads(linha) for linha in arquivo.read_text(encoding="utf-8").splitlines()
            if linha.strip()]


def comportamento_correto(esperado: str, recusou: bool) -> bool:
    """Recusar quando esperado; não recusar quando é para responder ou ressalvar."""
    return recusou if esperado == "recusar" else not recusou


def imprimir_resumo(arquivo: Path) -> None:
    registros = ler_registros(arquivo)
    erros = sum("erro" in r for r in registros)
    t = (pd.DataFrame([r for r in registros if "erro" not in r])
           .drop_duplicates(["id", "modelo"], keep="last"))
    t["n_invalidas"] = t["citacoes_invalidas"].apply(len)
    t["n_trechos"] = t["trechos_nao_encontrados"].apply(len)
    t["com_termos"] = t["termos_frequencia"].apply(len) > 0

    resumo = t.groupby("modelo").agg(
        respostas=("id", "size"),
        comportamento_correto=("comportamento_correto", "mean"),
        citacoes_invalidas=("n_invalidas", "sum"),
        trechos_nao_encontrados=("n_trechos", "sum"),
        respostas_com_termos_frequencia=("com_termos", "sum"),
        tokens_saida_medio=("tokens_saida", "mean"),
        segundos_mediana=("segundos", "median"),
    )
    print("\nRESUMO POR MODELO:")
    print(resumo.round(2).T.to_string())

    print("\nComportamento correto por tipo de pergunta:")
    print(t.pivot_table(index="tipo", columns="modelo", values="comportamento_correto",
                        aggfunc="mean").round(2).to_string())

    errados = t[~t["comportamento_correto"]]
    if not errados.empty:
        print("\nPerguntas com comportamento errado:")
        for _, r in errados.iterrows():
            print(f"  [{r['id']}] {r['modelo'].split('/')[-1]:14s} esperado={r['esperado']:9s} "
                  f"recusou={r['recusou']} | {r['pergunta'][:60]}")
    if erros:
        print(f"\n{erros} chamada(s) terminaram em erro; rode o mesmo comando para refazê-las.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    conferir_modelos()
    perguntas = carregar_perguntas()
    PASTA.mkdir(parents=True, exist_ok=True)
    arquivo = PASTA / f"{CONFIG}.jsonl"

    feitos = {(r["id"], r["modelo"]) for r in ler_registros(arquivo) if "erro" not in r}
    pendentes = [(p["id"], m) for p in perguntas for m in MODELOS if (p["id"], m) not in feitos]
    print(f"Configuração {CONFIG}: {len(feitos)} respostas já salvas, {len(pendentes)} a gerar.",
          flush=True)

    if pendentes:
        print("Carregando o buscador...", flush=True)
        buscador = Buscador()
        ultima_chamada = {m: 0.0 for m in MODELOS}
        with arquivo.open("a", encoding="utf-8") as saida:
            for p in perguntas:
                modelos_pendentes = [m for m in MODELOS if (p["id"], m) not in feitos]
                if not modelos_pendentes:
                    continue
                # Uma busca por pergunta: as mesmas evidências para todos os modelos
                evid = buscador.buscar(p["pergunta"], filtros=p.get("filtros"), k_final=K)
                evidencias = json.loads(evid[COLUNAS_EVIDENCIA].to_json(orient="records",
                                                                        force_ascii=False))
                for modelo in modelos_pendentes:
                    espera = INTERVALO_POR_MODELO - (time.monotonic() - ultima_chamada[modelo])
                    if espera > 0:
                        time.sleep(espera)
                    ultima_chamada[modelo] = time.monotonic()

                    registro = {"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                                "agregada": p.get("agregada", False), "pergunta": p["pergunta"],
                                "filtros": p.get("filtros"), "config": CONFIG, "modelo": modelo,
                                "documentos_no_filtro": int(evid.attrs.get("documentos_no_filtro", 0))}
                    try:
                        r = gerar_resposta(p["pergunta"], evid, modelo=modelo)
                        registro.update({c: r[c] for c in CAMPOS_RESULTADO})
                        registro["comportamento_correto"] = comportamento_correto(
                            p["esperado"], r["recusou"])
                        registro["evidencias"] = evidencias
                        status = f"{r['segundos']} s | recusou={r['recusou']}"
                    except Exception as erro:  # registra e segue; a próxima execução refaz
                        registro["erro"] = str(erro)
                        status = f"ERRO: {erro}"
                    saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    saida.flush()
                    print(f"[{p['id']}] {modelo.split('/')[-1]:14s} {status}", flush=True)

    imprimir_resumo(arquivo)


if __name__ == "__main__":
    main()