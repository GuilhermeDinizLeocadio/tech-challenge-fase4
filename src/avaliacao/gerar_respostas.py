"""Etapas 4 e 5 — Gera as respostas dos redatores para as perguntas de teste e salva em arquivo.

Cada configuração define a busca (de src/avaliacao/configuracoes.py), a versão do prompt e os
modelos. Para cada pergunta, a busca é feita UMA vez por tipo de busca, e as mesmas evidências vão
para todos os modelos e versões de prompt (comparação justa). As chamadas respeitam um intervalo
mínimo por modelo (limite de tokens por minuto da camada gratuita do Groq). Cada resposta é salva
assim que chega, junto com as evidências usadas; rodar de novo pula o que já foi gerado.

Uso (na raiz do projeto), com uma ou mais configurações:
    python -m src.avaliacao.gerar_respostas reescrita_rerank_k20_prompt_v2 reescrita_rerank_k20_prompt_v3
"""
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import pandas as pd

from src.avaliacao.configuracoes import executar
from src.avaliacao.validar_perguntas import carregar_perguntas
from src.rag.geracao import criar_cliente, gerar_resposta
from src.rag.recuperacao import Buscador

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "resultados" / "respostas"
K = 20
INTERVALO_POR_MODELO = 25  # segundos entre duas chamadas ao mesmo modelo
QWEN, GPT_OSS = "qwen/qwen3.8-27b", "openai/gpt-oss-120b"

CONFIGS = {
    # Linha de base (Etapa 4)
    "densa_k20_prompt_v2": {"busca": "densa", "versao_prompt": "v2", "modelos": [QWEN, GPT_OSS]},
    # Etapa 5: busca nova com o prompt antigo (isola o efeito da busca)...
    "reescrita_rerank_k20_prompt_v2": {"busca": "densa_reescrita_rerank", "versao_prompt": "v2",
                                       "modelos": [QWEN]},
    # ...e com o prompt novo (isola o efeito do prompt)
    "reescrita_rerank_k20_prompt_v3": {"busca": "densa_reescrita_rerank", "versao_prompt": "v3",
                                       "modelos": [QWEN]},
}

COLUNAS_EVIDENCIA = ["ref", "review_id", "chave_texto", "texto", "nota", "categoria_principal",
                     "situacao_entrega", "dias_atraso", "data_compra", "uf", "repeticoes",
                     "score_rerank"]
CAMPOS_RESULTADO = ["versao_prompt", "resposta", "recusou", "citacoes", "citacoes_invalidas",
                    "trechos_nao_encontrados", "registro_como_relato", "trechos_da_pergunta",
                    "termos_frequencia", "tokens_entrada", "tokens_saida", "segundos"]


def conferir_modelos(modelos: set[str]) -> None:
    """Erra cedo e com clareza se algum modelo não existir no Groq."""
    disponiveis = {m.id for m in criar_cliente("groq").models.list()}
    faltando = sorted(modelos - disponiveis)
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


def imprimir_resumo(config: str) -> None:
    registros = ler_registros(PASTA / f"{config}.jsonl")
    erros = sum("erro" in r for r in registros)
    t = (pd.DataFrame([r for r in registros if "erro" not in r])
           .drop_duplicates(["id", "modelo"], keep="last"))
    if t.empty:
        print(f"\n{config}: nenhuma resposta salva.")
        return
    contar = lambda coluna: t.get(coluna, pd.Series([[]] * len(t))).apply(
        lambda v: len(v) if isinstance(v, list) else 0)
    t["n_invalidas"] = contar("citacoes_invalidas")
    t["n_trechos"] = contar("trechos_nao_encontrados")
    t["n_registro"] = contar("registro_como_relato")
    t["com_termos"] = contar("termos_frequencia") > 0

    resumo = t.groupby("modelo").agg(
        respostas=("id", "size"),
        comportamento_correto=("comportamento_correto", "mean"),
        citacoes_invalidas=("n_invalidas", "sum"),
        trechos_nao_encontrados=("n_trechos", "sum"),
        registro_como_relato=("n_registro", "sum"),
        respostas_com_termos_frequencia=("com_termos", "sum"),
        tokens_saida_medio=("tokens_saida", "mean"),
        segundos_mediana=("segundos", "median"),
    )
    print(f"\nRESUMO — {config}:")
    print(resumo.round(2).T.to_string())
    errados = t[~t["comportamento_correto"]]
    for _, r in errados.iterrows():
        print(f"  comportamento errado: [{r['id']}] {r['modelo'].split('/')[-1]} "
              f"esperado={r['esperado']} recusou={r['recusou']} | {r['pergunta'][:60]}")
    if erros:
        print(f"  {erros} chamada(s) terminaram em erro; rode o mesmo comando para refazê-las.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    configs = sys.argv[1:]
    desconhecidas = [c for c in configs if c not in CONFIGS]
    if not configs or desconhecidas:
        print(f"Informe uma ou mais configurações entre: {', '.join(CONFIGS)}")
        return
    conferir_modelos({m for c in configs for m in CONFIGS[c]["modelos"]})
    perguntas = carregar_perguntas()
    PASTA.mkdir(parents=True, exist_ok=True)

    feitos = {c: {(r["id"], r["modelo"]) for r in ler_registros(PASTA / f"{c}.jsonl")
                  if "erro" not in r} for c in configs}
    pendentes = sum(1 for c in configs for p in perguntas for m in CONFIGS[c]["modelos"]
                    if (p["id"], m) not in feitos[c])
    print(f"Configurações: {', '.join(configs)} | {pendentes} respostas a gerar.", flush=True)

    if pendentes:
        print("Carregando o buscador...", flush=True)
        buscador = Buscador()
        ultima_chamada: dict[str, float] = {}
        with ExitStack() as pilha:
            saidas = {c: pilha.enter_context((PASTA / f"{c}.jsonl").open("a", encoding="utf-8"))
                      for c in configs}
            for p in perguntas:
                evidencias_por_busca: dict[str, pd.DataFrame] = {}
                for c in configs:
                    cfg = CONFIGS[c]
                    for modelo in cfg["modelos"]:
                        if (p["id"], modelo) in feitos[c]:
                            continue
                        if cfg["busca"] not in evidencias_por_busca:  # uma busca por tipo
                            evidencias_por_busca[cfg["busca"]] = executar(buscador, p, cfg["busca"], K)
                        evid = evidencias_por_busca[cfg["busca"]]

                        espera = INTERVALO_POR_MODELO - (time.monotonic() - ultima_chamada.get(modelo, 0.0))
                        if espera > 0:
                            time.sleep(espera)
                        ultima_chamada[modelo] = time.monotonic()

                        registro = {"id": p["id"], "tipo": p["tipo"], "esperado": p["esperado"],
                                    "agregada": p.get("agregada", False), "pergunta": p["pergunta"],
                                    "filtros": p.get("filtros"), "config": c, "modelo": modelo,
                                    "busca": cfg["busca"], "consulta": evid.attrs.get("consulta"),
                                    "documentos_no_filtro": int(evid.attrs.get("documentos_no_filtro", 0))}
                        try:
                            r = gerar_resposta(p["pergunta"], evid, modelo=modelo,
                                               versao_prompt=cfg["versao_prompt"])
                            registro.update({campo: r[campo] for campo in CAMPOS_RESULTADO})
                            registro["comportamento_correto"] = comportamento_correto(
                                p["esperado"], r["recusou"])
                            colunas = [col for col in COLUNAS_EVIDENCIA if col in evid.columns]
                            registro["evidencias"] = json.loads(
                                evid[colunas].to_json(orient="records", force_ascii=False))
                            status = f"{r['segundos']} s | recusou={r['recusou']}"
                        except Exception as erro:  # registra e segue; a próxima execução refaz
                            registro["erro"] = str(erro)
                            status = f"ERRO: {erro}"
                        saidas[c].write(json.dumps(registro, ensure_ascii=False) + "\n")
                        saidas[c].flush()
                        print(f"[{p['id']}] {c} | {modelo.split('/')[-1]} | {status}", flush=True)

    for c in configs:
        imprimir_resumo(c)


if __name__ == "__main__":
    main()