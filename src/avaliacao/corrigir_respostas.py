"""Etapa 5 — Aplica a correção guiada pelo verificador às respostas JÁ SALVAS de uma configuração.

Usa as mesmas perguntas, evidências e respostas: a única diferença entre a configuração de origem
e a corrigida é a correção, o que isola o seu efeito. Só chama o LLM para respostas com termos de
frequência ou contagens. Salva em resultados/respostas/<origem>_corrigida.jsonl; rodar de novo
pula o que já foi feito.

Uso (na raiz do projeto):
    python -m src.avaliacao.corrigir_respostas reescrita_rerank_k20_prompt_v4
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src.rag.geracao import corrigir_resposta

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "resultados" / "respostas"
INTERVALO = 20  # segundos entre chamadas de correção (são mais curtas que as de geração)


def ler(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    return [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 2:
        print("Informe a configuração de origem, ex.: reescrita_rerank_k20_prompt_v4")
        return
    origem = sys.argv[1]
    destino = f"{origem}_corrigida"
    respostas = {(r["id"], r["modelo"]): r for r in ler(PASTA / f"{origem}.jsonl")
                 if "erro" not in r}
    feitos = {(r["id"], r["modelo"]) for r in ler(PASTA / f"{destino}.jsonl") if "erro" not in r}
    pendentes = [r for chave, r in respostas.items() if chave not in feitos]
    print(f"{origem} → {destino}: {len(respostas)} respostas | {len(pendentes)} pendentes",
          flush=True)

    situacoes = Counter()
    ultima = 0.0
    with (PASTA / f"{destino}.jsonl").open("a", encoding="utf-8") as saida:
        for r in pendentes:
            evidencias = pd.DataFrame(r["evidencias"])
            registro = {**r, "config": destino}
            try:
                if not r["recusou"]:
                    espera = INTERVALO - (time.monotonic() - ultima)
                    if espera > 0:
                        time.sleep(espera)
                novo = corrigir_resposta(r["pergunta"], evidencias, r, modelo=r["modelo"])
                if novo["correcao"] != "não necessária":
                    ultima = time.monotonic()
                registro.update({**novo, "config": destino})
                status = novo["correcao"]
                if "problemas_antes" in novo:
                    status += f" | antes: {', '.join(novo['problemas_antes'])}"
            except Exception as erro:  # registra e segue; a próxima execução refaz
                registro["erro"] = str(erro)
                status = f"ERRO: {erro}"
            situacoes[status.split(" |")[0]] += 1
            saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            saida.flush()
            print(f"[{r['id']}] {status}", flush=True)

    print("\nResumo:", dict(situacoes))


if __name__ == "__main__":
    main()