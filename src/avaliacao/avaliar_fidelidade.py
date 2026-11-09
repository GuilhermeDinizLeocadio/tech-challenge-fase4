"""Etapas 4 e 5 — Avalia a fidelidade das respostas salvas: o juiz confere cada afirmação contra as
avaliações que ela cita.

Para cada resposta que NÃO foi recusa, o juiz recebe a pergunta, o recorte, o MESMO contexto que
o redator recebeu (inclusive o mesmo formato, v2 ou v3) e a resposta. Ele lista as afirmações
factuais e dá um veredito a cada uma (sustentada, parcial ou não sustentada), além de avaliar a
resposta como um todo. Os resultados são salvos assim que chegam; rodar de novo pula o que já foi
avaliado.

Uso (na raiz do projeto), com uma ou mais configurações:
    python -m src.avaliacao.avaliar_fidelidade reescrita_rerank_k20_prompt_v2 reescrita_rerank_k20_prompt_v3
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.avaliacao.juiz import Juiz
from src.rag.geracao import descrever_filtros, formatar_contexto

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RESPOSTAS = RAIZ / "resultados" / "respostas"
PASTA_FIDELIDADE = RAIZ / "resultados" / "fidelidade"
VEREDITOS = {"sustentada", "parcial", "nao_sustentada"}

PROMPT_FIDELIDADE = """Você é um auditor de fidelidade de respostas geradas por um sistema que
responde perguntas usando avaliações de clientes de um e-commerce brasileiro.

Você recebe a pergunta, o recorte consultado, o CONTEXTO (avaliações numeradas [R#]; o cabeçalho
de cada uma é o REGISTRO do sistema e o texto entre aspas é o RELATO do cliente) e a RESPOSTA.

Tarefa:
1. Liste as afirmações factuais das seções "Resposta" e "O que os clientes relatam". Uma afirmação
   é uma ideia sobre o que os clientes relatam ou sobre o registro. Ignore a seção "Limitações",
   frases de formato e frases que apenas descrevem o recorte.
2. Para cada afirmação, verifique SOMENTE as avaliações citadas nela e dê um veredito:
   - "sustentada": as avaliações citadas dizem isso (paráfrase fiel é aceitável);
   - "parcial": parte é sustentada, mas há exagero, generalização, causa não relatada pelos
     clientes ou detalhe a mais; também é "parcial" quando um dado do CABEÇALHO aparece entre
     aspas como se fosse fala do cliente;
   - "nao_sustentada": as avaliações citadas não dizem isso, dizem o contrário, ou a afirmação
     não tem citação.
3. "fatos_novos_na_interpretacao": true se a seção "Interpretação" introduz fatos, números ou
   causas que não aparecem nas afirmações citadas; false caso contrário.
4. "responde_a_pergunta": nota de 1 a 5 (5 = responde diretamente ao que foi perguntado;
   3 = responde em parte ou a uma pergunta parecida; 1 = fala de outra coisa).
5. "declara_limitacao_de_evidencia": true se a resposta deixa claro que há poucas avaliações ou
   evidência limitada no recorte; false caso contrário.

Os nomes Stark, Lannister, Targaryen e Baratheon são empresas anonimizadas.
Seja rigoroso. Justificativas com no máximo 20 palavras.
Responda SOMENTE com JSON neste formato:
{"afirmacoes": [{"afirmacao": "...", "citacoes": ["R1", "R3"], "veredito": "sustentada",
"justificativa": "..."}], "fatos_novos_na_interpretacao": false, "responde_a_pergunta": 5,
"declara_limitacao_de_evidencia": false}"""


def ler_jsonl(arquivo: Path) -> list[dict]:
    if not arquivo.exists():
        return []
    return [json.loads(linha) for linha in arquivo.read_text(encoding="utf-8").splitlines()
            if linha.strip()]


def montar_mensagem(r: dict) -> str:
    contexto = formatar_contexto(pd.DataFrame(r["evidencias"]), r.get("versao_prompt", "v2"))
    return (f"PERGUNTA: {r['pergunta']}\n"
            f"RECORTE: {descrever_filtros(r.get('filtros'))} "
            f"({r['documentos_no_filtro']:,} avaliações com texto)\n\n"
            f"CONTEXTO ENTREGUE AO SISTEMA:\n{contexto}\n\n"
            f"RESPOSTA A AUDITAR:\n{r['resposta']}")


def normalizar(avaliacao: dict) -> dict:
    """Confere e padroniza a resposta do juiz; erro se não houver nenhuma afirmação válida."""
    afirmacoes = []
    for a in avaliacao.get("afirmacoes", []):
        veredito = (str(a.get("veredito", "")).strip().lower()
                    .replace("ã", "a").replace(" ", "_"))
        if veredito in VEREDITOS:
            afirmacoes.append({"afirmacao": str(a.get("afirmacao", "")),
                               "citacoes": [str(c) for c in a.get("citacoes", [])],
                               "veredito": veredito,
                               "justificativa": str(a.get("justificativa", ""))})
    if not afirmacoes:
        raise ValueError("o juiz não devolveu nenhuma afirmação com veredito válido")
    nota = avaliacao.get("responde_a_pergunta", 3)
    nota = min(5, max(1, int(nota))) if str(nota).strip().lstrip("-").isdigit() else 3
    return {"afirmacoes": afirmacoes,
            "fatos_novos_na_interpretacao": bool(avaliacao.get("fatos_novos_na_interpretacao", False)),
            "responde_a_pergunta": nota,
            "declara_limitacao_de_evidencia": bool(avaliacao.get("declara_limitacao_de_evidencia", False))}


def imprimir_resumo(config: str) -> None:
    t = pd.DataFrame([r for r in ler_jsonl(PASTA_FIDELIDADE / f"{config}.jsonl")
                      if "erro" not in r]).drop_duplicates(["id", "modelo"], keep="last")
    if t.empty:
        print(f"\n{config}: nenhuma auditoria salva.")
        return
    for v in VEREDITOS:
        t[v] = t["afirmacoes"].apply(lambda afs, v=v: sum(a["veredito"] == v for a in afs))
    t["total"] = t["afirmacoes"].apply(len)
    resumo = t.groupby("modelo").agg(
        respostas=("id", "size"), afirmacoes=("total", "sum"),
        sustentadas=("sustentada", "sum"), parciais=("parcial", "sum"),
        nao_sustentadas=("nao_sustentada", "sum"),
        interpretacao_com_fatos_novos=("fatos_novos_na_interpretacao", "sum"),
        responde_a_pergunta_media=("responde_a_pergunta", "mean"),
    )
    resumo["fidelidade_%"] = 100 * resumo["sustentadas"] / resumo["afirmacoes"]
    print(f"\nRESUMO DA FIDELIDADE — {config}:")
    print(resumo.round(2).T.to_string())


def auditar(config: str, juiz: Juiz | None) -> Juiz | None:
    respostas = [r for r in ler_jsonl(PASTA_RESPOSTAS / f"{config}.jsonl") if "erro" not in r]
    unicas = {(r["id"], r["modelo"]): r for r in respostas}  # fica a última de cada par
    a_avaliar = [r for r in unicas.values() if not r["recusou"]]
    saida_arquivo = PASTA_FIDELIDADE / f"{config}.jsonl"
    feitos = {(r["id"], r["modelo"]) for r in ler_jsonl(saida_arquivo) if "erro" not in r}
    pendentes = [r for r in a_avaliar if (r["id"], r["modelo"]) not in feitos]
    print(f"\n{config}: {len(a_avaliar)} respostas a auditar | {len(feitos)} já auditadas | "
          f"{len(pendentes)} pendentes", flush=True)
    if not pendentes:
        return juiz

    juiz = juiz or Juiz()
    print(f"Juiz: {juiz.modelo}", flush=True)
    saida_arquivo.parent.mkdir(parents=True, exist_ok=True)
    with saida_arquivo.open("a", encoding="utf-8") as saida:
        for r in pendentes:
            registro = {"id": r["id"], "modelo": r["modelo"], "tipo": r["tipo"],
                        "esperado": r["esperado"], "agregada": r["agregada"],
                        "config": config, "juiz": juiz.modelo}
            try:
                avaliacao = None
                for tentativa in range(2):  # uma nova tentativa se o formato vier inválido
                    try:
                        avaliacao = normalizar(juiz.avaliar_json(PROMPT_FIDELIDADE,
                                                                 montar_mensagem(r)))
                        break
                    except ValueError as erro:
                        print(f"   formato inválido (tentativa {tentativa + 1}): {erro}",
                              flush=True)
                if avaliacao is None:
                    raise ValueError("formato inválido após 2 tentativas")
                registro.update(avaliacao)
                vs = [a["veredito"] for a in avaliacao["afirmacoes"]]
                status = (f"afirmações {len(vs)} | sustentadas {vs.count('sustentada')} | "
                          f"parciais {vs.count('parcial')} | "
                          f"não sustentadas {vs.count('nao_sustentada')}")
            except Exception as erro:  # registra e segue; a próxima execução refaz
                registro["erro"] = str(erro)
                status = f"ERRO: {erro}"
            saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            saida.flush()
            print(f"[{r['id']}] {r['modelo'].split('/')[-1]:14s} {status}", flush=True)
    return juiz


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    configs = sys.argv[1:] or ["densa_k20_prompt_v2"]
    juiz = None
    for config in configs:
        juiz = auditar(config, juiz)
    for config in configs:
        imprimir_resumo(config)


if __name__ == "__main__":
    main()