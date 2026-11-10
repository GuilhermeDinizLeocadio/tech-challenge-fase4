"""Etapa 5 — Reescrita da pergunta na linguagem dos clientes, para a primeira etapa da busca.

O gestor pergunta em termos abstratos ("atendimento ao tentar resolver um problema"); os clientes
escrevem em termos concretos ("liguei várias vezes e ninguém resolveu"). O LLM escreve 3 frases
como clientes escreveriam, e a busca usa a pergunta + essas frases. O reranker continua julgando
pela pergunta original, para não se desviar da intenção do gestor.

Prompt v2 (revisado uma vez, por inspeção, antes de medir): mantém a polaridade pedida, mantém os
nomes e termos da pergunta e não reescreve perguntas que não tratam de experiências de clientes.
A v1 está guardada em resultados/reescritas_v1.json.

Para a avaliação, as reescritas das perguntas de teste ficam salvas (o LLM não é determinístico).

Uso (na raiz do projeto):
    python -m src.rag.reescrita      (gera e salva as reescritas das perguntas de teste)
"""
import json
import re
import sys
import time
from pathlib import Path

from src.rag.geracao import criar_cliente

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO = RAIZ / "resultados" / "reescritas.json"
MODELO = "qwen/qwen3.8-27b"
VERSAO = "v2"
INTERVALO = 8  # segundos entre chamadas (limite de tokens por minuto do Groq)
SEM_REESCRITA = "SEM_REESCRITA"

PROMPT_REESCRITA = f"""Você ajuda a buscar avaliações de clientes de um e-commerce brasileiro.
Dada a pergunta de um gestor, escreva 3 frases curtas como clientes as escreveriam em avaliações
que ajudam a responder a pergunta.
Regras:
- linguagem de cliente: primeira pessoa, simples, como numa avaliação real;
- mantenha a mesma situação que a pergunta pede: se ela pergunta sobre problemas, escreva só
  problemas; se pergunta sobre elogios, só elogios; misture experiências boas e ruins apenas
  quando a pergunta for neutra (por exemplo, "como os clientes avaliam...");
- mantenha os nomes e termos específicos que aparecem na pergunta (por exemplo, nomes de empresas
  ou da transportadora);
- não invente números, datas nem nomes que não estejam na pergunta;
- se a pergunta citar um recorte (categoria, período, estado, nota), ignore o recorte e foque no
  conteúdo;
- se a pergunta NÃO tratar de experiências de clientes com compras, produtos, entregas,
  pagamento ou atendimento, responda apenas: {SEM_REESCRITA}
- uma frase por linha, sem numeração e sem comentários."""


def reescrever_pergunta(pergunta: str, cliente=None, modelo: str = MODELO) -> str:
    """Devolve as frases na linguagem dos clientes, unidas num só texto ("" = sem reescrita)."""
    cliente = cliente or criar_cliente("groq")
    resposta = cliente.chat.completions.create(
        model=modelo, temperature=0, max_tokens=600,
        messages=[{"role": "system", "content": PROMPT_REESCRITA},
                  {"role": "user", "content": pergunta}])
    texto = resposta.choices[0].message.content or ""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()  # raciocínio, se houver
    if SEM_REESCRITA in texto.upper():
        return ""
    frases = [linha.strip(" -•*\t\"") for linha in texto.splitlines() if linha.strip()]
    if not frases:
        raise ValueError("a reescrita veio vazia")
    return " ".join(frases[:3])


def montar_consulta(pergunta: str, reescrita: str) -> str:
    """Texto da primeira busca: a pergunta original seguida das frases dos clientes (se houver)."""
    return f"{pergunta} {reescrita}".strip() if reescrita else pergunta


def carregar_reescritas() -> dict[str, dict]:
    if not ARQUIVO.exists():
        return {}
    return json.loads(ARQUIVO.read_text(encoding="utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.avaliacao.validar_perguntas import carregar_perguntas

    reescritas = carregar_reescritas()
    cliente = criar_cliente("groq")
    for p in carregar_perguntas():
        salva = reescritas.get(p["id"], {})
        if salva.get("pergunta") == p["pergunta"] and salva.get("versao") == VERSAO:
            continue  # já existe para esta pergunta e esta versão do prompt
        for tentativa in range(3):
            try:
                texto = reescrever_pergunta(p["pergunta"], cliente)
                break
            except Exception as erro:  # limite de uso ou resposta vazia: espera e tenta de novo
                print(f"[{p['id']}] tentativa {tentativa + 1} falhou: {erro}", flush=True)
                time.sleep(20)
        else:
            print(f"[{p['id']}] sem reescrita; rode de novo depois", flush=True)
            continue
        reescritas[p["id"]] = {"pergunta": p["pergunta"], "reescrita": texto,
                               "modelo": MODELO, "versao": VERSAO}
        ARQUIVO.parent.mkdir(exist_ok=True)
        ARQUIVO.write_text(json.dumps(reescritas, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{p['id']}] {p['pergunta']}\n       → {texto or '(' + SEM_REESCRITA + ')'}",
              flush=True)
        time.sleep(INTERVALO)
    print(f"\n{len(reescritas)} reescritas salvas em {ARQUIVO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()