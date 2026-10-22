"""Etapa 3 — Pergunta ao sistema pelo terminal: resposta + evidências citadas + verificações.

Exemplos (na raiz do projeto):
    python -m src.rag.perguntar "O que os clientes relatam sobre atrasos na entrega?"
    python -m src.rag.perguntar "Quais problemas aparecem?" --categoria "cama mesa banho" --nota-max 2
    python -m src.rag.perguntar "O que os clientes elogiam?" --uf RJ --de 20180101 --ate 20180630

O modelo padrão é o LLM_MODELO do .env; use --modelo para trocar.
"""
import argparse
import sys
import textwrap

from src.rag.geracao import gerar_resposta
from src.rag.recuperacao import Buscador

SITUACOES_ENTREGA = ["no prazo", "atrasado", "não entregue"]


def montar_filtros(args: argparse.Namespace) -> dict:
    """Converte as opções do terminal na linguagem de filtros do Buscador."""
    filtros: dict = {}
    if args.categoria:
        filtros["categoria_principal"] = args.categoria
    if args.uf:
        filtros["uf"] = args.uf.upper()
    if args.regiao:
        filtros["regiao"] = args.regiao
    if args.entrega:
        filtros["situacao_entrega"] = args.entrega

    nota = {}
    if args.nota_min is not None:
        nota["$gte"] = args.nota_min
    if args.nota_max is not None:
        nota["$lte"] = args.nota_max
    if nota:
        filtros["nota"] = nota

    periodo = {}
    if args.de is not None:
        periodo["$gte"] = args.de
    if args.ate is not None:
        periodo["$lte"] = args.ate
    if periodo:
        filtros["data_compra"] = periodo
    return filtros


def ler_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pergunte às avaliações dos clientes da Olist (RAG).")
    p.add_argument("pergunta", help="pergunta em linguagem natural, entre aspas")
    p.add_argument("--categoria", help='categoria do produto, ex.: "cama mesa banho"')
    p.add_argument("--uf", help="estado do cliente, ex.: RJ")
    p.add_argument("--regiao", help="Norte, Nordeste, Centro-Oeste, Sudeste ou Sul")
    p.add_argument("--entrega", choices=SITUACOES_ENTREGA, help="situação da entrega")
    p.add_argument("--nota-min", type=int, choices=range(1, 6), help="nota mínima (1 a 5)")
    p.add_argument("--nota-max", type=int, choices=range(1, 6), help="nota máxima (1 a 5)")
    p.add_argument("--de", type=int, help="data da compra a partir de (AAAAMMDD)")
    p.add_argument("--ate", type=int, help="data da compra até (AAAAMMDD)")
    p.add_argument("--k", type=int, default=20, help="quantas avaliações enviar ao LLM (padrão 20)")
    p.add_argument("--modelo", help="modelo do LLM (padrão: LLM_MODELO do .env)")
    p.add_argument("--modo", choices=["densa", "hibrida"], default="densa",
                   help="tipo de busca (padrão: densa)")
    return p.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # acentos e aspas curvas no terminal do Windows
    args = ler_argumentos()
    filtros = montar_filtros(args)

    print("Carregando modelo e índices...", flush=True)
    buscador = Buscador()
    evidencias = buscador.buscar(args.pergunta, filtros=filtros, k_final=args.k, modo=args.modo)
    if evidencias.empty:
        print("\nNenhuma avaliação com texto encontrada neste recorte. Tente filtros mais amplos.")
        return

    r = gerar_resposta(args.pergunta, evidencias, modelo=args.modelo)

    linha = "=" * 80
    print(f"\n{linha}\nPERGUNTA: {args.pergunta}\n{linha}\n\n{r['resposta']}\n")

    print(f"{linha}\nEVIDÊNCIAS CITADAS\n{linha}")
    validas = [c for c in r["citacoes"] if c in set(evidencias["ref"])]
    if not validas:
        print("(nenhuma)")
    for ref, e in evidencias.set_index("ref").loc[validas].iterrows():
        data = str(e["data_compra"])
        print(f"[{ref}] review_id {e['review_id']} | nota {e['nota']} | "
              f"{e['categoria_principal']} | compra {data[:4]}-{data[4:6]} | {e['uf']}")
        print(textwrap.indent(textwrap.fill(e["texto"], 74), "      "))

    print(f"\n{linha}\nVERIFICAÇÕES AUTOMÁTICAS\n{linha}")
    print(f"  modelo: {r['modelo']} | {r['segundos']} s | tokens entrada/saída: "
          f"{r['tokens_entrada']}/{r['tokens_saida']}")
    print(f"  recorte: {evidencias.attrs['documentos_no_filtro']:,} avaliações com texto | "
          f"enviadas ao LLM: {len(evidencias)}")
    print(f"  recusou: {'sim' if r['recusou'] else 'não'}")
    print(f"  citações inválidas: {', '.join(r['citacoes_invalidas']) or 'nenhuma'}")
    print(f"  trechos não encontrados: {' | '.join(r['trechos_nao_encontrados']) or 'nenhum'}")
    print(f"  termos de frequência: {', '.join(r['termos_frequencia']) or 'nenhum'}")


if __name__ == "__main__":
    main()