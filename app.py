"""Etapa 8 — Interface (Streamlit) do Voice of Customer Intelligence com RAG.

A interface NÃO tem lógica própria: ela só chama src.rag.pipeline.responder, o MESMO fluxo avaliado
nas Etapas 5 a 7 (roteador → panorama de temas, nas perguntas agregadas, ou reescrita → busca →
reranker → resposta → correção, nas específicas). Assim, os números da avaliação valem para o que
o usuário vê na tela.

Uso (na raiz do projeto):
    streamlit run app.py
"""
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.rag.geracao import descrever_filtros
from src.rag.pipeline import responder
from src.rag.recuperacao import CORPUS, Buscador
from src.temas.panorama import Panorama

RAIZ = Path(__file__).resolve().parent
TABELA = RAIZ / "resultados" / "tabela_comparativa.csv"

EXEMPLOS = {
    "Principais problemas": "Quais são os principais problemas relatados pelos clientes?",
    "Atendimento": "O que os clientes relatam sobre o atendimento ao tentar resolver um problema?",
    "Correios": "O que os clientes relatam sobre os Correios?",
    "Fora do escopo": "Qual foi o faturamento da Olist em 2018?",
}

st.set_page_config(page_title="Voz do Cliente · Olist", page_icon="💬", layout="wide")


# ---------------------------------------------------------------- formatação em português
def numero(n: float, casas: int = 0) -> str:
    """3255 -> '3.255'; 22.5 (casas=1) -> '22,5'."""
    texto = f"{n:,.{casas}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def separar_paragrafos(texto: str) -> str:
    """No Markdown, uma quebra de linha simples não separa parágrafos: linhas que começam com
    negrito (ex.: "**Motivo:**" na recusa) ganham uma linha em branco antes."""
    return re.sub(r"(?<!\n)\n(?=\*\*)", "\n\n", texto)


# ---------------------------------------------------------------- carregamento (uma única vez)
@st.cache_resource(show_spinner="Carregando modelos e índices (só na primeira vez, ~15 s)...")
def carregar() -> tuple[Buscador, Panorama]:
    buscador = Buscador()
    _ = buscador.reranker  # carrega o reranker agora, e não no meio da primeira pergunta
    return buscador, Panorama()


@st.cache_data
def opcoes_filtros() -> dict:
    c = pd.read_parquet(CORPUS, columns=["categoria_principal", "uf", "regiao", "data_compra"])
    return {"categorias": sorted(c["categoria_principal"].dropna().unique()),
            "ufs": sorted(c["uf"].dropna().unique()),
            "regioes": sorted(c["regiao"].dropna().unique()),
            "data_min": int(c["data_compra"].min()), "data_max": int(c["data_compra"].max())}


def para_data(n: int) -> date:
    s = str(n)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def para_inteiro(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


# ---------------------------------------------------------------- barra lateral
def barra_lateral(op: dict) -> tuple[dict, dict]:
    st.sidebar.header("Recorte")
    st.sidebar.caption("Os filtros só valem quando alterados.")
    filtros: dict = {}
    categoria = st.sidebar.selectbox("Categoria do produto", ["(todas)"] + op["categorias"])
    if categoria != "(todas)":
        filtros["categoria_principal"] = categoria
    uf = st.sidebar.selectbox("Estado (UF)", ["(todos)"] + op["ufs"])
    if uf != "(todos)":
        filtros["uf"] = uf
    regiao = st.sidebar.selectbox("Região", ["(todas)"] + op["regioes"])
    if regiao != "(todas)":
        filtros["regiao"] = regiao
    entrega = st.sidebar.selectbox("Situação da entrega (registro do sistema)",
                                   ["(todas)", "no prazo", "atrasado", "não entregue"])
    if entrega != "(todas)":
        filtros["situacao_entrega"] = entrega
    nota_min, nota_max = st.sidebar.slider("Nota da avaliação", 1, 5, (1, 5))
    if (nota_min, nota_max) != (1, 5):
        filtros["nota"] = {"$gte": nota_min, "$lte": nota_max}
    if st.sidebar.checkbox("Filtrar por período da compra"):
        inicio, fim = para_data(op["data_min"]), para_data(op["data_max"])
        periodo = st.sidebar.date_input("Período", (inicio, fim), min_value=inicio,
                                        max_value=fim, format="DD/MM/YYYY")
        if isinstance(periodo, tuple) and len(periodo) == 2:
            filtros["data_compra"] = {"$gte": para_inteiro(periodo[0]),
                                      "$lte": para_inteiro(periodo[1])}

    with st.sidebar.expander("Configuração avançada"):
        st.caption("Desligue técnicas para ver o efeito de cada uma (padrão: todas ligadas).")
        opcoes = {"rotear": st.checkbox("Roteador (visão do conjunto x assunto específico)", True),
                  "reescrever": st.checkbox("Reescrita da pergunta", True),
                  "reranquear": st.checkbox("Reranker", True),
                  "corrigir": st.checkbox("Correção guiada pelo verificador", True),
                  "k": st.slider("Avaliações enviadas ao LLM", 5, 30, 20)}
    return filtros, opcoes


# ---------------------------------------------------------------- partes da resposta
def mostrar_panorama(r: dict) -> None:
    for p in r["panorama"]:
        nome = "Problemas (notas 1 a 3)" if p["grupo"] == "problema" else "Elogios (notas 4 e 5)"
        st.markdown(f"**{nome}** — {numero(p['total'])} avaliações com texto no recorte")
        tabela = pd.DataFrame([{"Tema": t["nome"] + (" (fora do ranking)" if t["mistos"] else ""),
                                "Avaliações": numero(t["n"]), "%": numero(t["pct"], 1) + "%",
                                "Participação": t["pct"]} for t in p["temas"]])
        st.dataframe(tabela, hide_index=True, column_config={
            "Participação": st.column_config.ProgressColumn("Participação", min_value=0,
                                                            max_value=100, format="%.0f%%")})
    st.caption("Temas atribuídos automaticamente por agrupamento; validação humana numa amostra: "
               "90% de acerto nos problemas e 60% nos elogios.")


def mostrar_evidencias(evidencias: pd.DataFrame, r: dict) -> None:
    citadas = [c for c in r["citacoes"] if c in set(evidencias["ref"])]
    if not citadas:
        st.write("Nenhuma avaliação citada.")
        return
    for ref, e in evidencias.set_index("ref").loc[citadas].iterrows():
        data = str(e["data_compra"])
        tema = f" · tema: {e['tema_nome']}" if "tema_nome" in e and pd.notna(e["tema_nome"]) else ""
        st.markdown(f"**[{ref}]** nota {e['nota']} · {e['categoria_principal']} · "
                    f"compra {data[:4]}-{data[4:6]} · {e['uf']}{tema}")
        st.markdown(f"> {e['texto']}")


def mostrar_verificacoes(r: dict, agregada: bool) -> None:
    itens = [
        ("Todas as citações existem no contexto", r["citacoes_invalidas"]),
        ("Trechos entre aspas encontrados nas avaliações", r["trechos_nao_encontrados"]),
        ("Registro do sistema não apresentado como fala do cliente", r["registro_como_relato"]),
    ]
    if agregada:
        itens.append(("Todos os números conferem com o panorama",
                      r.get("numeros_fora_do_panorama", [])))
    else:
        itens.append(("Sem termos de frequência sobre a amostra", r["termos_frequencia"]))
        itens.append(("Sem contagens dentro da amostra", r.get("contagens", [])))
    for nome, problemas in itens:
        st.markdown(f"{'✅' if not problemas else '⚠️'} {nome}"
                    + (f" — {', '.join(problemas)}" if problemas else ""))
    if not agregada and r.get("correcao") == "aplicada":
        st.markdown("🔧 Correção automática aplicada: expressões de frequência foram reescritas "
                    "como afirmações de existência.")


def mostrar_saida(saida: dict) -> None:
    r, evidencias, rota = saida["resultado"], saida["evidencias"], saida["rota"]
    if r is None:
        st.warning("Nenhuma avaliação com texto neste recorte. Tente ampliar os filtros.")
        return
    agregada = rota["tipo"] == "agregada"
    c1, c2, c3 = st.columns(3)
    c1.metric("Rota", "Visão do conjunto" if agregada else "Assunto específico")
    c2.metric("Avaliações com texto no recorte",
              numero(int(evidencias.attrs.get("documentos_no_filtro", 0))))
    c3.metric("Tempo total", f"{numero(saida['segundos_total'], 1)} s")
    if r["recusou"]:
        st.info("O sistema recusou: as avaliações disponíveis não sustentam uma resposta.")

    st.markdown(separar_paragrafos(r["resposta"]))

    if agregada:
        with st.expander("Panorama de temas (contagem real no recorte)"):
            mostrar_panorama(r)
    n_citadas = len([c for c in r["citacoes"] if c in set(evidencias["ref"])])
    with st.expander(f"Avaliações citadas ({n_citadas})"):
        mostrar_evidencias(evidencias, r)
    with st.expander("Verificações automáticas"):
        mostrar_verificacoes(r, agregada)
    if saida["reescrita"]:
        with st.expander("Como a pergunta foi reescrita para a busca"):
            st.write(saida["reescrita"])


# ---------------------------------------------------------------- abas
def aba_perguntar(filtros: dict, opcoes: dict) -> None:
    st.write("Experimente um exemplo:")
    for coluna, (rotulo, texto) in zip(st.columns(len(EXEMPLOS)), EXEMPLOS.items()):
        if coluna.button(rotulo):
            st.session_state["pergunta"] = texto
    pergunta = st.text_area("Sua pergunta", key="pergunta", height=80,
                            placeholder="Ex.: O que os clientes relatam sobre produtos que "
                                        "chegaram quebrados?")
    if filtros:
        st.caption(f"Recorte ativo: {descrever_filtros(filtros)}")

    if st.button("Perguntar", type="primary", disabled=not pergunta.strip()):
        buscador, panorama = carregar()
        with st.spinner("Analisando as avaliações..."):
            inicio = time.perf_counter()
            try:
                saida = responder(buscador, pergunta.strip(), filtros=filtros or None,
                                  k=opcoes["k"], reescrever=opcoes["reescrever"],
                                  reranquear=opcoes["reranquear"], corrigir=opcoes["corrigir"],
                                  rotear=opcoes["rotear"], panorama=panorama)
                saida["segundos_total"] = time.perf_counter() - inicio
                st.session_state["saida"] = saida
            except Exception as erro:
                st.session_state.pop("saida", None)
                st.error(f"Não foi possível responder agora ({erro}). Se for limite de uso da "
                         "API gratuita, aguarde alguns minutos e tente de novo.")
    if "saida" in st.session_state:
        st.divider()
        mostrar_saida(st.session_state["saida"])


def aba_avaliacao() -> None:
    st.subheader("Como o sistema foi avaliado")
    st.markdown(
        "- **31 perguntas de teste** com gabarito de comportamento (responder, recusar, ressalvar), "
        "incluindo perguntas fora do escopo e sem evidência na base.\n"
        "- **Busca:** 2.462 rótulos de relevância de um juiz LLM validado por amostra humana. "
        "Precisão@20 de 0,821 (linha de base) para 0,935 (reescrita + reranker).\n"
        "- **Respostas:** cada afirmação conferida por um juiz LLM contra as avaliações citadas, "
        "combinada com verificadores automáticos (fidelidade ajustada).\n"
        "- **Generalização:** 8 perguntas novas, congeladas antes de rodar, executadas ao vivo.")
    if not TABELA.exists():
        st.info("Tabela de resultados não encontrada (resultados/tabela_comparativa.csv).")
        return
    t = pd.read_csv(TABELA)
    nomes = {"densa_k20_prompt_v2": "Linha de base (Etapa 4)",
             "final": "Sistema final (31 perguntas)", "holdout": "Perguntas novas (8)"}
    colunas = {"comportamento_correto_%": "Comportamento correto (%)",
               "fidelidade_ajustada_%": "Fidelidade ajustada (%)",
               "nao_sustentadas_%": "Afirmações não sustentadas (%)",
               "registro_como_relato_v2": "Registro citado como fala do cliente",
               "respostas_com_termos_freq": "Respostas com termos de frequência",
               "respostas_com_contagem": "Respostas com contagens na amostra",
               "numeros_fora_do_panorama": "Números fora do panorama"}
    selecao = t[(t["config"].isin(["final", "holdout"]))
                | ((t["config"] == "densa_k20_prompt_v2") & (t["modelo"] == "qwen3.8-27b"))].copy()
    # A linha de base não tinha panorama: a métrica não se aplica
    selecao.loc[selecao["config"] == "densa_k20_prompt_v2", "numeros_fora_do_panorama"] = None
    tabela = (selecao.set_index("config").loc[list(nomes), list(colunas)]
              .rename(index=nomes, columns=colunas).T)
    tabela = tabela.map(lambda v: "—" if pd.isna(v)
                        else numero(v, 0 if float(v).is_integer() else 1))
    st.dataframe(tabela)
    st.caption("Detalhes e todas as configurações intermediárias: docs/registro_decisoes.md e "
               "resultados/tabela_comparativa.csv.")


# ---------------------------------------------------------------- página
st.title("💬 Voz do Cliente · Olist")
st.caption("Pergunte às 42 mil avaliações com texto da base Olist (2016–2018). As respostas citam "
           "as avaliações [R#], separam o relato do cliente do registro do sistema e são "
           "conferidas automaticamente. Perguntas sobre o conjunto usam contagens reais por tema.")
filtros_ativos, opcoes_ativas = barra_lateral(opcoes_filtros())
perguntar, avaliacao = st.tabs(["Perguntar", "Como o sistema foi avaliado"])
with perguntar:
    aba_perguntar(filtros_ativos, opcoes_ativas)
with avaliacao:
    aba_avaliacao()