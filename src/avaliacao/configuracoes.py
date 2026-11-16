"""Etapa 5 — Configurações de busca comparadas na avaliação.

Compartilhadas pela rotulagem de relevância e pelas métricas, para que as duas usem exatamente
as mesmas buscas. "reescrever" usa as reescritas salvas em resultados/reescritas.json (a avaliação
não chama o LLM para reescrever: usa sempre a mesma reescrita).
"""
from src.rag.reescrita import carregar_reescritas, montar_consulta

CONFIGURACOES = {
    "densa": {"modo": "densa"},
    "hibrida_ingenua": {"modo": "hibrida"},
    "hibrida_corrigida": {"modo": "hibrida_corrigida"},
    "densa_rerank": {"modo": "densa", "reranquear": True},
    "hibrida_rerank": {"modo": "hibrida", "reranquear": True},
    "hibrida_corrigida_rerank": {"modo": "hibrida_corrigida", "reranquear": True},
    "densa_reescrita_rerank": {"modo": "densa", "reranquear": True, "reescrever": True},
    "hibrida_corrigida_reescrita_rerank": {"modo": "hibrida_corrigida", "reranquear": True,
                                           "reescrever": True},
}

# Configurações criadas na Etapa 5: as avaliações que elas mandariam ao LLM entram na rotulagem
NOVAS = ["hibrida_corrigida", "hibrida_corrigida_rerank", "densa_reescrita_rerank",
         "hibrida_corrigida_reescrita_rerank"]

_reescritas: dict | None = None


def executar(buscador, pergunta: dict, nome: str, k: int):
    """Roda a configuração `nome` para uma pergunta de teste e devolve as k evidências."""
    global _reescritas
    opcoes = dict(CONFIGURACOES[nome])
    consulta = None
    if opcoes.pop("reescrever", False):
        if _reescritas is None:
            _reescritas = carregar_reescritas()
        salva = _reescritas.get(pergunta["id"])
        if salva is None or salva["pergunta"] != pergunta["pergunta"]:
            raise RuntimeError(f"Sem reescrita salva para {pergunta['id']}: "
                               "rode python -m src.rag.reescrita")
        consulta = montar_consulta(pergunta["pergunta"], salva["reescrita"])
    return buscador.buscar(pergunta["pergunta"], filtros=pergunta.get("filtros"), k_final=k,
                           consulta=consulta, **opcoes)