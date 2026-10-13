# Registro de decisões e achados

Diário técnico do projeto. Cada entrada vira insumo para o README e para o vídeo.

## Etapa 0 — Ambiente e infraestrutura (07/10/2026)

### Ambiente
- **Python 3.13**, validado antes de começar com `scripts/testar_stack.py`: torch 2.14.1,
  sentence-transformers 6.1.0, chromadb 1.5.9, bm25s 0.3.13, streamlit 1.65.0. pandas 3.0.6.
- **`requirements.txt` fixa só as dependências diretas** (gerado por `scripts/fixar_versoes.py`):
  um `pip freeze` completo no Windows inclui pacotes específicos da plataforma que quebram a
  instalação em Linux/Mac.

### Provedores de LLM (camada gratuita)
- **Cliente único compatível com a API da OpenAI** para Groq, Gemini e Ollama: trocar de provedor
  é mudar uma linha do `.env`.
- **Modelos com nome fixo, nunca aliases `latest`**: apelidos mudam de versão sem aviso e
  quebram a reprodutibilidade.
- **Groq (gerador):** `openai/gpt-oss-120b` e `qwen/qwen3.8-27b` têm 30 req/min, 1 mil req/dia,
  8 mil tokens/min e 200 mil tokens/dia, contados por modelo.
  - O limite de 8 mil tokens/min define o teto de cada chamada e, portanto, o tamanho do contexto.
  - O limite diário obriga a avaliação a ser dividida em dois dias, com resultados salvos em disco.
  - Na mesma pergunta, o gpt-oss gastou 86/82 tokens (entrada/saída) contra 27/46 do Qwen, por
    causa do raciocínio oculto. Escolha final por teste comparativo na Etapa 3.
- **Gemini (juiz):** `gemini-3.5-flash-lite` (15 req/min, 500 req/dia). Os Flash "completos" têm
  só 20 req/dia, o que é insuficiente para corrigir a avaliação com um juiz único. O juiz será
  validado contra uma amostra corrigida manualmente.

### Dados
- **Licença CC BY-NC-SA 4.0:** os 6 CSVs usados estão versionados em `data/raw/`, com atribuição
  no README, para o avaliador não precisar de conta no Kaggle.
- **`.gitattributes` com `*.csv -text`:** impede o Git do Windows de alterar as quebras de linha
  dentro dos comentários, mantendo os dados idênticos byte a byte.
- **Nomes de empresas anonimizados:** a Olist trocou nomes de empresas e parceiros citados nos
  comentários por casas de Game of Thrones (ex.: "lojas lannister"). O LLM precisa ser avisado no
  prompt.

### Segurança
- Chaves só no `.env` (ignorado pelo Git), com `.env.example` como molde.
- Varredura antes de cada commit:
  `git diff --cached | Select-String "gsk_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{30,}|AQ\.[A-Za-z0-9_-]{20,}"`.
- **Melhoria contínua:** a primeira versão do padrão procurava só o prefixo das chaves
  (`gsk_`, `AIza`) e gerou um falso alarme ao encontrar o próprio comando documentado neste arquivo.
  O padrão atual exige o prefixo seguido de dezenas de caracteres, como numa chave real, e
  eliminou o falso positivo sem perder a detecção.

## Etapa 1 — EDA (13/10/2026)

### Volumes
| Tabela | Linhas | Observação |
|---|---|---|
| avaliações | 99.224 | 98.410 `review_id` únicos (814 linhas repetidas) |
| pedidos | 99.441 | 1 linha por pedido |
| itens | 112.650 | para 98.666 pedidos; pedidos com vários itens; ~775 pedidos sem item |
| categorias (tradução) | 71 | dicionário português → inglês |
| clientes | 99.441 | `customer_id` é por pedido; a pessoa real é `customer_unique_id` |

### Presença de texto
| Situação | Avaliações | % |
|---|---|---|
| Só título | 1.737 | 1,8 |
| Só mensagem | 31.121 | 31,4 |
| Título e mensagem | 9.829 | 9,9 |
| Nenhum texto | 56.537 | 57,0 |

- **42.687 avaliações (43%) têm algum texto** e são as candidatas a documento na busca.
- **Avaliações sem texto ficam fora da busca, mas entram nas estatísticas:** a nota continua
  sendo informação sobre o cliente.
- **27 mensagens e 2 títulos continham só espaços em branco:** não são nulos para o computador,
  mas estão vazios. Detectados com `.str.strip()` antes do teste de vazio; sem isso, entrariam na
  base como documentos em branco.


  ### Avaliações repetidas
- **789 `review_id` aparecem em 1.603 linhas (814 linhas extras).** Nota, título, mensagem e data
  são idênticos; só o `order_id` muda (até 3 pedidos por avaliação). É a mesma avaliação ligada a
  pedidos diferentes, não avaliações distintas.
- **Decisão: 1 documento por `review_id`, com a lista de pedidos ligados.** Indexar as cópias faria
  a mesma opinião contar mais de uma vez na recuperação e nas contagens.
- **Consequência:** os metadados do pedido (categoria, atraso) são agregados sobre todos os pedidos
  ligados à avaliação.
- **551 pedidos têm mais de uma avaliação** (`review_id` diferentes). São mantidas como opiniões
  distintas; o impacto é de cerca de 0,6% da base e fica registrado como limitação.

  ### Notas e viés de escrita
| Nota | % das avaliações | % com texto |
|---|---|---|
| 1 | 11,5 | 77,3 |
| 2 | 3,2 | 68,7 |
| 3 | 8,2 | 44,6 |
| 4 | 19,3 | 32,8 |
| 5 | 57,8 | 38,1 |

- **Clientes insatisfeitos escrevem cerca de duas vezes mais:** as notas 1–2 são ~15% das avaliações,
  mas ~26% dos textos. A base textual super-representa insatisfação.
- **Decisão:** proporções e contagens são calculadas com pandas sobre todas as avaliações, nunca
  estimadas pelo LLM a partir dos textos recuperados. O viés é declarado como limitação.

### Tamanho dos textos
- 42.370 avaliações únicas com texto. Mediana de 9 palavras, média 11,7, p90 28, p99 38, máximo 48
  (o formulário aparentemente limitava o tamanho).
- **Decisão: 1 avaliação = 1 documento, sem chunking.** Não há texto longo o bastante para dividir,
  e dividir quebraria o sentido de opiniões já curtas.
- Textos curtos permitem enviar mais evidências por pergunta dentro do limite de tokens do provedor.

### Textos curtos e ruído
- 6.232 textos (15%) com até 2 palavras; o topo é de elogios genéricos ("muito bom" ×501,
  "bom" ×388, "recomendo" ×385).
- **Não filtramos por tamanho:** textos curtos podem ser muito informativos ("entrega rápida",
  "não recebi o produto").
- **Decisões:**
  - descartar só textos sem nenhuma letra (77 viraram vazios após remover pontuação);
  - metadado `texto_curto` (≤ 3 palavras) para análise e filtro;
  - textos idênticos são colapsados nos resultados da busca ("aparece N vezes"), evitando
    evidências repetidas;
  - acentos removidos **só** no índice BM25 ("ótimo" e "otimo" viram a mesma palavra). O texto
    enviado ao modelo de embedding fica intacto.

    ### Empresas anonimizadas (casas de Game of Thrones)
- 1.951 textos (4,6%) citam alguma casa: Lannister 1.177, Stark 292, Baratheon 269, Targaryen 213.
- Pelo contexto, os nomes substituem lojas e parceiros ("lojas lannister", "targaryen parceira da
  lannister", atendimento do "baratheon").
- **Decisões:** manter os nomes no texto (carregam informação); avisar o LLM no prompt de que são
  empresas anonimizadas; metadado `empresas_citadas`; não tentar identificar as empresas reais,
  respeitando a anonimização feita pela Olist.

### Leitura qualitativa
- **A nota nem sempre reflete o texto:** há reclamações graves com nota 3 ("não recebi meu tapete,
  mas está marcando como entregue") e nota 4 (produto violado).
- **Decisão:** a nota é tratada como indicador aproximado de sentimento. O filtro por nota só é
  aplicado quando a pergunta o pede explicitamente; perguntas gerais sobre problemas consideram
  todas as notas.
- **Ruído de escrita** (erros de digitação, caixa alta, abreviações): sem correção ortográfica
  automática, que poderia alterar o sentido. Modelos de embedding multilíngues toleram esse ruído.
- **Quebras de linha (`\r\n`)** dentro dos textos são substituídas por espaço na limpeza.
- **Temas recorrentes já visíveis na amostra:** produto diferente do anunciado, produto trocado,
  itens faltando, "não recebido mas consta entregue", produto avariado, atraso, qualidade.


  ### Entrega
- 96.478 pedidos entregues (97%); 2.965 sem data de entrega ao cliente (em trânsito, cancelados,
  indisponíveis).
- **Situação da entrega por avaliação** (pior caso entre os pedidos ligados):

| Situação | % | Nota média | % nota 1 |
|---|---|---|---|
| No prazo | 90,6 | 4,29 | 6,6 |
| Atrasado | 6,5 | 2,27 | 53,7 |
| Não entregue | 2,9 | 1,76 | 70,4 |

- **O atraso multiplica por ~8 a chance de nota 1**, e cerca de metade das notas 1 vem de pedidos
  atrasados ou não entregues (<10% dos pedidos). Justifica os metadados `situacao_entrega`,
  `dias_atraso` e `status_pedido`.

### Categorias
- 97% das avaliações tocam uma única categoria; 963 (1%) tocam 2–3; 2.069 (2,1%) nenhuma.
- **Decisões:** `categoria_principal` (item mais caro) + lista completa; ausentes viram
  `"sem categoria"`; nomes em português, sem sublinhados (as perguntas serão em português, e a
  tabela de tradução não é necessária).
- Categorias com mais avaliações escritas: cama mesa banho (4.425), beleza saúde (3.567),
  esporte lazer (3.063), informática acessórios (2.813), móveis decoração (2.672).

### Período
- Avaliações de 02/10/2016 a 31/08/2018. Out/2016–jan/2017 quase vazios (dez/2016: 45 avaliações).
  Pico em dez/2017 (efeito Black Friday).
- **Decisões:** filtro de período pela data da compra (como no enunciado); datas guardadas como
  números (AAAAMMDD) para permitir filtros por intervalo no banco vetorial; períodos com pouca
  evidência são sinalizados na resposta.

### Estados
- SP concentra 42% das avaliações (nota 4,2; atraso 4,3%). RJ (3,9; 11,5%), BA (3,9; 11,6%) e
  ES (4,0; 10,2%) combinam mais atraso com nota menor; o RJ é o achado não óbvio.
- **Decisões:** metadados `uf` e `regiao` (a região agrega estados com poucas avaliações).



### Ficha do documento (base de conhecimento)
Cada avaliação com texto válido vira **um documento**:

| Campo | Conteúdo |
|---|---|
| `review_id` | identificador rastreável, usado nas citações |
| `texto` | título + mensagem, com limpeza leve |
| `nota` | 1 a 5 |
| `situacao_entrega`, `dias_atraso`, `status_pedido` | entrega (pior caso entre os pedidos ligados) |
| `categoria_principal`, `categorias` | categoria do item mais caro + lista completa |
| `data_compra`, `data_avaliacao` | inteiros AAAAMMDD (permitem filtro por intervalo) |
| `uf`, `regiao` | estado do cliente e região |
| `empresas_citadas` | casas anonimizadas citadas no texto |
| `n_palavras`, `texto_curto` | tamanho do texto |
| `chave_texto` | texto padronizado para colapsar textos idênticos |
| `pedidos` | pedidos de origem (rastreabilidade) |

- **Só o texto vai para o embedding; os metadados entram como filtros.** Embutir metadados no texto
  aproximaria avaliações da mesma categoria mesmo quando falam de coisas diferentes.
- **Fora de escopo, de propósito:** vendedor, preço e tempo total de entrega (não exigidos; fáceis
  de acrescentar).
- **Limpeza leve:** sem remoção de stopwords, sem stemming e sem corretor ortográfico, porque
  modelos de embedding usam a frase inteira e palavras como "não" mudam o sentido.
- **Dois arquivos** (`src/ingestao/construir_corpus.py`):
  - `avaliacoes.parquet`: 98.410 avaliações únicas, com e sem texto, para estatísticas;
  - `corpus.parquet`: 42.251 documentos com texto válido, para a busca.

### Resultado da construção
- 119 textos descartados por não terem letra nem número.
- 34.956 textos distintos em 42.251 documentos (~17% repetidos): os embeddings serão calculados só
  para os textos distintos.
- 8.324 textos curtos (≤ 3 palavras); 1.951 citam empresa anonimizada; 1.080 sem categoria.
- Entrega no corpus: 86,5% no prazo, 9,1% atrasado, 4,5% não entregue (contra 90,6/6,5/2,9 na base
  completa), o que confirma o viés de escrita.


  ### Validação da ficha
- A avaliação do cartucho HP (ligada a 2 pedidos) virou **um único documento**, com os 2 pedidos no
  campo `pedidos`.
- **Percepção vs registro:** o cliente relata "atraso de mais de 15 dias", e o sistema registra 6
  dias (pior caso entre os pedidos). O sistema de respostas deve distinguir o que o cliente relata
  (citado como relato) do que o registro mostra (apresentado como dado), sem "corrigir" o cliente.
- `dias_atraso` negativo indica entrega antes do prazo prometido; a frequência de valores negativos
  sugere prazos estimados folgados.