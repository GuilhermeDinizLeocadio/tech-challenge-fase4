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


## Etapa 2 — Embeddings e índices (19/10/2026)

### Modelo de embedding
| Opção | Português (MTEB-por) | Custo | Reprodução |
|---|---|---|---|
| **multilingual-e5-large-instruct** | **0,641** | grátis, local (CPU) | sem chave |
| bge-m3 | 0,616 | grátis, local | sem chave |
| multilingual-e5-base | menor (mesma família) | grátis, ~3x mais rápido | sem chave |
| gemini-embedding-001 | melhor do ranking nativo | API grátis com limite diário | exige chave |

- **Escolhido: multilingual-e5-large-instruct**, o melhor modelo aberto em português testado, sem
  chave e sem custo. Critério definido antes da medição: manter se a indexação levasse até ~40 min
  na CPU; medido 16 textos/s (estimativa de 36 min; real de 33 min).
- **Pesos convertidos de float16 para float32:** o modelo é distribuído em meia precisão; na CPU,
  float32 é o formato nativo e garante precisão para comparar similaridades muito próximas.
- **Assimetria pergunta/documento:** perguntas recebem a instrução do e5-instruct; documentos entram
  puros. As regras ficam num único módulo (`src/indexacao/embeddings.py`), usado na indexação e na
  busca, para que perguntas e documentos estejam sempre no mesmo espaço vetorial.

### Comportamento observado
- **O embedding capta o assunto, não a polaridade:** para "o que reclamam do atraso?", o elogio
  "chegou antes do prazo" (0,820) ficou praticamente empatado com uma reclamação real (0,826).
  Isso justifica o reranker.
- **Faixa de similaridade espremida:** até um texto irrelevante obteve 0,755. Um limiar de recusa
  fixo não funciona; ele será calibrado com dados.

### Banco vetorial e BM25
- **Chroma:** filtro por metadados nativo, instalação com `pip`, persistência em pasta. Com ~35 mil
  vetores, velocidade não é critério; simplicidade é.
- **Distância l2 sobre vetores normalizados:** equivale à ordem por similaridade de cosseno
  (similaridade = 1 − distância/2) e não depende de configurações que mudam entre versões.
- **Metadados ausentes não são enviados ao Chroma** (que não aceita campos vazios); por exemplo,
  `dias_atraso` de pedidos não entregues.
- **BM25 (`bm25s`) sobre a `chave_texto`** (minúsculas, sem acento, sem pontuação), **sem remover
  stopwords**: o BM25 já reduz o peso de palavras comuns, e listas de stopwords em português incluem
  "não", que muda o sentido.

### Construção (`src/indexacao/construir_indice.py`)
- Embeddings só dos **34.956 textos distintos**; textos idênticos compartilham o mesmo vetor.
- **Progresso salvo em 18 partes:** se interrompido, o script continua de onde parou.
- Tempos num i5-10400 (CPU): embeddings 1.966 s, Chroma 38 s, BM25 1 s; total ~33 min.
- `indice/manifesto.json` registra modelo, dimensão, precisão, quantidades e tempos.
- O índice não é versionado no Git: é regerado pelo script.

### Teste
- "O produto chegou quebrado ou danificado?" → os 5 primeiros resultados tratam de produto quebrado
  (notas 1–2; similaridade 0,922–0,930), incluindo textos com erro de digitação.
- Dois dos 5 resultados eram o mesmo texto com pontuação diferente, o que confirma a necessidade de
  colapsar textos idênticos na etapa de recuperação.

### Teste dos índices (notebook 02, célula 4)
- **Filtro de metadados:** "Quais problemas os clientes relatam sobre o produto?" com
  `categoria_principal = cama mesa banho` e `nota <= 2` → 5 resultados da categoria, todos nota 1
  (produto diferente do comprado, qualidade, cores erradas, não entrega).
- **BM25 ("correios"):** todos os resultados contêm o termo, mas o 1º lugar é o texto "Correios",
  de uma palavra. O BM25 favorece textos curtos, o que justifica combiná-lo com a busca por sentido
  (híbrida) e com um reranker.
- **Nota vs texto, de novo:** "Os correios atrasaram." (nota 5), "Problemas com os Correios"
  (nota 5), "Correios não entregou" (nota 4). Parte dos clientes atribui o problema à transportadora
  e não pune a loja na nota, o que reforça que problemas devem ser buscados pelo texto, não só pela
  nota.

## Etapa 3 — RAG básico (19/10/2026)

### Recuperação (`src/rag/recuperacao.py`)
- `Buscador` carrega modelo e índices uma vez (~11 s) e responde a várias buscas.
- **Linguagem única de filtros** (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`), traduzida para o
  Chroma e para o pandas, garantindo que as duas buscas avaliem o mesmo recorte.
- **Fusão RRF** (k = 60, Cormack et al., 2009): combina posições em vez de notas, porque as notas da
  busca densa (0–1) e do BM25 têm escalas diferentes.
- **Colapso de textos idênticos** pela `chave_texto`, com a contagem de repetições no recorte filtrado.

### A busca híbrida ingênua piorou o resultado
- "Os produtos chegam com defeito ou quebrados?": a busca densa trouxe 8/8 resultados relevantes;
  a híbrida, 5/8. O BM25 casou palavras comuns ("produtos", "chegam") e não casou "quebrados" com
  "quebrado", porque não trata variações das palavras do português.
- **Decisão:** a Etapa 3 usa a **busca densa como linha de base**. A híbrida será corrigida na
  Etapa 5 (redução das palavras ao radical e pesos diferentes no RRF) e só será adotada se superar a
  densa no conjunto de teste.

### O problema das perguntas agregadas, observado
- "O que os clientes mais elogiam?" (nota ≥ 4; 27.791 avaliações no recorte) trouxe textos que falam
  de "clientes" e de "elogio", e não os elogios mais frequentes. A busca encontra textos parecidos com
  a pergunta, mas não mede frequência.
- Isso confirma a necessidade da camada de temas pré-computados com contagens reais (Etapa 6).

### Linha de base (busca densa) e colapso
- "O produto chegou quebrado" → 10/10 resultados relevantes (similaridade 0,928–0,946), incluindo
  paráfrases ("veio quebrado", "uns parte quebrada", "entregue danifica").
- O texto "O produto chegou quebrado" (2 cópias na base) ocupou **uma** posição, com repetições = 2.
- **Limitação:** o colapso pega textos idênticos após padronização, não quase idênticos
  ("Produto veio quebrado" e "o produto veio quebrado" ficaram separados). Possível melhoria
  futura: agrupar por similaridade muito alta.

### Geração (`src/rag/geracao.py`) e comparação de redatores
- Contexto numerado [R#]: cabeçalho com metadados (registro do sistema) + texto entre aspas
  (relato do cliente). Temperatura 0; espera e nova tentativa ao atingir o limite gratuito;
  `reasoning_effort = low` no gpt-oss.
- **Primeira resposta citada em 19/10/2026** (marco previsto para 25/10): 1,4 s, 1.890 tokens de
  entrada, 0 citações inválidas.
- **Comparação v1 (3 perguntas × 2 modelos):**

| Critério | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| Recusa fora do escopo | limpa | correta, seguida de conteúdo irrelevante |
| Causa | afirmada pelo modelo ("devido a") | atribuída ao cliente ("é citada como") |
| Trechos literais | raros | frequentes |
| Interpretação sem fatos novos | melhor | pior |
| Tokens de saída | ~290–325 | ~580–614 |

- **Decisão:** gpt-oss-120b como redator provisório (recusa, custo, velocidade). A escolha final
  será confirmada pela fidelidade medida pelo juiz na Etapa 4.
- **Temperatura 0 não garante determinismo:** a mesma pergunta gerou respostas diferentes (17 e
  13 citações). A avaliação será feita sobre respostas salvas em disco.

### Prompt v2
Ajustes motivados pelos erros observados na comparação:
1. proibição de termos de frequência ("a maioria", "principalmente", "recorrente"...);
2. uma ideia por ponto, citando só as avaliações que a mencionam;
3. trechos literais entre aspas como prova;
4. causas apenas quando relatadas e atribuídas ao cliente;
5. relato do cliente separado do registro do sistema;
6. recorte (filtros e tamanho) informado ao modelo e declarado na resposta;
7. interpretação como hipótese, sem fatos novos;
8. formato curto e exclusivo para recusas.

Verificações automáticas no código: citações inválidas, trechos entre aspas inexistentes, termos de
frequência e recusa.

- **Limite de tokens por minuto:** na comparação (3 perguntas por modelo em ~1 minuto, ~2.500–2.800
  tokens por chamada), o limite de 8 mil tokens/min do Groq foi atingido 2 vezes; a espera
  automática de 20 s resolveu na 2ª tentativa. Na avaliação, as chamadas terão ritmo controlado
  (≤ 2 por minuto por modelo, alternando modelos).


  ### Resultados do prompt v2 (comparação nas mesmas 3 perguntas)
- **Recusas perfeitas nos dois modelos:** frase padrão, motivo, sugestão, 0 citações (~90 tokens).
  O Qwen deixou de preencher seções com conteúdo irrelevante.
- **Recorte declarado** quando há filtros ("recorte de categoria cama mesa banho com nota ≤ 2").
- **Registro rotulado como registro** ("segundo o registro, o pedido atrasou 42 dias"), separado
  do relato do cliente entre aspas.
- **Citações mais precisas:** 9 em vez de 13–17 na pergunta específica.
- **Persistiram:** termos de frequência ("frequentemente", "recorrentes") e pontos que juntam mais
  de uma ideia. Uma regra no prompt reduz, mas não elimina, esses hábitos do modelo.
- **Custo e velocidade empataram** (~540–650 tokens de saída e 1,6–1,8 s nos dois modelos), porque
  os trechos literais alongaram as respostas do gpt-oss.

### Verificação automática de trechos literais
- Cada trecho entre aspas é conferido no texto das avaliações citadas (ou em todo o contexto, se não
  houver citação), com texto padronizado.
- **Calibração com casos reais:** dos 4 trechos sinalizados na primeira versão, 2 eram erros reais
  (1 trecho inventado, 1 paráfrase entre aspas) e 2 eram falsos alarmes (o modelo corrigiu
  "qualilidade" e completou "condico", palavra cortada no texto do cliente).
- **Tolerância por semelhança** (`difflib`, limiar 0,85): aceita pequenas diferenças de grafia e
  rejeita paráfrases. Após o ajuste, só os 2 erros reais continuaram sinalizados. O limiar foi
  escolhido em 4 casos e será reavaliado na Etapa 4.
- **Caso de fronteira:** aspas de destaque numa palavra ("recebido") são sinalizadas embora não
  enganem o leitor. Decisão adiada para a Etapa 4, com mais casos (possível regra: conferir só
  trechos com 2 ou mais palavras).
- **Achado:** textos cortados no meio da palavra ("boas condico") confirmam que o formulário da
  Olist limitava o número de caracteres.

### Mudança de redator provisório
| Erro de fidelidade (prompt v2, 3 perguntas) | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| Trecho inventado | 1 | 0 |
| Paráfrase entre aspas | 1 | 0 |
| Sentido invertido | 1 | 0 |

- Com o prompt v2, recusa, custo e velocidade empataram; na fidelidade, critério principal do
  enunciado, o Qwen teve 0 erros contra 3. **Redator provisório trocado para qwen3.8-27b**, com
  confirmação pelo juiz na Etapa 4.

### Interface de terminal (`src/rag/perguntar.py`)
- `python -m src.rag.perguntar "pergunta" [--categoria] [--uf] [--regiao] [--entrega]
  [--nota-min] [--nota-max] [--de AAAAMMDD] [--ate AAAAMMDD] [--k] [--modelo] [--modo]`.
- Mostra a resposta, as **evidências citadas** (com `review_id`, nota, categoria, data, UF e texto
  original) e as **verificações automáticas**.
- **Testes (qwen3.8-27b):**
  - "atrasos na entrega" (2,1 s): resposta útil, mas o verificador pegou dados do registro
    ("atrasado (15 dias após o prazo)") colocados entre aspas como se fossem fala do cliente, além
    de termos de frequência.
  - "problemas" em cama mesa banho com nota ≤ 2: recorte declarado, contagens que batem com as
    citações ("5 das avaliações"), registro e relato separados na mesma frase.
  - "melhor celular para comprar em 2026": recusa correta, justificada pelas datas da base
    (2017–2018).
- **Próximo passo (Etapa 5):** os problemas que persistem são justamente os que o código detecta;
  uma reescrita automática única, acionada pelas verificações, pode corrigi-los.

### Correções de processo
- **Valores duplicados no `.env`** (`LLM_MODELO=LLM_MODELO=...`): erro latente que nunca disparou
  porque o modelo era passado explicitamente no código. Corrigido; o script de avaliação vai
  conferir os nomes dos modelos antes de começar.
- **Célula 4 do notebook 02 não salva antes do commit da Etapa 2:** restaurada com resultado
  idêntico (a busca é determinística). Regra adotada: salvar todos os arquivos antes de `git add`.


  ## Etapa 4 — Conjunto de teste e métricas (26/10/2026)

### Conjunto de perguntas (`data/avaliacao/perguntas.json`)
31 perguntas com gabarito de comportamento:

| Tipo | Qtd. | Esperado | Objetivo |
|---|---|---|---|
| enunciado | 8 | responder | as perguntas exatas do enunciado, incluindo o desafio adicional |
| especifica | 10 | responder | temas concretos, sem filtro |
| filtro | 5 | responder | categoria, UF, região, período e situação da entrega |
| fora_escopo | 4 | recusar | perguntas sem relação com avaliações |
| sem_evidencia | 3 | recusar | temas plausíveis de e-commerce, inexistentes na base (criptomoedas, drones, realidade aumentada) |
| evidencia_escassa | 1 | ressalvar | recorte com apenas 5 avaliações: responder citando-as e declarar o recorte pequeno |

- Perguntas que pedem frequência ("principais", "mais elogiados") são marcadas como `agregada`:
  são medidas já agora para servir de linha de base para a camada de temas (Etapa 6).
- **Validação** (`src/avaliacao/validar_perguntas.py`): filtros e categorias existentes, tamanho de
  cada recorte e conferência humana das perguntas que devem ser recusadas.

### A similaridade sozinha não separa o que deve ser recusado
| Esperado | sim_top1 mínima | média | máxima |
|---|---|---|---|
| recusar | 0,857 | 0,873 | 0,891 |
| responder | 0,880 | 0,902 | 0,934 |

- As faixas se sobrepõem: "drones" (deve recusar) obteve 0,891, acima de perguntas legítimas com
  filtro (0,880). Um limiar de similaridade separaria no máximo 6 das 8 recusas, com margem de
  0,001. Perguntas com filtro têm similaridade menor (recorte menor), e um limiar único as
  puniria.
- **Decisão:** o limiar de recusa será calibrado sobre a nota do reranker (Etapa 5).
- **Exemplo para o vídeo:** "previsão do tempo para amanhã" casou com avaliações sobre "previsão de
  entrega": o embedding captou a palavra, não o sentido.

### Revisão do gabarito
- A pergunta sobre entregas na região Norte em 2016 foi criada como "sem evidência", mas a
  conferência humana mostrou 2 relatos reais entre as 5 avaliações do recorte. Ela passou a
  `evidencia_escassa`, com comportamento esperado "ressalvar": recusar ignoraria evidência real;
  responder sem ressalva generalizaria a partir de 5 avaliações.


  ### Gabarito da busca: rotulagem de relevância pelo juiz
- **Pooling:** para cada pergunta, os candidatos das buscas densa e híbrida (até 50 de cada; ~75
  distintos) foram rotulados pelo juiz (`gemini-3.5-flash-lite`): 2 = responde diretamente,
  1 = mesmo assunto mas não responde, 0 = sem relação. Instrução: "na dúvida entre 2 e 1, escolha 1".
- 2.222 rótulos em 91 chamadas, sem erros (lotes de 25; 1 chamada a cada 4,5 s; cada rótulo salvo
  ao chegar em `resultados/relevancia.jsonl`, versionado para reprodução sem custo).
- **Coerência:** perguntas a recusar quase sem nota 2 (0 nas de faturamento, tempo, criptomoedas e
  realidade aumentada). Em "drones", 64 avaliações nota 1 (falam de entrega) e nenhuma nota 2: a
  escala de 3 níveis separou "mesmo assunto" de "responde".
- **Limitação:** o recall é relativo aos candidatos rotulados, não à base inteira.

### Métricas da busca (24 perguntas a responder/ressalvar, K = 20)
| Métrica | Densa | Híbrida ingênua |
|---|---|---|
| Precisão@20 | **0,821** | 0,702 |
| Hit@5 | 0,958 | **1,000** |
| MRR | **0,965** | 0,958 |
| Recall@20 (relativo) | **0,357** | 0,310 |

- A híbrida ingênua envia 12 pontos a mais de contexto irrelevante ao LLM, o que confirma com
  números a decisão da Etapa 3. Mas ela acerta uma pergunta em que a densa falha: a correção da
  híbrida (Etapa 5) se justifica.
- Precisão@20 da densa por tipo: filtro 1,000; enunciado 0,881; específica 0,725; evidência
  escassa 0,400 (o máximo possível: 2 relevantes num recorte de 5).
- Perguntas a recusar recebem em média 6,6 "iscas" (nota 1) entre as 20 avaliações enviadas.

### Falhas da busca densa e técnicas a testar na Etapa 5
| Falha | Pergunta | Precisão@20 | Técnica |
|---|---|---|---|
| Pergunta "meta" ("quais avaliações sustentam a conclusão de que...") | P05 | 0,45 (hit@5 = 0) | reescrita da pergunta |
| Condição específica ("atendimento ao tentar resolver um problema") | P13, P11 | 0,25 / 0,35 | reranker |
| Termo exato ("Correios") | P15 | 0,50 | híbrida corrigida |

### Validação do juiz de relevância
- 21 pares estratificados (7 por nota do juiz, incluindo os 2 casos suspeitos), notas humanas dadas
  às cegas.

| Medida | 1ª rodada (às cegas) | Após consenso |
|---|---|---|
| Concordância exata | 52% | 86% |
| Concordância relevante/não | 57% | 90% |
| Kappa de Cohen | 0,29 | 0,79 |

- **1ª rodada:** nas 10 discordâncias, o humano tendeu a dar nota 2 a avaliações positivas ou
  genéricas quando a pergunta pedia algo específico (drones, realidade aumentada, Lannister), o que
  revelou uma aplicação ambígua da régua.
- **Consenso** (adjudicação das discordâncias): o juiz estava correto em 7, o humano em 2 (um caso
  de juiz exigente demais e um de juiz generoso demais) e 1 caso era ambíguo. Os erros residuais do
  juiz não têm viés para um lado.
- **Ressalvas:** amostra de 21 pares; consenso feito com revisão assistida e apenas nos pares
  discordantes, o que tende a otimizar o número final. Por isso os dois resultados são reportados.

  ### Geração das respostas da linha de base (`src/avaliacao/gerar_respostas.py`)
- 31 perguntas × 2 redatores = 62 respostas, salvas com as evidências usadas em
  `resultados/respostas/densa_k20_prompt_v2.jsonl`. Sem erros e sem atingir o limite de uso.
- **Comparação justa:** uma busca por pergunta; as mesmas 20 evidências vão para os dois modelos.
- **Ritmo controlado:** chamadas alternadas entre os modelos, com 30 s entre duas chamadas ao
  mesmo modelo.

| Medida | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| Comportamento correto (recusar/responder) | 97% (30/31) | **100%** (31/31) |
| Sem evidência (drones, cripto, realidade aumentada) | 100% | 100% |
| Citações inválidas | 0 | 0 |
| Trechos não encontrados | 31 | **5** |
| Respostas com termos de frequência | **10** | 13 |
| Tokens de saída (média) / tempo (mediana) | 370 / 1,5 s | 394 / 1,5 s |

- **Falha de comportamento:** o gpt-oss reinterpretou "Como configurar um roteador?" (pedido de
  instrução) como "o que os clientes dizem sobre configurar Wi-Fi" e respondeu, em vez de recusar.
- **Análise dos trechos sinalizados:**
  - Qwen: 1 erro real (dado do registro entre aspas) e 4 falsos alarmes (aspas de destaque,
    inserção editorial entre colchetes, termos da própria pergunta citados numa recusa).
  - gpt-oss (amostra de 10): 6 eram dados do **registro** do sistema ("60 dias após o prazo",
    "pedido não entregue") apresentados entre aspas como fala do cliente; 4 eram costuras ou
    paráfrases com reticências.
- **Decisões para a Etapa 5:** refinar o verificador (ignorar trechos presentes na pergunta, remover
  inserções entre colchetes, classificar "registro citado como relato" à parte) e separar
  visualmente registro e relato no contexto. A linha de base será recalculada com o verificador
  refinado sobre as respostas salvas, para que antes e depois usem a mesma régua.
- A regra "conferir só trechos com 2 ou mais palavras" **não foi adotada**: apenas 1 de 36 trechos
  sinalizados tinha uma palavra.


  ### Fidelidade: o juiz confere cada afirmação (`src/avaliacao/avaliar_fidelidade.py`)
- Para cada resposta não recusada (24 do Qwen, 25 do gpt-oss), o juiz recebe o mesmo contexto do
  redator e dá um veredito a cada afirmação: sustentada, parcial ou não sustentada. Também avalia
  fatos novos na interpretação, se a resposta responde à pergunta (1 a 5) e se declara limitação de
  evidência. 49 auditorias, 333 afirmações, sem erros.
- **Suspeita de leniência:** fidelidade de ~95% nos dois modelos, nota ~5 até nas perguntas
  agregadas, e respostas aprovadas por inteiro apesar de conterem dados do registro entre aspas,
  que o verificador por código tinha sinalizado.

### Validação do juiz de fidelidade
- 12 afirmações (amostra enviesada para casos de risco: 6 "sustentadas" de respostas com trechos
  sinalizados, 3 de respostas limpas, 3 parciais/não sustentadas), vereditos humanos às cegas e
  depois consenso com os textos originais lado a lado.

| Medida | Valor |
|---|---|
| Juiz x consenso | 75% (9 de 12) |
| Leniência (juiz "sustentada", consenso não) | 2 de 9 (22%) |
| ...pegas pelo verificador por código | 1 de 2 |
| Juiz exigente demais | 1 |

- Os 2 casos lenientes eram do gpt-oss: um trecho com parte inventada (pego pelo verificador) e
  uma **inversão de sentido** ("outros não obtêm reposição", citando um cliente que escreveu "não tô
  solicitando outro produto nem a devolução"), a mesma avaliação e o mesmo erro vistos na Etapa 3.
- **Nenhum mecanismo isolado é suficiente:** a primeira leitura humana também aprovou os dois casos;
  só a releitura com os textos lado a lado os revelou.

### Fidelidade ajustada e tabela da linha de base (`src/avaliacao/relatorio.py`)
- **Fidelidade ajustada:** uma afirmação "sustentada" pelo juiz que contém um trecho não encontrado
  pelo verificador é rebaixada para "parcial". Cada mecanismo cobre a fraqueza do outro.

| Medida (densa, K = 20, prompt v2) | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| Comportamento correto | 96,8% | **100%** |
| Sem evidência recusadas | 100% | 100% |
| Trechos não encontrados | 31 | **5** |
| Respostas com termos de frequência | **10** | 13 |
| Fidelidade (juiz) | 95,6% | 95,3% |
| **Fidelidade ajustada** | 89,6% | **94,0%** |
| Não sustentadas | **1,1%** | 2,0% |
| Interpretação com fatos novos | **0** | 2 |
| Tokens de saída / tempo | 370 / 1,5 s | 394 / 1,5 s |

- **Decisão: qwen3.8-27b confirmado como redator.** Vence nos critérios centrais do enunciado
  (comportamento diante de falta de evidência e fidelidade), com custo e tempo equivalentes.
- **Limitações declaradas:** o juiz de fidelidade é leniente com erros sutis (seus números são um
  teto); inversões de sentido não são detectadas por nenhum mecanismo automático do projeto;
  validações com amostras pequenas (21 e 12 casos).
- A tabela é gerada a partir de todas as configurações salvas: cada melhoria da Etapa 5 entra como
  uma nova coluna, medida com a mesma régua.


  ## Etapa 5 — Melhorias medidas

### Estratégia
Melhorias na busca são medidas offline, com os rótulos de relevância da Etapa 4 (sem custo de API);
só a melhor configuração passa pela geração de respostas e pelo juiz de fidelidade.

### Reranker (`BAAI/bge-reranker-v2-m3`, cross-encoder local)
- Lê a pergunta junto com cada um dos 50 primeiros candidatos (já sem textos repetidos) e reordena
  pela nota de relevância. Atuar só nos 50 primeiros garante que todo candidato já tem rótulo.
- Escolha: multilíngue e forte em português, local e sem chave, compatível com o
  `sentence-transformers`. Critério definido antes de medir: até 10 s por pergunta.

| Configuração | Precisão@20 | Hit@5 | MRR | Recall@20 | Tempo (mediana) |
|---|---|---|---|---|---|
| densa | 0,821 | 0,958 | 0,965 | 0,357 | 0,16 s |
| híbrida ingênua | 0,702 | 1,000 | 0,958 | 0,310 | 0,14 s |
| **densa + reranker** | **0,896** | **1,000** | **0,979** | **0,402** | 5,03 s |
| híbrida + reranker | 0,865 | 1,000 | 0,958 | 0,392 | 4,47 s |

- Maior ganho nas perguntas específicas (precisão 0,725 → 0,870): o embedding capta o assunto, o
  reranker capta o detalhe da pergunta. P09 (quebrados) e P11 (itens faltando) foram a 1,00; a
  P05 ("meta") passou a ter avaliação relevante entre as 5 primeiras.
- **Não resolvida:** P13 ("atendimento ao tentar resolver um problema", 0,25 → 0,20). O reranker só
  reordena os candidatos da primeira busca; se eles não contêm avaliações relevantes, não há o que
  reordenar → reescrita da pergunta (parte 2).
- **P15 ("Correios"):** 0,70 com densa + reranker, 0,90 com híbrida + reranker → termos exatos se
  beneficiam do BM25 → híbrida corrigida (parte 2).

### Recusa calibrada pelo reranker: descartada com base em dados
- A nota do reranker na 1ª avaliação ficou entre 0,000 e 0,009 nas perguntas a recusar, mas
  perguntas legítimas agregadas ou genéricas tiveram notas iguais ou menores (P06 "principais
  reclamações" 0,002, a mesma de "entrega por drones"; P22 0,004; P01 0,005).
- Motivo: o reranker mede se UMA avaliação responde à pergunta; perguntas sobre o conjunto não são
  respondidas por nenhuma avaliação isolada.
- Decisão: sem porteiro por limiar. O redator já recusa corretamente 100% das perguntas sem
  evidência; um limiar recusaria perguntas do próprio enunciado. Além disso, com 31 perguntas,
  qualquer limiar seria ajustado na própria prova.


  ### Híbrida corrigida e reescrita da pergunta (parte 2)
- **Híbrida corrigida:** BM25 com radicais (PyStemmer, português: "quebrado", "quebrados" e
  "quebrou" → "quebr") e peso 0,5 na fusão (definido antes de medir).
- **Reescrita da pergunta** (`src/rag/reescrita.py`): o LLM escreve 3 frases como clientes
  escreveriam; a 1ª busca usa pergunta + frases, o reranker julga pela pergunta original.
  Reescritas salvas em `resultados/reescritas.json` (avaliação reprodutível).
  - **Prompt v1** (guardado em `reescritas_v1.json`), inspecionado antes de medir: invertia a
    polaridade ("atraso" ganhava "chegou antes do previsto"), perdia termos da pergunta ("Correios",
    "Lannister") e inventava clientes para perguntas fora do assunto ("previsão do tempo").
  - **Prompt v2** (revisão única, por princípios gerais): manter a polaridade e os termos da
    pergunta; `SEM_REESCRITA` quando a pergunta não trata de experiências de clientes (aplicado a
    faturamento, celular, roteador e previsão do tempo).
- Rotulagem incremental: só os pares novos foram ao juiz (240 rótulos, 2.462 no total).

| Configuração (24 perguntas a responder) | Precisão@20 | Hit@5 | MRR | Recall@20* |
|---|---|---|---|---|
| densa | 0,821 | 0,958 | 0,965 | 0,305 |
| híbrida ingênua | 0,702 | 1,000 | 0,958 | 0,266 |
| híbrida corrigida | 0,827 | 0,958 | 0,917 | 0,308 |
| densa + reranker | 0,896 | 1,000 | 0,979 | 0,339 |
| híbrida + reranker | 0,865 | 1,000 | 0,958 | 0,329 |
| híbrida corrigida + reranker | 0,896 | 1,000 | 0,979 | 0,339 |
| **densa + reescrita + reranker** | **0,935** | **1,000** | **0,979** | **0,368** |

\* Recall relativo ao pool de rotulagem, que cresceu nesta etapa: só comparável dentro da mesma
rodada.

- **Perguntas que falhavam** (densa → + reranker → + reescrita): P05 0,45 → 0,60 → 0,90;
  P13 0,25 → 0,20 → 0,55; P14 0,60 → 0,80 → 1,00; P15 0,50 → 0,70 → 1,00.
  A reescrita resolveu o que o reranker não alcançava: melhorou a 1ª busca (P13) e, mantendo o
  termo "Correios", tornou a busca por palavras desnecessária (P15). P13 segue a mais difícil.
- **Híbrida descartada, com explicação matemática:** com peso 0,5 e RRF k = 60, a 1ª colocada só
  do BM25 vale 0,5/61 ≈ 0,0082, menos que a 50ª da densa (1/110 ≈ 0,0091). O BM25 nunca traz
  candidata nova ao top-50, só reordena; com o reranker, o resultado é idêntico ao da densa.
  Sem reranker, a correção funcionou (0,702 → 0,827), mas não supera a densa. Outros pesos não
  foram testados para não ajustar o sistema à prova; o motivo da híbrida (P15) já foi resolvido.
- **Custo da reescrita nas agregadas:** precisão 0,98 → 0,93 nas agregadas e 0,874 → 0,937 nas
  demais. É o viés de confirmação previsto (a reescrita busca o que o LLM supõe); a decisão sobre
  as agregadas fica para a Etapa 6 (camada de temas).
- Recusas sem piora (iscas 5,9 → 6,0). Tempo: ~5,3 s por busca + uma chamada ao LLM para reescrever.
- **Decisão: busca final = densa + reescrita + reranker.**


### Verificador refinado (v2) e linha de base recalculada (parte 3a)
- `classificar_trechos` separa os trechos entre aspas que não estão no relato em três categorias:
  inexistentes, **registro citado como relato** (dados do cabeçalho entre aspas como fala do
  cliente) e **termos da própria pergunta** (ex.: numa recusa). Inserções editoriais
  [entre colchetes] são removidas antes de comparar. O verificador v1 foi mantido.
- O relatório recalcula v1 e v2 a partir das respostas salvas: todas as configurações passam a ser
  medidas com a mesma régua, sem nova chamada à API. O v1 recalculado bate com a Etapa 4 (31 e 5).

| Linha de base (densa, prompt v2) | gpt-oss-120b | qwen3.8-27b |
|---|---|---|
| Trechos sinalizados (v1) | 31 | 5 |
| Inexistentes (v2) | 19 | 1 |
| Registro citado como relato (v2) | 12 | 1 |
| Termos da pergunta (v2, falsos alarmes) | 0 | 2 |
| Fidelidade ajustada com v1 → v2 | 89,6% → 89,6% | 94,0% → 94,7% |

- Dos 5 sinalizados do Qwen, só 2 eram erros reais, de tipos diferentes.
- A categoria "registro citado como relato" é um limite inferior: só é reconhecida quando o trecho
  se parece com o texto do cabeçalho ("pedido não entregue" não casa com "entrega: não entregue").
- A fidelidade ajustada passa a usar o v2: falsos alarmes deixam de rebaixar afirmações corretas,
  e todo erro real continua contando.


  ### Fluxo completo no código (parte 3b)
- `src/rag/pipeline.py`: função única `responder` (reescrita → busca → reranker → resposta),
  usada pela linha de comando e, na Etapa 8, pela interface, para que todos os caminhos sigam o
  mesmo fluxo. Se a reescrita falhar (limite de uso, servidor fora do ar), a busca continua só
  com a pergunta original: a resposta pode piorar um pouco, mas o sistema não para.
- Linha de comando: o fluxo vencedor passou a ser o padrão, com `--sem-reescrita`,
  `--sem-reranker` e `--prompt` para reproduzir as configurações anteriores.
- Geração e auditoria passaram a receber as configurações no comando; o juiz recebe o contexto no
  mesmo formato (v2, v3 ou v4) que o redator recebeu.
- Teste na P13 (a pergunta mais difícil da prova): resposta equilibrada, separando problemas
  resolvidos rapidamente de demora, falta de retorno e telefones sem atendimento, com trechos
  literais e todas as verificações automáticas limpas.

### Isolando o efeito da busca e do prompt (parte 3c)
- Duas configurações novas, só com o Qwen e com as mesmas evidências por pergunta: busca nova +
  prompt v2 (isola o efeito da busca) e busca nova + prompt v3 (isola o efeito do prompt).

| Medida (Qwen) | Linha de base | Busca nova + v2 | Busca nova + v3 |
|---|---|---|---|
| Comportamento correto | 100% | 100% | 100% |
| Trechos não encontrados (v2) | 1 | 2 | 2 |
| Registro citado como relato | 1 | 0 | 0 |
| Respostas com termos de frequência | 13 | 12 | 10 |
| Respostas com contagem | 18 | 24 | 24 |
| Fidelidade ajustada | 94,7% | 91,7% | 86,3% |

- **Tempos não comparáveis:** com duas chamadas seguidas ao mesmo modelo e intervalo de 25 s, a
  reserva de tokens do Groq (entrada + máximo de saída) estourou o limite por minuto e a
  biblioteca esperou sozinha, inflando o cronômetro (medianas de 8,6 s e 5,3 s). O tempo real de
  geração segue em ~1,5 s. Intervalo aumentado para 35 s.

### Diagnóstico: a queda da fidelidade vinha das contagens
- A fidelidade caiu justamente com a busca melhor. A inspeção das justificativas do juiz
  (`src/avaliacao/inspecionar_fidelidade.py`) mostrou que 9 de 11 (v2) e 16 de 19 (v3) afirmações
  não sustentadas eram **contagens** que não batiam com as citações ("8 das avaliações
  recuperadas" citando 2; "18" citando 19; "9 das 20" sem citação). O juiz estava certo.
- **Causa:** a regra 4 do prompt (Etapa 3) sugeria "N das avaliações recuperadas" como alternativa
  a "a maioria". Com a busca melhor, quase todas as 20 evidências são relevantes, e o modelo passou
  a contar números grandes citando só exemplos.
- **Problema conceitual:** mesmo correta, a contagem é frequência disfarçada. A busca escolhe as
  avaliações pelo assunto; dizer que "18 das 20 mencionam entrega antes do prazo" mede a busca, não
  os clientes. Frequências reais exigem contar na base inteira (Etapa 6).
- Fora as contagens, restaram 2 (v2) e 3 (v3) erros de conteúdo em mais de 130 afirmações cada,
  todos de citação trocada ou fraca.

### Prompt v4 e detector de contagens
- **Prompt v4** = v3 com a regra 4 reescrita: proíbe contar avaliações ou clientes da amostra e
  usar porcentagens; a única quantidade permitida é o "aparece Nx", que conta textos idênticos no
  recorte inteiro.
- **Detector de contagens no código** (`encontrar_contagens`, fora da seção Limitações), seguindo a
  lição da Etapa 4 de não confiar só no prompt. Validado nas respostas salvas: as contagens já
  apareciam em 18 das 24 respostas não recusadas da linha de base e em todas as 24 com a busca nova.
- Meta do v4: respostas com contagem perto de zero e fidelidade de volta ao patamar da linha de
  base ou acima.

  ### Prompt v4: sem contagens (parte 3c, continuação)

| Medida (Qwen, busca nova) | Prompt v3 | Prompt v4 |
|---|---|---|
| Respostas com contagem | 20 | 1 |
| Respostas com termos de frequência | 10 | 16 |
| Fidelidade ajustada | 86,3% | 98,4% |
| Interpretação com fatos novos | 1 | 0 |

- O diagnóstico se confirmou: sem as contagens, a fidelidade voltou e passou do patamar da linha
  de base (94,7%). Com o intervalo de 35 s, os tempos voltaram ao normal (1,6 s).
- **Efeito "cobertor curto":** proibido de contar, o modelo trocou números por palavras
  ("os clientes elogiam principalmente X", "chegam frequentemente antes do prazo", "queixas
  recorrentes", "falha sistêmica"). É a mesma afirmação de frequência sobre uma amostra escolhida
  pelo assunto, mas sem número para conferir, e o juiz não pune.
- Das 5 respostas que o detector apontou com contagem, 4 eram falsos alarmes: 3 recusas
  ("nenhuma das 20 avaliações menciona...") e 1 trecho literal do cliente ("100% de descaso").
- **Detectores refinados:** passam a ignorar trechos entre aspas (fala do cliente), a seção
  Limitações e, nas contagens, as recusas. Recalculados para todas as configurações.

### Correção guiada pelo verificador
- Decisão: não fazer um prompt v5. Três versões de prompt (v2, v3, v4) não eliminaram as
  afirmações de frequência; insistir seria ajustar o sistema às 31 perguntas da prova.
- **Mecanismo:** se os detectores encontram termos de frequência ou contagens, uma chamada extra
  pede ao modelo que reescreva só essas expressões. O código aceita a correção apenas se: nenhuma
  citação nova, no máximo 20% de citações a menos, nenhum trecho inexistente ou registro como
  relato a mais e menos problemas de frequência. Senão, fica a resposta original.
- **Avaliação isolada:** a correção foi aplicada às respostas v4 já salvas (mesmas perguntas,
  evidências e respostas); a única diferença entre as configurações é a correção.
- **Correção v1** (guardada como `..._corrigida_v1`): 16 aplicadas, 0 rejeitadas. Zerou termos e
  contagens, mas apagar o termo **universalizava** a frase: "a grande maioria das entregas ocorreu
  no prazo" virou "essas entregas ocorreram no prazo"; numa Interpretação, a frase nova contradizia
  os atrasos citados logo depois.
- **Correção v2:** troca a frequência por **existência** ("há relatos de X", "há clientes que...")
  e proíbe afirmações sobre todos.

| Medida (Qwen) | v4 | Correção v1 | Correção v2 |
|---|---|---|---|
| Respostas com termos de frequência / contagem | 16 / 1 | 0 / 0 | 0 / 0 |
| Fidelidade ajustada | 98,4% | 96,9% | 97,7% |
| Interpretação com fatos novos | 0 | 2 | 0 |

- As 2 parciais da correção v2 não vêm dela (uma citação trocada que já existia no v4 e uma
  implicância do juiz entre "no prazo" e "antes do prazo"). A diferença para o v4 é de cerca de
  uma afirmação em 129, dentro da variação do juiz.

### Configuração final da Etapa 5
Busca densa + reescrita da pergunta + reranker, prompt v4 e correção v2 guiada pelo verificador
(padrão do `pipeline` e da linha de comando).

| Medida | Linha de base (Etapa 4) | Final (Etapa 5) |
|---|---|---|
| Precisão da busca (@20) | 0,821 | 0,935 |
| Comportamento correto | 100% | 100% |
| Fidelidade ajustada | 94,7% | 97,7% |
| Registro citado como relato | 1 | 0 |
| Respostas com termos de frequência / contagem | 13 / 15 | 0 / 0 |
| Tempo por pergunta | ~1,7 s | ~7 a 9 s |

- **Custo:** a reescrita (~1,5 s), o reranker (~5 s) e a correção (~1,5 s, em cerca de metade das
  respostas) somam alguns segundos, aceitáveis para quem espera uma análise.
- **Limitações declaradas:** juiz de fidelidade leniente com erros sutis (números são teto);
  inversões de sentido não são detectadas automaticamente; a P13 segue a pergunta mais difícil
  para a busca (precisão 0,55); a reescrita custa precisão nas perguntas agregadas (viés de
  confirmação), tratadas na Etapa 6.