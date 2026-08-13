# Architectural Blueprint: Kaggle Platform Dynamics & Sampling Bias

## 1. O Paradoxo do Volume Isolado
Durante o cruzamento dos perfis do banco de dados relacional (`results.db`) contra o Leaderboard estático (`kaggle_leaderboard.csv`), um abismo estrutural emergiu na arquitetura da competição de Pokémon TCG AI Battle. 
O ecossistema oficial possui **6.791 times únicos**. Contudo, apenas 1.059 times figuram em todo o registro histórico de partidas exportadas pelo mecanismo de ZIPs (replays).

O Kaggle falha em exportar a integralidade das simulações que rodam no *backend*.

## 2. A Métrica de Cobertura Diária (Day-by-Day Extraction)
Para entender se o isolamento de dados ocorria exclusivamente por mecânicas tardias de fusão (*merge* das equipes perto da *deadline* de 10 de Agosto), dissecamos dia a dia o percentual dos times oficiais que efetivamente geravam rastros dentro dos arquivos `.json` de log de episódio.
A extração provou que a fratura operou retroativamente desde o primeiro dia (14 de Julho):

- **Período de Julho**: As extrações fluíam a uma taxa de **1.05% a 2.37%** de cobertura do *pool* total.
- **Período de Agosto (Pré-Deadline)**: Com a escalada massiva de submissões e o teto da infraestrutura, a cobertura máxima que os JSONs locais atingiram foi um pico de **5.57%**.

Isso estipula que cerca de **95% da base de jogadores ativa na plataforma opera no completo anonimato estatístico** a cada ciclo de exportação diária.

## 3. O Efeito de Cegueira sobre a Elite (Top-Tier Masking)
A anomalia não recai apenas sobre usuários casuais ou bots primitivos jogando nas faixas baixas de Elo. A filtragem de replays omite seletivamente algumas das maiores referências técnicas do Leaderboard.
Na análise global, documentamos equipes ranqueadas no Top 10, Top 25 e Top 100 absoluto que não possuem sequer um registro de partida espelhado no banco de replays local:
* *AlphaTCG (Rank 10)*
* *Mahog (Rank 25)*
* *Vibrava (Rank 93)*

Assim como o time orgânico *Fitalabs*, estas equipes são exemplos vivos de pontos cegos. A lógica interna do *matchmaking* do Kaggle que dita *quais* episódios viram *dump* para o público é hermética (talvez limitação randômica, cap baseada em nó *worker*, ou compressão predatória). 

## 4. Implicações para Treinamento e Alinhamento Político (GRPO / RL)
Desenvolver políticas baseadas estritamente na modelagem estatística bruta do `results.db` gerará sobreajuste catastrófico (*overfitting*) contra as instâncias enviesadas. Como estamos condicionados a operar com menos de 15% do conhecimento estrutural total da competição:
1. Deve-se aplicar **Elo Invariante** (Bradley-Terry com suavização $N_0$) para que a raridade estatística de encontros ($N$ < 5 para equipes *top-tier*) não deforme a curva de calibragem da função de recompensa.
2. O aprendizado de comportamento (BC - Behavioral Cloning) herda uma imagem estilhaçada do Meta da plataforma. Ele deve atuar apenas como fundação; qualquer lapso lógico no tabuleiro deve ser sanado internamente com *Self-Play* PPO e correções semânticas por TBPTT. A "verdade" do Leaderboard não pode ser deduzida dos replays sem ceticismo.
