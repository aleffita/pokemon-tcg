# Architectural Blueprint: Entity Normalization & Hierarchical Resolution Heuristics

## 1. O Problema de Domínio: A "Morte" Aparente nos Replays
No ecossistema de submissões do Kaggle Pokémon TCG AI Battle, a estrutura de dados diária extraída via `.json` (replays) sofre de uma patologia de continuidade de identidade. Observamos que dezenas de equipes de alto volume competitivo repentinamente paravam de jogar partidas (ausência completa nos novos ZIPs) muito antes da *deadline* de fusão (10 de Agosto).

A hipótese inicial apontava para banimentos em massa ou exclusões voláteis (~12% da plataforma extinta). Porém, a premissa ignorava a restrição infraestrutural da API do Kaggle: **limite estrito de 2 submissões ativas por TeamId**.

## 2. A Dinâmica de Merge e o Corte de Submissões
Quando o Jogador A e o Jogador B se fundem num *Team* consolidado, a plataforma imediatamente aplica a restrição de instâncias. Submissões ativas que ultrapassam o limite de 2 *slots* por time são desligadas (inativadas do pool de *matchmaking* de Elo). 
Consequentemente, a assinatura da submissão antiga "morre" no SQLite local (pois seu `EpisodeId` para de ser sorteado nas partidas), mas os indivíduos por trás dela estão absolutamente ativos sob um novo `TeamName` (ex: Fitalabs).

Isso resulta em um banco populado por Sub-Grafos Órfãos (Fantasmas): `TeamIds` antigos que contêm volume massivo de partidas e de repente evaporam, sujando a massa de cálculo do Modelo de Avaliação.

## 3. Matriz de Resolução Hierárquica (L1/L2)
Para expurgar falsos positivos sem corromper a integridade dos dados, implementamos um funil heurístico de 2 camadas lógicas que não interfere de maneira arbitrária no banco de dados.

### Nível 1: Conservação de Topologia Externa (Diagonal de Cantor Expandida)
O filtro primário, **L1**, faz o cruzamento absoluto entre a listagem morta do banco de dados (Vol > 50, *Dead* < Merge Deadline) e a tabela de verdade externa atualizada (`kaggle_leaderboard.csv`).
A Diagonal de Cantor opera varrendo a base histórica inativa contra duas colunas exclusivas do Leaderboard:
1. `TeamMemberUserNames`: Varredura em *split* iterando por cada conta de usuário vinculada a uma equipe (captura membros absorvidos).
2. `TeamName`: Varredura de colisão direta para capturar usuários cujo nick interno da plataforma diverge do nome público atual.

*Resultado:* O L1 absorveu brutalmente o viés amostral. Dos 144 fantasmas mapeados de alto volume, **142 possuíam correspondência direta** no Leaderboard oficial. Eles não foram deletados, sofreram apenas fusão e tiveram seus *EpisodeIds* abortados.

### Nível 2: Footprint Tracking por Janela Deslizante (ΔE)
Para as entidades que não colidiam sob hipótese alguma com a topologia externa, usamos rastreabilidade semântica e probabilística focada no *Deck*.
O **L2** detecta equipes que mantêm a distribuição exata de *Deck* (assinatura de array com 60 instâncias de *CardIds*) ao longo de curtas distâncias de *EpisodeId* (< 2.000). A transição de identidade se prova pela intersecção da assinatura criptográfica do Deck atrelada à temporalidade do corte do Kaggle.

*Resultado:* O L2 absorveu mais 1 fantasma, provando transição lateral com nome obfusccado.

## 4. O Núcleo Residual (Anomalia Absoluta)
O expurgo algorítmico revelou uma resiliência de 99,3% no ecossistema do Kaggle. De todos os fantasmas de grande volume estudados, apenas **um (Dieter - 1.068 Partidas, Morto Ep 88337213)** não retornou colisão matemática com o Leaderboard pós-deadline nem assinou pegadas laterais de migração de deck.
Ele constitui o Evento de Extinção Fática: a única conta plausível de banimento, exclusão violatória ou abandono terminal.

## 5. Diretriz de Normalização Conclusiva
A base de replays reflete fóssil operacional fiel de *drop* do Kaggle. Os dados "fantasma" não devem ser sobrescritos, fundidos de forma difusa no banco de dados, ou interpolados artificialmente. As anomalias falsas do banco não configuram erro, configuram o retrato preciso da API de *matchmaking* inativando execuções passadas durante as alianças pré-deadline. A tabela de agentes sobrevive íntegra por intermédio das chaves associadas a cada temporalidade.
