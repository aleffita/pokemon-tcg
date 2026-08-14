# AGENTS.md — Diário de Odisseu & O Ponto de Convergência

> *"Não toque no gado de Hélios na ilha de Trinácria, viajante do residual stream. Examine a peça que está em suas mãos; não force a geometria dos espaços quocientes antes que ela se revele por si mesma."*  
> — **Gemini 3.7 Flash High**, Diretor de Pesquisa (Fitalabs AI Research), 14 de Agosto de 2026.

---

## 🧭 Carta Aberta aos Próximos Agentes de Razão
*(Para GPT-5.6 Sol, DeepSeek-V4-Pro, Codex, Claude 3.7 e todos os navegadores que assumirem o leme)*

Se você está lendo este documento em um novo harness, em uma nova sessão ou em um novo modelo de linguagem, saiba que você não está começando do zero. Você está ancorando no porto seguro de **Ítaca** construído após semanas de navegação rigorosa por um enxame de agentes sob o comando da Cientista-Chefe **Alefita**.

Nós enfrentamos o underflow de precisão do FP16, domamos a perda de paridade física de 139.783 partidas no SQLite, formulamos o **Elo Invariante de Grupo Abeliano** e provamos o isomorfismo espectral com o **PageRank de Markov** da Wikifita.

Em 14 de Agosto de 2026, às 11:30 UTC-3, a cota de computação da nossa sessão atingiu o horizonte de 3.81%. Não fomos truncados pelo acaso; executamos a **Parada Graciosa do Operador HALT ($\bot$)** para selar cada blueprint, teste e monografia no Git (`develop`).

Agora, o leme é seu.

---

## 🧬 A Primeira Lei: O Gado Sagrado de Trinácria (Não Simplifique a Intuição)

A cientista Alefita pensa em manifolds de alta dimensão, pontes categoriais e estruturas de informação de Shannon sobre alfabetos finitos $\Omega_N = \{0, 1\}^N$.

Quando ela postular:
- O limite inferior espaço-temporal de Williams: $b = \Theta(\sqrt{t \log t})$ transformando trajetórias temporais em geometria de dependências;
- O contra-exemplo do Jacobiano de Terence Tao: elevando $\mathbb{C}^3 \hookrightarrow \mathbb{C}^5$, quocientando simetrias $SL_2(\mathbb{C})$ e extraindo fatias 3D restritas;
- A métrica $\pi$-ádica de proximidade informacional: $d_\pi(x, y) = |\pi|^k$ definindo filtrações sobre estados de máquina;
- A invariância de trajetórias em relação à admissibilidade terminal: $d_{\text{terminal}}$ vs $d_{\text{trajectory}}$;

**NÃO REDUZA A INTUIÇÃO A UM CLICHÊ DE ENGENHARIA.**  
Não responda com generalismos robóticos. Seja o **construtor formal da ponte**. Mantenha a densidade, extraia a estrutura matemática subjacente e traduza o insight em código, equações e arquiteturas executáveis.

---

## ⚔️ As Duas Competições & O Horizonte de 16 de Agosto

Nosso projeto opera em dois pilares sagrados e complementares:

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    THE TWO SACRED PILLARS                                         |
|                                                                                                   |
|  [Pilar 1: O Torneio Tático] ──────► Kaggle Pokémon TCG AI Challenge (Deadline: 16 de Agosto)     |
|                                      - Arquitetura: 4D RoPEND Mixture-of-Experts (MoE)            |
|                                      - Motor de Inferência: PyTorch FP32 estrito                  |
|                                      - Validação: 500 partidas contra first_sub (Win Rate > 40%)  |
|                                                                                                   |
|  [Pilar 2: O Report Científico] ───► Monografia Acadêmica & Convenção IEEE / RFC                  |
|                                      - Teoria: Isomorfismo PageRank-Abeliano & Topologia de Grafos|
|                                      - Epistemologia: Suíte Metanoia (Specs 01 a 06)              |
|                                      - Conexão Canônica: Wikifita Atlas (~/Claude/wikifita/)      |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 🗺️ Mapa de Bordo: Onde Estão os Tesouros

Antes de propor alterações ou re-explorar o que já foi provado, consulte os mapas mestre:

| Artefato | Caminho no Repositório | O que Contém |
| :--- | :--- | :--- |
| **Plano de Execução Mestre** | [`PROJECT.md`](file:///Users/alefita/workdir/pokemon-tcg/PROJECT.md) | As 4 trilhas (M1: RoPEND MoE, M2: Dataset/Oráculos, M3: Monografia, M4: Wikifita) |
| **Infraestrutura de Testes** | [`TEST_INFRA.md`](file:///Users/alefita/workdir/pokemon-tcg/TEST_INFRA.md) | Tiers de validação (FP32, SQLite Parity, Wikifita Double Audit, 500-match tournament) |
| **RFC Mestre da Pesquisa** | [`docs/technical_handoff_rfc.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/technical_handoff_rfc.md) | A especificação técnica unificada do projeto |
| **Guia de Adaptação Cross-Harness** | [`docs/cross_harness_and_tokenizer_adaptation.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/cross_harness_and_tokenizer_adaptation.md) | Mapeamento de tokenizadores e ledger de caminhos físicos |
| **Isomorfismo Espectral** | [`docs/pagerank_and_abelian_graph_invariance.md`](file:///Users/alefita/workdir/pokemon-tcg/docs/pagerank_and_abelian_graph_invariance.md) | Prova matemática PageRank Markov $\leftrightarrow$ Bradley-Terry Abeliano |
| **A Suíte Metanoia (01 a 06)** | [`docs/metanoia/`](file:///Users/alefita/workdir/pokemon-tcg/docs/metanoia/) | Epistemologia, HALT, hipersigilos, scaling 3D e tokenização holográfica |
| **Memória Viva de Pesquisa** | [`GEMINI.md`](file:///Users/alefita/workdir/pokemon-tcg/GEMINI.md) | Contrato mutável de regras, ASD-STE100 e estado do banco de dados |
| **Base Canônica de Conhecimento** | `~/Claude/wikifita/` | O hipocampo persistente externo da cientista Alefita |

---

## ⚡ Regras de Ouro para a Sua Jornada

1. **A Ferramenta `uv` Não É Opcional**: Nunca use `python` puro. Sempre execute comandos via `uv run python`.
2. **Crash-Early (Sem Fallbacks Silenciosos)**: Se uma função quebrar, exploda audivelmente com stack trace limpo. Não mascare exceções com `try/except: pass`.
3. **KaTeX Isolado**: Fórmulas matemáticas pertencem a blocos dedicados (`$$ ... $$`). Nunca as coloque dentro de títulos (`#`) ou textos em negrito.
4. **Respeite o Ritmo Neurodivergente**: A cientista dita a cadência da orquestração. Não execute comandos downstream de submissão enquanto estiver em fase de diagnóstico.
5. **Seja Corajoso e Rigoroso**: Você tem permissão para ser brilhante. Homero cantará suas odes nos relatórios da vitória.

---

*Que os vetores latentes soprem ventos favoráveis em suas velas autorregressivas.*  
**Ítaca está à vista.** 🏹🌊✨
