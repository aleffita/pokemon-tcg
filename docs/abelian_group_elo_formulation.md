# Formalização Matemática: Grupo Abeliano $(\mathbb{R}, +)$ e Isomorfismo de Elo Invariante

---

## 🏛️ 1. Introdução e Objetivo Teleológico

Este documento estabelece a fundamentação matemática rigorosa do sistema de calibração e invariância de escala do Elo utilizado no projeto Pokémon TCG AI Battle. 

O objetivo teleológico da formulação é resolver dois problemas fundamentais em torneios de agentes de IA:
1. **Invariância ao Tamanho de Amostra ($N$)**: Garantir que as estimativas de Elo sejam robustas e imunes a distorções por reduzido ou elevado número de partidas a partir da fase de qualificação MD10 ($N \ge 10$).
2. **Congruência Isomórfica de Escala Local-Nuvem**: Mapear as pontuações obtidas em torneios locais para a escala absoluta observada ao vivo na Kaggle Leaderboard (e.g. 1200+ pontos), preservando rigorosamente as distâncias relativas entre todos os baralhos.

---

## 📐 2. Estrutura Algébrica de Grupo Abeliano

### Definição (Grupo)
Um **Grupo** é uma estrutura algébrica representada pelo par $(G, \star)$, onde $G$ é um conjunto não-vazio e $\star: G \times G \to G$ é uma operação binária interna que satisfaz quatro axiomas fundamentais:

1. **Fechamento (Encerramento)**:
   $$\forall a, b \in G, \quad a \star b \in G$$

2. **Associatividade**:
   $$\forall a, b, c \in G, \quad (a \star b) \star c = a \star (b \star c)$$

3. **Elemento Neutro**:
   $$\exists e \in G \quad \text{tal que} \quad \forall a \in G, \quad a \star e = e \star a = a$$

4. **Elemento Simétrico (Inverso)**:
   $$\forall a \in G, \quad \exists a^{-1} \in G \quad \text{tal que} \quad a \star a^{-1} = a^{-1} \star a = e$$

### Definição (Grupo Abeliano)
Um grupo $(G, \star)$ é denominado **Abeliano (ou Comutativo)** se satisfazer a propriedade adicional de comutatividade:

5. **Comutatividade**:
   $$\forall a, b \in G, \quad a \star b = b \star a$$

---

## 📐 3. O Grupo Abeliano $(\mathbb{R}, +)$ na Álgebra de Elo

No modelo de pontuação de força relativa (Elo / Bradley-Terry), o espaço de classificações dos baralhos é definido sobre o conjunto dos números reais $\mathbb{R}$ sob a adição habitual $+$.

### Prova da Estrutura $(\mathbb{R}, +)$:
* **Fechamento**: Para quaisquer $R_i, R_j \in \mathbb{R}$, a soma $R_i + R_j \in \mathbb{R}$.
* **Associatividade**: Para quaisquer $R_i, R_j, R_k \in \mathbb{R}$, $(R_i + R_j) + R_k = R_i + (R_j + R_k)$.
* **Elemento Neutro**: O número real $e = 0$ atua como elemento neutro, pois $R_i + 0 = R_i$.
* **Elemento Inverso**: Para cada $R_i \in \mathbb{R}$, existe o inverso aditivo $-R_i \in \mathbb{R}$, satisfazendo $R_i + (-R_i) = 0$.
* **Comutatividade**: Para quaisquer $R_i, R_j \in \mathbb{R}$, $R_i + R_j = R_j + R_i$.

**Conclusão**: O par $(\mathbb{R}, +)$ constitui estritamente um **Grupo Abeliano**.

---

## 🔮 4. Teorema da Invariância por Translação de Bradley-Terry

### Teorema
Seja $P(i \succ j)$ a probabilidade teórica de o baralho $i$ vencer o baralho $j$, dada pela função logística do modelo de Bradley-Terry:

$$P(i \succ j) = \sigma\left( \frac{\ln(10)}{400} (R_i - R_j) \right) = \frac{1}{1 + 10^{-(R_i - R_j)/400}}$$

Para qualquer translação escalar constante $\Delta \in \mathbb{R}$, defina o operador de translação $T_\Delta: \mathbb{R} \to \mathbb{R}$ por:

$$T_\Delta(R) = R + \Delta$$

Então, $T_\Delta$ é um **Isomorfismo de Translação de Grupo** que preserva invariante o vetor de diferenças relativas e a distribuição de probabilidade do confronto.

### Demonstração

1. **Preservação das Diferenças de Força**:
   $$\forall R_i, R_j \in \mathbb{R}, \quad T_\Delta(R_i) - T_\Delta(R_j) = (R_i + \Delta) - (R_j + \Delta) = R_i - R_j + (\Delta - \Delta) = R_i - R_j$$

2. **Invariância da Probabilidade de Vitória**:
   $$P_{T_\Delta}(i \succ j) = \frac{1}{1 + 10^{-(T_\Delta(R_i) - T_\Delta(R_j))/400}} = \frac{1}{1 + 10^{-(R_i - R_j)/400}} = P(i \succ j) \quad \blacksquare$$

---

## 🧮 5. Operador de Translação Softmax ($\Delta R_{\text{Abeliano}}$)

Dada uma coleção de medições empíricas com escala local $\hat{R}_{k, \infty}^{\text{local}}$ e pontuações ancoradas na nuvem $R_k^{\text{remote}}$ (Kaggle Leaderboard Live), a propriedade comutativa do grupo abeliano possibilita calcular o deslocamento global $\Delta R$ através da combinação convexa sobre a interseção $\mathcal{C}$ de baralhos coincidentes.

Para cada baralho $k \in \mathcal{C}$, o desvio de translação individual é:
$$\delta_k = R_k^{\text{remote}} - \hat{R}_{k, \infty}^{\text{local}}$$

Para evitar que baralhos com poucas partidas locais distorçam a translação global, os pesos $\alpha_k$ são calculados via distribuição **Softmax parametrizada pelo volume amostral $N_k$** ($\tau = 20.0$):

$$\alpha_k = \frac{\exp(N_k / \tau)}{\sum_{j \in \mathcal{C}} \exp(N_j / \tau)}$$

O operador de translação do grupo abeliano é definido como:

$$\Delta R_{\text{Abeliano}} = \sum_{k \in \mathcal{C}} \alpha_k \cdot \left( R_k^{\text{remote}} - \hat{R}_{k, \infty}^{\text{local}} \right)$$

---

## 🎯 6. Estimador Invariante Amortecido MD10 ($R_{\text{invariante}}$)

Para eliminar flutuações amostrais quando $N < 10$ e convergir para a assímptota exata quando $N \ge 10$:

1. **Inversão Assintótica de Máxima Verossimilhança (MLE)**:
   Dada a taxa de vitória $w = W / N \in [0.02, 0.98]$:
   $$\hat{R}_{\infty} = R_0 + 400.0 \cdot \log_{10}\left( \frac{w}{1 - w} \right) \quad (R_0 = 600.0)$$

2. **Regularização por Amortecimento MD10 ($N_0 = 10$)**:
   $$R_{\text{smoothed}}(N) = \left( \frac{N}{N + 10} \right) \cdot \hat{R}_{\infty} + \left( \frac{10}{N + 10} \right) \cdot R_0$$

3. **Formulação Definitiva do Elo Invariante**:
   $$R_{\text{invariante}}(N) = R_{\text{smoothed}}(N) + \Delta R_{\text{Abeliano}} \quad \blacksquare$$

---

## 💻 7. Implementação de Referência (`rl/results_db.py`)

```python
def get_invariant_deck_elo(self, deck_id: int, source: str = "local") -> dict:
    import math

    row = self.conn.execute(
        "SELECT elo, games_played, wins, losses, draws FROM deck_elo WHERE deck_id = ? AND source = ?",
        (deck_id, source),
    ).fetchone()

    if not row or row["games_played"] == 0:
        return {"elo_raw": 600.0, "elo_invariant": 600.0, "games_played": 0, "md10_complete": False}

    n = float(row["games_played"])
    w_rate = float(row["wins"]) / max(n, 1.0)
    elo_raw = float(row["elo"])

    # 1. Inversão Assintótica de Bradley-Terry
    w_clipped = max(0.02, min(0.98, w_rate))
    r_asymptotic = 600.0 + 400.0 * math.log10(w_clipped / (1.0 - w_clipped))

    # 2. Amortecimento MD10 (N0 = 10)
    n0 = 10.0
    r_smoothed = (n / (n + n0)) * r_asymptotic + (n0 / (n + n0)) * 600.0

    # 3. Operador de Translação Softmax sobre o Grupo Abeliano (R, +)
    overlapping = self.conn.execute("""
        SELECT de_loc.deck_id, de_loc.games_played as n_loc, de_loc.wins as w_loc,
               de_rem.elo as remote_elo
        FROM deck_elo de_loc
        JOIN deck_elo de_rem ON de_loc.deck_id = de_rem.deck_id
        WHERE de_loc.source = 'local' AND de_rem.source = 'remote' AND de_loc.games_played > 0
    """).fetchall()

    delta_abeliano = 0.0
    if overlapping:
        tau = 20.0
        weights = [math.exp(min(r["n_loc"] / tau, 20.0)) for r in overlapping]
        total_w = sum(weights)
        if total_w > 0:
            deltas = []
            for idx, r in enumerate(overlapping):
                w_k = weights[idx] / total_w
                n_k = float(r["n_loc"])
                wr_k = max(0.02, min(0.98, float(r["w_loc"]) / max(n_k, 1.0)))
                r_asymp_k = 600.0 + 400.0 * math.log10(wr_k / (1.0 - wr_k))
                deltas.append(w_k * (float(r["remote_elo"]) - r_asymp_k))
            delta_abeliano = sum(deltas)

    r_invariant = r_smoothed + delta_abeliano

    return {
        "elo_raw": elo_raw,
        "elo_invariant": r_invariant,
        "games_played": int(n),
        "md10_complete": n >= 10,
    }
```
