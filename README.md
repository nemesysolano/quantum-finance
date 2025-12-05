# Quantum Trading with Gaps #

## Trend Run ##

### Fast Trend Run ${R_{f}(t)}$ ##

The purpose of the **fast trend run** ${R_{f}(t)}$ is to quantify the magnitude and direction of the last directional price push. 

Let $t_a$ and $t_b$ be two consecutive moments in the time series. A fast trend begins when the closing price changes direction between two consecutive bars, $a$ and $b$ (i.e., $c(t_a) < c(t_b)$ or $c(t_a) > c(t_b)$). When a **fast trend** begins, we denote $t_f = t_a$ as the **starting point** of the **fast trend**, and define:

${R_f(t) = c(t) - c(t_f)}$: The trend run (signed).

---

### Slow Trend Run ${R_s(t)}$ ##

The purpose of the **slow trend run** is to gauge the magnitude and direction of the run since the structural trend was first established. The formulae is similar to the fast trend run, except for the **starting point**. We define:

${R_s(t) = c(t) - c(t_s-1)}$: The trend run (signed), where $c(t_s-1)$ is the closing price of the bar *before* the structural trend starting point $t_s$.

Now we need a clear definition of ${t_s}$ and for that purpose we will introduce the **Structural Direction (${S_d(t)}$)** concept.

---

## The Structural Direction ${S_d(t)}$ ##
The structural direction captures when both a structural support/resistance and a price low/high are moving in the same direction.

| Priority | Condition | Structural Trend | ${S_d(t)}$ |
|----------|-----------|------------------|------------|
| **1** | ${h(t) > h(t-1)}$ and ${l(t) \ge l(t-1)}$ | **Ascending** (New High Structure) | +1 |
| **2** | ${l(t) < l(t-1)}$ and ${h(t) \le h(t-1)}$ | **Descending** (New Low Structure) | -1 |
| **3** | Otherwise (${h(t) \le h(t-1)}$ and ${l(t) \ge l(t-1)}$ **OR** ${h(t) > h(t-1)}$ and ${l(t) < l(t-1)}$) | **Continuation** (No Structural Change) | ${S_d(t-1)}$ |

### Definition of ${t_s}$ (Slow Trend Run Starting Point) ###

The slow trend run only starts (${t_s=t}$) when the structural direction changes, signifying the beginning of a new run and the end of the previous one.

Scenario                                 | Condition | Update for ${t_s}$
-----------------------------------------|-----------------------|----------------
Structural Reversal (Start of a new run) | ${S_d(t) \ne S_d(t-1)}$ | ${t_s = t}$
Trend Continuation (Run is Going)        | ${S_d(t) = S_d(t-1)}$ | ${t_s = t_{s-1}}$


## The Breaking Gap ##

Suppose that a new bar ${(o(t), h(t), l(t), c(t))}$ comes along violating the trend. We define the breaking gap ${G(t)}$ as the distance between the violated structural level and the extreme price that violated it. This is valuable for determining the age and total magnitude of the structural level being tested.

### Trend Violation for Slow Ascending Trends ###

If the slow trend at time ${t}$ is **ascending**, the violation occurs when the price breaks **below the structural low** ${l(t-2)}$.

If ${l(t) < l(t-2)}$, the Breaking Gap is defined as ${G(t) = l(t-2) - l(t)}$. The magnitude of this breach becomes the new **Last Structural Breach Value** ($G_b$).

### Trend Violation for Slow Descending Trends ###

If the slow trend at time ${t}$ is **descending**, the violation occurs when the price breaks **above the structural high** ${h(t-2)}$.

If ${h(t) > h(t-2)}$, the breaking Gap is defined as ${G(t) = h(t) - h(t-2)}$. The magnitude of this breach becomes the new **Last Structural Breach Value** ($G_b$).

---

If no structural breach occurs (i.e., neither ${l(t) < l(t-2)}$ nor ${h(t) > h(t-2)}$ is observed), the **Breaking Gap** ${G(t)}$ decays over time. The decay is **linear** and proportional to the time elapsed since the last structural breach.

We define $G_b$ as the magnitude of the **most recent structural breach** (the **Last Structural Breach Value**). Let $k$ be the number of bars elapsed since $G_b$ was set (where $k=1$ is the first bar after the breach). The value of $G(t)$ is calculated by:

$G(t) = \mathbf{max}(G_b - k \cdot Q, 0)$

The initial value of ${G(t)}$ is zero. The term ${Q}$ denotes the **quantization delta** (the per-bar decay rate). This implements a decay mechanism where the magnitude of the gap at any point in time is relative to the **initial strength** of the last breach, and this influence gradually diminishes proportionally to the time elapsed until a new structural breach occurs.

---

## The Swing Ratio ${S(t)}$ ##
Consider the bar sequence ${\mathbf{OHLC}}$ = ${(o_{t-n}, h_{t-n}, l_{t-n}, c_{t-n}),...,(o(t), h(t), l(t), c(t))}$ up to the current bar ${t}$; it is assumed that this sequence is longer than 2 bars.

### The Fast Swing Ratio ${S_f(t)}$ ###
The **fast swing ratio** ${S_f(t)}$ is calculated using the **Breaking Gap** ${G(t)}$ and the magnitude of the **fast trend** ${|R_f(t)|}$:

$S_f(t) = \mathbf{min}\left(2, \left(\frac {G(t)}{|R_f(t)|}\right)^2\right)$

*Note: If $|R_f(t)| = 0$, $S_f(t)$ should be defined separately (e.g., $S_f(t) = 0$).*

### The Slow Swing Ratio ${S_s(t)}$ ###

We define the **last opposite to $R_s(t)$** (namely $R^*_s(t)$) as the most recent element in the sequence of slow trend runs ($R_s(t-n),...,R_s(t-1)$) whose direction is opposite to $R_s(t)$.

The slow swing ratio is calculated using the magnitudes of the current slow trend run and the last opposite run:

$S_s(t) = \mathbf{min}\left(2, \left(\frac {|R_s(t)|}{|R^*_s(t)|}\right)^2\right)$

*Note: If $|R^*_s(t)| = 0$, $S_s(t)$ should be defined separately (e.g., $S_s(t) = 0$).*

---

## Directional Probabilities ##
The directional probabilities at time ${t}$ estimate the likelihood that the price will move in a particular direction in the next period ($t$). We assume that directional probabilities occur pair-wise and are complementary (i.e., $P_↑(t) + P_↓(t) = 1$).

### Conflicting Trends ###
In this case, the slow and fast trends run in the opposite directions.

#### Fast Ascending Trend vs Slow Descending Trend ####
If the fast trend is ascending and the slow trend is descending, the directional probabilities are:

$P_↓(t) = \frac {S_f(t)}{S_f(t) + S_s(t)}\ \text{when}\ S_f(t) > 0\ \text{or}\ 0.5\ \text{otherwise.}$

$P_↑(t) = 1 - P_↓(t)$

#### Fast Descending Trend vs Slow Ascending Trend ####
If the fast trend is descending and the slow trend is ascending, the directional probabilities are:

$P_↑(t) = \frac {S_f(t)}{S_f(t) + S_s(t)}\ \text{when}\ S_f(t) > 0\ \text{or}\ 0.5\ \text{otherwise.}$

$P_↓(t) = 1- P_↑(t)$
 

### Aligned Trends ###

In this case, both trends run in the same direction.

#### Ascending Trends ####
When both trends are ascending:

$P_↑(t) = \mathbf{min}(1,\frac {|S_s(t)|} {4} + |S_f(t)|)$

$P_↓(t) = 1- P_↑(t)$

#### Descending Trends ####

$P_↓(t) = \mathbf{min}(1,\frac {|S_s(t)|} {4} + |S_f(t)|)$

$P_↑(t) = 1- P_↓(t)$

## Price-Volume Oscillator ##

We define ${Y(t)}$ as the **price-volume** oscillator:

$Y(t) =  2 \frac {p(t)-p(t-1)}{p(t)+p(t-1)} \frac {v(t)}{v(t-1)}$, where

- ${p(t)}$ price at time ${t}$
- ${p(t-1)}$ price at time ${t-1}$
- ${v(t)}$ traded volume at time ${t}$
- ${v(t-1)}$ traded volume at time ${t-1}$

This oscillator detects strong bullish (${Y(t) \rarr +\infty}$) or bearish (${Y(t) \rarr -\infty}$) behavior. Moreover, ${p(t)}$ represents ${h(t)}$ (high price in OHLC bar) when
analysing resistance/bearish momentum, ${l(t)}$ (low price in OHLC bar) when analysing support/bullish momentum or ${c(t)}$ close price when analysing trends.

## Baseline Forecast Models ##

In order to assess how much improvement quantum mechanics can add to existing neural network models, we are going to draft two baselines models; these  will be
subsequently improved using quantum mechanics.

### Probability Difference Forecast ###

This model forcasts **probability difference** at time $t$, denoted as $P_d(t)$, from last $k$ probability differences.

#### Prediction Target ####

The probability difference defined above which a signed value ranging in [-1,1].

#### Input Features ####

The model uses a fixed **lookback window of $k$ bars** (from $t-1$ to $t-k$) and the features are **probability differences** ($P_d(t)$) defined below:

| Feature | $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- | :--- |
| $P_d(t)$ | $P_d(t-1)$ | $P_d(t-2)$ | ... | $P_d(t-k)$ |

where $P_d(t-k) = P_↑(t-k) - P_↓(t-k)$

### Price-Volume Difference Forecast ###

This model forcasts ****price-volume difference**** at time $t$, denoted as $Y_d(t)$, from last $k$ consecutive $Y(t) - Y(t-1)$ differences.

#### Prediction Target ####

The **price-volume difference** difference mentioned above and defined below which a signed value.

#### Input Features ####

The model uses a fixed **lookback window of $k$ bars** (from $t-1$ to $t-k$) and the features are ****price-volume difference** ($Y_d(t)$) defined below:

| Feature | $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- | :--- |
| $Y_d(t)$ | $Y_d(t-1)$ | $Y_d(t-2)$ | ... | $Y_d(t-k)$ |

where $Y_d(τ) = Y(τ) - Y(τ-1)$