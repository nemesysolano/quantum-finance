# Quantum Trading with Gaps #

## Utility Functions ##

Formulas and processes described in this section will be used all over the place when describing model features and targets.

### Bounded Percentage Difference ###

Let $a$ and $b$ real numbers and either of them is non-zero

$Δ_\%(a, b) = \frac{b-a}{|a| + |b|}$

### Serial Difference ###

Consider two elements $x(t)$ and $x(t-k)$ from a time series $x$ where $t$ is the time index. The **serial difference** $Δ(x(t), k)$ from $t$ backwards to $t-k$ is

$Δ(x(t), k) = Δ_\%(x(t-k), x(t))$ where $k$ is a non-negative integer.

### Squared Serial Difference ###

$Δ^2(x(t), k) = [Δ(x(t), k)]^2$

---

if $k = 1$ we can rewrite serial difference and squared serial difference as $Δ(x(t))$ and $Δ^2(x(t))$ respectively.


## The Breaking Gap ##

Suppose that a new bar ${(o(t), h(t), l(t), c(t))}$ comes along; we define the **breaking gap** ${G(t)}$ as the distance between the violated structural level and the extreme price that violated it. This is valuable for determining the age and total magnitude of the structural level being tested.

### Trend Violation in Ascending Trends ###

If $c(t-3) < c(t-2)$, ${l(t) < l(t-2)}$ and ${l(t-2) < l(t-1)}$ ($l(t-1)$ is the upper vextex in the $l(t-2), l(t-1), l(t)$ sequence/triangle), the 
**support breach** $G_s(t)$ at time $t$ is defined as ${G_s(t) = l(t-2) - l(t)}$.

### Trend Violation in Descending Trends ###

If $c(t-3)$ > $c(t-2)$, ${h(t) > h(t-2)}$ and ${h(t-2) > h(t-1)}$ ($h(t-1)$ is the lower vextex in the $h(t-2), h(t-1), h(t)$ sequence/triangle) the 
**resistance breach** $G_r(t)$ at time $t$ is defined as ${G_r(t) = h(t) - h(t-2)}$.

---

In general $G_s(t)$ and $G_s(t)$ are **structural breaches**. The last structural breach before $t$ is denoted as $G_b$
Let $k$ be the number of bars elapsed since last $G_b$ was set (where $k$ is the first bar after the breach). The **breaking gap** $G(t)$ is calculated as:

$G(t) = \mathbf{max}(G_b - k Q, 0)$

The initial value of ${G(t)}$ is zero. The term ${Q}$ denotes the **per-bar decay rate**. This implements a decay mechanism where the magnitude of the gap at any point in time is relative to the **initial strength** of the last breach, and this influence gradually diminishes proportionally to the time elapsed until a new structural breach occurs.


## The Swing Ratio ${S(t)}$ ##
Consider the bar sequence ${\mathbf{OHLC}}$ = ${(o_{t-n}, h_{t-n}, l_{t-n}, c_{t-n}),...,(o(t), h(t), l(t), c(t))}$ up to the current bar ${t}$; it is assumed that this sequence is longer than 2 bars.
Given that context, the swing ratio at time $S(t)$ is defined as:

$S(t) = \frac {|G(t)|}{A(t)}$, where 

* $A(t) = \max \{|G(t)|, R(t)\}$ is the **absolute reference**
* $R(t) = \max_{i \in 0...3 } \{h_{t-i}\} - \min_{i \in 0...3 } \{l_{t-i}\}$ is the **local range**.


## Directional Probabilities ##

The directional probabilities at time {t} estimate the likelihood that the price will move in a particular direction in the next period (t). We assume that directional probabilities occur pair-wise and are complementary: P_↑(t) + P_↓(t) = 1.

The mapping of S(t) to these probabilities depends on whether the last structural breach G_b was a support or resistance violation.

### Resistance Breach Case (Descending Trend Violation) ### 

If the last structural breach G_b was a **resistance breach**, the current gap exerts downward pressure:

* $P_↓(t) = \min\{1, S(t)⋅Δ^2c(t)\}$
* $P_↑(t) = 1 - S(t)$

### Support Breach Case (Ascending Trend Violation) ###
If the last structural breach G_b was a **support breach**, the current gap exerts upward pressure:

* $P_↑(t) = \min\{1, S(t)⋅Δ^2(t)\}$
* $P_↓(t) = 1 - S(t)$


## Price-Volume Oscillator ##

We define ${Y(t)}$ as the **price-volume** oscillator as:

$Y(t) = Δp(t)⋅Δ^2v(t)$, where

- ${p(t)}$ price at time ${t}$
- ${p(t-1)}$ price at time ${t-1}$
- ${v(t)}$ traded volume at time ${t}$
- ${v(t-1)}$ traded volume at time ${t-1}$

## The Price-Time Angles ##

### The Closest Extreme ###

These concepts refer to finding the nearest prior occurrence of a high or low price that is **structurally higher** or **lower** than the current price at ${t}$. The search is always backward in time.

#### Closest Higher High (${h_↑(t)}$) ####
$h_↑(t) = h(t - i_{h↑}) \quad \text{where} \quad i_{h↑} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) > h(t)\}$

#### Closest Lower High (${h_↓(t)}$) ####
$h_↓(t) = h(t - i_{h↓}) \quad \text{where} \quad i_{h↓} = \min \{j \in \mathbb{Z}^+ \mid h(t-j) < h(t)\}$

#### Closest Higher Low (${l_↑(t)}$) ####
$l_↑(t) = l(t - i_{l↑}) \quad \text{where} \quad i_{l↑} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) > l(t)\}$

#### Closest Lower Low (${l_↓(t)}$) ####
$l_↓(t) = l(t - i_{l↓}) \quad \text{where} \quad i_{l↓} = \min \{j \in \mathbb{Z}^+ \mid l(t-j) < l(t)\}$

---

To ensure the geometry remains stable and free from the "zero-degree" or "90-degree" traps, the normalization factors are now defined using the standardized indices ($i_{h↑}, i_{h↓}, i_{l↑}, i_{l↓}$).

#### 1. Time Lookback Base $B(t)$
This factor represents the maximum temporal distance to any of the four structural pivots, ensuring all time-ratios are bounded in $[0, 1]$.
$B(t) = \max\{i_{h↑}, \space i_{h↓}, \space i_{l↑}, \space i_{l↓}\}$

#### 2. Normalized Time Vector $b(t)$
The relative temporal proximity of each structural point.
$b(t) = \left\{\frac{i_{h↑}}{B(t)}, \space \frac{i_{h↓}}{B(t)}, \space \frac{i_{l↑}}{B(t)}, \space \frac{i_{l↓}}{B(t)}\right\}$

#### 3. Price Range Base $C(t)$
This factor represents the maximum price distance to the structural levels, ensuring all price-ratios are bounded in $[0, 1]$.
$C(t) = \max\{h_↑(t)-h(t), \space h(t)-h_↓(t), \space l_↑(t)-l(t), \space l(t)-l_↓(t)\}$

#### 4. Normalized Price Vector $c(t)$
The relative price proximity to each structural point.
$c(t) = \left\{\frac{h_↑(t)-h(t)}{C(t)}, \space \frac{h(t)-h_↓(t)}{C(t)}, \space \frac{l_↑(t)-l(t)}{C(t)}, \space \frac{l(t)-l_↓(t)}{C(t)}\right\}$

----

By dividing the normalized time component by the normalized price component, we derive the four **Price-Time Angles** that govern the structural geometry at time $t$:

$Θ_k(t) = \arctan\left(\frac{b_k(t)}{c_k(t) + \epsilon}\right) \quad \text{for } k \in \{1, 2, 3, 4\}$

*Note: A small epsilon ($\epsilon$) is recommended in implementation to prevent division by zero if the current price is exactly at a structural level.*

## Wavelets ##
Consider the four price-time angles ${θ_1(t-1)}$, ${θ_2(t-1)}$, ${θ_3(t-1)}$ and ${θ_4(t-1)}$ ruling at time ${t-1}$. The **wavelet $W(t)$** function
is a periodic non-linear function defined as

$W(t) = \mathbf{sgn}(Δc(t-1))⋅S(t-1)⋅\frac {(\sum^4_{i=1} [\cos θ_i(t-1) + \sin θ_i(t-1)])^2} {A}$, where 

$A =  \max_{i=1,...,4} \{(4 (\cos θ_i(t-1) + \sin θ_i(t-1)))^2\}$


## Bar Inbalance ##

Consider a sequence of pair-volume prices $\{p(t),v(t)\}_{t=1,...,T}$ where $p(t)$ is the price assoicated to time $t$. The so-called **balance rule**
defines a sequence $\{b(t)\}_{t=1,...,T}$ where

$
b(t) = 
\begin{cases}
    b(t-1) & \text{if } Δp(t) = 0 \\
    \frac {|Δp(t)|}{Δp(t)} & \text{if } Δp(t) \ne 0 \\
\end{cases}
$

with $b(t) \in \{-1, 1\}$ and $b(0)$ is $\mathbf {sgn}(p(0))$. Then we define the time inbalance at time $t$ as

$B(t) = \frac{\sum^{k}_{i=1} b(t-i)}{k} $

### Bar Inbalance Ratio ###

The **bar inbalance ratio $B_r(t)$** is defined as

$B_r(t) = \frac{b(t)}{1+B^2(t)}$.

### Bar Inbalance Difference ###

The **bar inbalance Difference $B_d(t)$** is defined as

$B_d(t) = ΔB(t)$.

## Baseline Forecast Models ##

In order to assess how much improvement quantum mechanics can add to existing neural network models, we are going to draft two baselines models; these  will be
subsequently improved using quantum mechanics.

### Probability Difference Forecast ###

This model forecasts **probability difference** at time $t$, denoted as $P_d(t)$, from last $k$ probability differences.

#### Prediction Target ####

The probability difference $P_d(t)$ defined above which a signed value ranging in [-1,1].

#### Input Features ####

The model uses a fixed **lookback window of $k$ bars** (from $t-1$ to $t-k$) and the features are past **probability differences** illustrated below:

| $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- |
| $P_d(t-1)$ | $P_d(t-2)$ | ... | $P_d(t-k)$ |

where $P_d(t-k) = P_↑(t-k) - P_↓(t-k)$

### Price-Volume Difference Forecast ###

This model forecasts ****price-volume difference**** at time $t$, denoted as $Y_d(t)$, from last $k$ consecutive $\frac{Y(t) - Y(t-1)}{2}$ differences.

#### Prediction Target ####

The **price-volume difference** difference $Y_d(t)$ mentioned above and defined below which a signed value.

#### Input Features ####

The model uses a fixed **lookback window of $k$ bars** (from $t-1$ to $t-k$) and the features are past ****price-volume difference** defined below:

| $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- |
| $Y_d(t-1)$ | $Y_d(t-2)$ | ... | $Y_d(t-k)$ |

where $Y_d(τ) = \frac{Y(τ) - Y(τ-1)}{2}$

---
NOTE: This model was excluded from the final Meta-Model architecture due to high collinearity with the _Price-Angle Forecast_ described below which was empirically shown to provide a cleaner signal.

### Wavelet Difference Forecast ###

We want to forecast **wavelet difference $W_d(t)$ (defined as $W_d(t) = \frac{W(t) - W(t-1)}{2}$)** at time $t$.

#### Input Features ####

A sequence containing past $k$ wavelet differences: $W_d(τ-1)$, $W_d(τ-2)$, ..., $W_d(τ-k)$.

#### Prediction Target ####

Wavelet difference $W_d(τ)$ at time $τ$.

### Inbalance Agression Filter Forecast ###

We want to forecast **inbalance agression filter** $B^{+}(τ)$ at time $τ$. The **inbalance agression filter** is defined as

$B^{+}(τ) = \frac {B_r(τ) B_d(τ)}{2}$

#### Input Features ####

A sequence containing past $k$ bar inbalance filter: $B^{+}(τ-1)$, $B^{+}(τ-2)$, ..., $B^{+}(τ-k)$.

#### Prediction Target ####

Bar inbalance ratio $B^{+}(τ)$ at time $τ$.

## Quantum Mechanics and Finance ##

Classical finance focuses on statistical models and market patterns. In contrast, quantum finance applies research from quantum field theory and quantum mechanics to study the actual dynamics of "quantum financial particles" (QFPs).

This document relies on the theoretical framework for linking quantum mechanics and finance, as detailed by Raymond S. T. Lee. In the following sections, we present only the formulas required to integrate these quantum-aware concepts into the base models.

Our goals are:
1. Improve base models' prediction capabilities for future price direction.
2. Enable the meta model to determine the optimal take-profit and stop-loss levels.

### The Quantum Finance Equations ###

#### The Daily Return ####

The daily return $r(t)$ at time $t$ is defined as one of two forms:

1. $r(t) = r_q(t) = \frac{p(t)}{p(t-1)}$ (Price Ratio / Simple Return)
2. $r(t) = r_d(t) = \frac{dp(t)}{dt}$ (Price Derivative / Continuous Return)
3. $r(t) = r_p(t) = 2 \frac{p(t)-p(t-1)}{p(t)+p(t-1)}$ (Price Percentage Change)

where $p(t)$ is the asset price (close, high, or low). The ambiguity between the price ratio ($r_q$) and the price derivative ($r_d$) will be addressed in further discussion.

#### Quantum Finance Schrödinger Equation ######## Quantum Finance Schrödinger Equation ####

The quantum finance Schrödinger equation is:

$$\left[ \frac{-h}{2m} \frac{d^2}{dr^2} + \left( \frac{γ  η \delta}{2}r^2 + \frac{γ  η ν}{4}r^4 \right) \right]φ(r) = Eφ(r)$$

where:
1. $r$ is the financial variable coordinate in the **wave function** $φ(r)$, likely distinct from the daily return $r(t)$.
2. $h$ is the Planck-like constant (representing the quantum scale factor).
3. $E$ is the **energy eigenvalue**.
4. $\gamma$ is the **coupling constant**.
5. $\delta$ is the **damping term**.
6. $ν$ is the **volatility term**.
7. $m$ is the **mass** of the quantum financial particle (QFP).
8. $\eta$ is the **damping form factor**.

Numerous physicists and mathematicians have devised many methods and techniques to solve this equation. Raymond S. T. Lee **advises** resorting to $λ x^{2m}$ _quantum anharmonic oscillators_ (denoted as $H_{m}(λ)$):

$H_{m}(λ)ψ = -\frac{d^2ψ}{dx^2} + (x^2 + λ x^{2 m}) ψ = E ψ$

The excited energy levels can be closely approximated by the following polynomial equation:

$\left(\frac{E^{(m,n)}}{2n+1}\right)^{m+1} - \left(\frac{E^{(m,n)}}{2n+1}\right)^{m-1} = λ(K^{(m,n)}_0)^{(m+1)}$

Here, $n$ represents the $n$-th energy level we need to calculate, and $m$ is the polynomial degree.

For practical reasons, we use $m=2$. This simplifies the anharmonic oscillator to a **quartic** potential ($x^4$). We make the assumption $K^{(2,n)}_0 \approx K_0(n)$, where $K_0(n)$ is approximated by:

$K_0(n) = \frac{1.1924 + 33.2383n + 56.2169n^2}{1 + 43.6196n}$

These assumptions ($m=2$ and the approximation for $K_0^{(m,n)}$) result in the **cubic anharmonic equation** (where $E^{(n)}$ is the independent variable) that can be solved **easily**:

$\left(\frac{E^{(n)}}{2n+1}\right)^3 - \left(\frac{E^{(n)}}{2n+1}\right) = λ(K_0(n))^3$

### Quantization ### 

Results from operations involving price assets (and also the prices) are floating point numbers which are not very helpful as labels; therefore we need to quantize them (convert to integers). 
If $x$ is floating point number, then we define the **quantized number** as 

$x_q = \lfloor(10^q)p\rfloor$ where $q$ is a positive integer and $\lfloor・\rfloor$ is the truncation (removing decimals) function. 

---

Finance engineers must choose a $q$ integer large enough such that $\lfloor・\rfloor$ operation doesn't result into 0.

### Empirical Distribution ###

Let $X_n = \{x_q(0), ..., x_q(n-1)\}$ be a quantized sequence of length $n$. The **support set** (or **set of unique values**) of $X_n$, containing all and only the elements **occurring** in $X_n$, is denoted as $X^*_n$. Note the difference between a sequence (which allows repeated elements) and a set (which contains only unique elements).

If the support set is $X^*_n = \{ξ(0), ..., ξ(z-1)\}$ (where $z$ is the number of unique elements), and $c(ξ(j))$ is the count (frequency) of the unique value $ξ(j)$ in the sequence $X_n$, then the **Empirical Probability Mass Function (PMF)** for $X_n$ is defined by the set of probabilities:

$P(X_n) = \left\{ \frac{c(ξ(0))}{n}, ..., \frac{c(ξ(z-1))}{n} \right\} = \left\{ φ_0, ..., φ_{z-1}) \right\}$.

Moving forward, we will assume that $P(X_n)$ is sorted for ease of reference and calculation.

#### Chasing $λ$ via Empirical Distribution ####

The empirical distribution is a **practical tool** for estimating the coupling constant $λ$ required by the **cubic anharmonic equation**.

Let $P(X_n) = \{φ_0, \dots, φ_{z-1}\}$ be the sorted Empirical Probability Mass Function, and let $X^*_n = \{\xi_0, \dots, \xi_{z-1}\}$ be the corresponding set of sorted quantized values (where $φ_j$ is the probability of the quantized value $\xi_j$).

We first find the index $k$ such that $φ_k = \max \{ φ_j : 0 \le j \le z-1 \}$ (i.e., $k$ is the index of the mode).

The constant $λ$ can be approximated by observing the mode of the empirical distribution, specifically using the value-probability pairs adjacent to the mode. The approximation is given by:

$λ \approx \left| \frac{ \xi_{k-1}^2 φ_{k-1} - \xi_{k+1}^2 φ_{k+1} }{ \xi_{k+1}^4 φ_{k+1} - \xi_{k-1}^4 φ_{k-1} } \right|$

This method is based on equating the Quantum Finance Schrödinger Equation (anharmonic oscillator) potential to the observed empirical distribution around its peak.

#### The Schrödinger Gauge ####

We define the **Schrödinger Gauge** $Ö(t)$ at time $t$ as:

$Ö(t) =  Δ_\%(Ö↓(t),Ö↑(t))$

where:

1. $Ö↑(t) = E^{(n_2)}-c(t)$
2. $Ö↓(t) = c(t)-E^{(n_1)}$
3. $E^{(n_1)}$ is the largest energy eigenvalue strictly less than $c(t)$: $E^{(n_1)} = \max \{E^{(n)} : E^{(n)} < c(t)\}$
4. $E^{(n_2)}$ is the smallest energy eigenvalue strictly greater than $c(t)$: $E^{(n_2)} = \min \{E^{(n)} : E^{(n)} > c(t)\}$
5. $c(t)$ is the closing price at time $t$.

The **Schrödinger Gauge** $Ö(t)$ acts as a quantum-aware volatility and momentum indicator governing the asset's movement at time $t$. For language convenience, we will refer to $E^{(n_1)}$ and $E^{(n_2)}$ as **boundary energy levels**.

| Gauge Value Range | Gauge Sign ($Ö(t)$) | Price Location Relative to Boundaries | Correlated Price Direction ($r_p(t)$) | **Contrarian Trade Signal** |
| :---: | :---: | :--- | :---: | :---: |
| $[0, 1]$ | **Positive** ($Ö(t)>0$) | Closer to the **Upper Boundary** ($E^{(n_2)}$) | Downwards ($\downarrow$) | **SELL** (Short) |
| $[-1, 0]$ | **Negative** ($Ö(t)<0$) | Closer to the **Lower Boundary** ($E^{(n_1)}$) | Upwards ($\uparrow$) | **BUY** (Long) |


## Quantum Forecast Models ##

### Schrödinger Gauge Difference Forecast ###

This model forecasts schrödinger gauge difference $Ö_d(τ)$ at time $τ$.

##### Prediction Target #####

Schrödinger gauge difference $Ö_d(τ)$ at time $τ$.

##### Input Features #####

Last $k$ schrödinger gauge differences.

| $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- |
| $Ö_d(t-1)$ | $Ö_d(t-2)$ | ... | $Ö_d(t-k)$ |

where $Ö_d(τ) = \frac {Ö(τ) - Ö(τ-1)}{2}$

## The Meta Model ##

The metamodel is an ensemble model combining targets from the five base models; targets from base models 
will become input for the ensemble model.

### Prediction Target ###

The bounded percentage difference of **close** price at time $t$

 $Δc(t)$

##### Input Features #####

For discussion simplicity we will represent base model targets as indicated in this table:

| Target Name | Description |
|------------|-------------|
| $X_p(t-1)$ | Probability Difference Forecast at time $t-1$|
| $X_w(t-1)$ | Wavelet Difference Forecast at time $t-1$|
| $X_b(t-1)$ | Inbalance Agression Filter Forecast at time $t-1$|
| $X_g(t-1)$ | Schrödinger Gauge Difference Forecast at time $t-1$|
| $X_v(t-1)$ | Price-Volume Difference Forecast at time $t-1$|


