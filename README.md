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

### Logarithmic Difference (or Log-Return) ###

Let $a$ and $b$ be _strictly positive_ real numbers ($a,b > 0$). **The Log-Return** is defined as

$L(a,b) = \mathbf{ln}(\frac{b}{a})$

Along the same lines as **serial difference**, we can define the **logarithmic serial difference** as:

$L(x(t), k) = L(x(t-k), x(t))$.

---
Let's highlight some important points about $L(a,b)$:

1. Logarithmic Differences typically exhibit heavy tails and are not strictly normally distributed.
2. When making inference with L(a,b), especially with short samples (10 to 30 elements), the Student-t distribution is often used to better model the observed kurtosis.

### Logarithmic filter ###

 $ρ(x) = \frac{2}{\log{2}} \frac{\log(1+x)}{1+x}$ where,
 
 $x \in (ε,1]$ and 
 
 $ε = 9^{-5}$.

### Serial Bounded Ratio ###

Consider two elements $x(t)$ and $x(t-k)$ from strictly positive time series $x$ where $t$ is the time index. The **serial bounded ratio** $δ(x(t), k)$ from $t$ backwards to $t-k$ is

$δ(x(t), k) = ρ(\max(\frac{x(t)}{x(t-k)}, ε))$
---

if $k = 1$ we can rewrite serial difference, squared serial difference, serial bounded ratio and logarithmic serial differenceas $Δ(x(t))$, $Δ^2(x(t))$, $δ(x(t))$ and $L(x(t))$ respectively.


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

* $P_↓(t) = S(t)⋅δ(x(t))$
* $P_↑(t) = 1 - S(t)$

### Support Breach Case (Ascending Trend Violation) ###
If the last structural breach G_b was a **support breach**, the current gap exerts upward pressure:

* $P_↑(t) = S(t)⋅δ(x(t))$
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
    b(t-1) & \text{if } p(t) - p(t-1) = 0 \\
    \mathbf {sgn} (p(t) - p(t-1)) & \text{if } p(t) - p(t-1) \ne 0 \\
\end{cases}
$

with $b(t) \in \{-1, 1\}$ and $b(0)$ is $\mathbf {sgn}(p(0))$. Then we define the **bar inbalance** at time $t$ ($I(t)$) as the cumulative sum of all directional signs ($b(t)$) from the beginning of the sequence.

### Bar Inbalance Momemtum ###

The **bar inbalance momentum $I_m(t)$** is defined as

$I_m(t) = δ(c(t-1)) \cdot Δ(I(t-1))$,

Where:
* $c(t-1)$ is the closing price.
* $I(t-1)$ is the Bar Inbalance (Cumulative sum of the Balance Rule $b$).
* $δ$ is the **Serial Bounded Ratio**.
* $Δ$ is the **Serial Difference**.


### Bar Inbalance Difference ###

The **bar inbalance difference $I_d(τ)$** is defined as

$I_d(τ) = \frac{I_m(τ-1) - I_m(τ-2)}{2}$

## Fractional Differentation ##

A time series has memory when future values are related to past observations. In order to perform inferential analyses, researchers need to work
with invariant processes, such as returns on prices (or changes in log-prices), changes in yield, or changes in volatility. Invariance is often achieved via data transformations
that make the series stationary, at the expense of removing all memory from the original series.

Although stationarity is a necessary property for inferential purposes, it is rarely the case in signal processing that we wish
all memory to be erased, as that memory is the basis for the model’s predictive power.

The dilemma is that _returns are stationary, however memory-less, and prices have memory, however they are non-stationary_. The question arises: _What
is the minimum amount of differentiation that makes a price series stationary while preserving as much memory as possible?_ 

### Differentiated Time Series ###

Let's consider the $X=\{x(1), x(2),..., x(t),...\}$ time series representing an stochastic prices.

$x(t) = \sum^k_{i=1}w_i x(t-i)$ where

$w_0 = 1$ and $w_i = w_{i-1} \frac {i - 1 - d} {i}$

The $\hat X$ time series will be called **differentiated time series** Let's comment on some corner cases

* **$d = 0$**: For $d=0$, all weights $w_i$ are 0 except for $w_0=1$. That is the case where the differentiated series coincides with the original one.
* **$d = 1$**: For $d=1$, all weights $w_i$ are 0 except for $w_0=1$ and $w_1=-1$. That is the standard first-order integer differentiation, which is used to derive log-price returns.
* Anywhere in between the above two cases, all weights after $w_0=1$ are negative and greater than $−1$
* **$k > d + 1$**: For $k > d + 1$, $w_i$ will be negative if $⌊d⌋$ is even,and positive otherwise.

#### Estimating $d$ ####

The `estimate_d` function estimates the fractional integration parameter  from a stochastic time series  by leveraging the relationship between **Binomial Coefficients** and **Linear Autoregression**.

##### 1. The Model

We assume the time series  is governed by a fractional process of order , which can be expressed as a linear combination of its past  values:

$x(t) = \sum^k_{i=1}w_ix(t-i)$

where the weights  are defined by the binomial recurrence relation:

$w_1 = 1$

$w_i = w_{i-1} \frac {i - 1 - d} {i}$

##### 2. Step 1: Solving the Linear System

Because the equation is linear with respect to the weights , we use **Ordinary Least Squares (OLS)** regression to find the weight vector  that minimizes the squared error:

$\min_w \|Y-Xw\|^2$

Where  contains the current values  and  is the matrix of lagged observations. This step "extracts" the memory signature from the noisy stochastic data.

##### 3. Step 2: Parameter Extraction (Non-linear Mapping)

Once the empirical weights  are found, we map them back to the single scalar . Since  is embedded non-linearly in higher-order weights, we use the **Levenberg-Marquardt algorithm** (via `curve_fit`) to find the  that best fits the entire weight profile:

$\min_d \sum^k_{i=1}(w_i - \mathbf {Binomial}(d,i))^2$

##### 4. Step 3: Auto-Differencing (Numerical Stability)

To ensure high precision, the function utilizes an **Integer-Stripping** technique.

* If the series is non-stationary ($d ≥ 1$), it calculates on the first-order difference $Δx(t)$.
* The final result is then reconstructed as $d_{\text {total}}=d_{\text {estimated}} + 1$.
* This keeps the core math in the stable $(0,1)$  range where the "memory signature" is most distinct from noise.


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

### Bar Inbalance Difference Forecast ###

We want to forecast **bar inbalance difference** $I_d(τ)$ at time $τ$.

#### Input Features ####

A sequence containing past $k$ bar inbalance differences $I_d(τ-1),..., I_d(τ-k)$.

#### Prediction Target ####

The inbalance difference $I_d(τ)$ at time $τ$.

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

### The Student-T Distribution ###

The Student-t distribution serves as a robust generalization of the standard normal distribution, characterized by heavier tails and a symmetric, bell-shaped profile..

#### Probability Density Function ####  

The Student-T distribution has the **probability density function (pdf)** given by

$f(t,ν) = \frac{Γ(\frac{ν+1}{2})}{\sqrt{πν} Γ(\frac{ν}{2})}(1+\frac{t^2}{ν})^{-(ν+1)/2}$, where 

1. $ν$ is the number of degrees of freedom and
2. $Γ$ is the gamma function,

---
Bear in mind that $t$ in the context of the Student-T pdf formula is a random variable belonging to the Student-T distribution.

#### Cumulative Distribution Function ####

The Student-T distribution has the **cumulative distribution function (cdf)** given by

$F(t,ν) = 1-\frac{1}{2} I_{x(t)} (\frac{ν}{2}, \frac{1}{2})$

where $I$ is the incomplete beta function and 

$x(t) = \frac{k}{t^2 + k}$

#### Estimating $λ$ via the Student-t Distribution ####

The Student-T distribution is a practical tool for estimating the coupling constant $λ$ required by the **cubic anharmonic equation**.

Let $\{L(x(τ-k)), \dots, L(x(τ-1))\}$ be a sample sequence of Log-Returns, which we can model (as we already know) with the Student-t distribution.

The constant $λ$ can be approximated with this formula:

$λ \approx \left| \frac{ L^2_0 f(L_0,ν) - L^2_1 f(L_1,ν) }{ L^4_1 f(L_1,ν) - L^4_0 f(L_0,ν) } \right|$, where

1. $L_0 = -σ$,
2. $L_1 =  σ$,
3. $ν = $ degrees of freedom and
4. $f(L,ν)$ = pdf for Student-T distribution.

This method is based on equating the Quantum Finance Schrödinger Equation (anharmonic oscillator) potential to the observed empirical distribution around its peak.

#### The Schrödinger Gauge #### 

We define the **Schrödinger Gauge** $Ö(t)$ at time $t$ as:

$Ö(t) =   Δ_\%(Ö↓(t),Ö↑(t))$

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

### Schrödinger Gauge Forecast ###

This model forecasts schrödinger gauge $Ö(τ)$ at time $τ$.

##### Prediction Target #####

Schrödinger gauge $Ö(τ)$ at time $τ$.

##### Input Features #####

Last $k$ schrödinger gauge differences.

| $t-1$ | $t-2$ | ... | $t-k$ |
| :--- | :--- | :--- | :--- |
| $Ö(t-1)$ | $Ö(t-2)$ | ... | $Ö(t-k)$ |

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
| $X_b(t-1)$ | Bar Inbalance Difference Forecast at time $t-1$|
| $X_g(t-1)$ | Schrödinger Gauge Difference Forecast at time $t-1$|
| $X_v(t-1)$ | Price-Volume Difference Forecast at time $t-1$|