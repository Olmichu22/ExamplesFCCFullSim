# Bug: signo de `gv_ga` en el cálculo de la polarización del τ

## Resumen

El código calcula `gv_ga = -1 + 4·sin²θ_W` cuando la expresión correcta es
`1 - 4·sin²θ_W`. Esto invierte el signo de A_e y A_τ, dando una polarización
del SM positiva (~+0.15) cuando la predicción conocida es negativa (~−0.15).
Como consecuencia, los pesos de reweighting son sistemáticamente demasiado
pequeños en z > 0, produciendo un undershoot al comparar SM×weight con las
muestras generadas a polarización fija.

El error está presente en el repositorio original y en la versión actual, en
cuatro lugares: `weightsPol.py` (líneas 16, 66, 168) y `optimalVariabRho.py`
(línea 68).

---

## 1. Acoplamientos del Z a fermiones en el SM

El vértice Z-fermión en el Modelo Estándar es:

$$\mathcal{L} \supset \frac{g}{2\cos\theta_W}\, Z_\mu\, \bar{f}\gamma^\mu(g_V^f - g_A^f\gamma^5)f$$

Para leptones cargados (T₃ = −½, Q = −1):

$$g_V^f = T_3 - 2Q\sin^2\theta_W = -\tfrac{1}{2} + 2\sin^2\theta_W \approx -0.038$$

$$g_A^f = T_3 = -\tfrac{1}{2}$$

El parámetro de asimetría se define como:

$$A_f = \frac{2\,g_V^f\,g_A^f}{(g_V^f)^2+(g_A^f)^2} = \frac{2\,(g_V/g_A)}{1+(g_V/g_A)^2}$$

El cociente de acoplamientos para leptones cargados es:

$$\frac{g_V}{g_A} = \frac{-\frac{1}{2}+2\sin^2\theta_W}{-\frac{1}{2}} = 1 - 4\sin^2\theta_W \approx +0.075$$

Con sin²θ_W = 0.2312: **A_τ = A_e ≈ +0.1465** (positivo). Este es el valor
medido en LEP.

---

## 2. Polarización del τ en Z → τ⁺τ⁻

La sección eficaz diferencial de e⁺e⁻ → Z → τ⁺τ⁻ lleva a una polarización
media del τ⁻ dependiente del ángulo de producción θ (ángulo del τ⁻ respecto
al haz de e⁻):

$$P(\tau^-,\cos\theta) = -\frac{A_\tau(1+\cos^2\theta) + 2A_e\cos\theta}{1+\cos^2\theta + 2A_e A_\tau\cos\theta}$$

Para haces no polarizados y producción central (cos θ = 0):

$$P = -A_\tau \approx -0.147 \quad \Rightarrow \quad \tau^- \text{ ligeramente left-handed}$$

Valor medido en LEP (combinación ALEPH/DELPHI/L3/OPAL):

$$P_\tau = -0.1439 \pm 0.0043$$

---

## 3. Observable óptimo en el canal ρ

Para τ⁻ → ρ⁻ν_τ → π⁻π⁰ν_τ, la distribución angular completa del ρ en el
frame del τ (Kühn & Santamaria 1990, implementada en TAUOLA) es:

$$\frac{1}{\Gamma}\frac{d\Gamma}{dz\,d\cos\psi\,d\cos\beta} \propto
\frac{2}{3}\Big[(1-Pz) - r(1+Pz)\Big] + r(1+Pz)
+ \text{términos en } P_2(\cos\psi)\cdot P_2(\cos\beta)$$

donde:
- $z = \cos\theta_\rho$: ángulo del ρ en el frame del τ respecto a la dirección del τ en el laboratorio
- $r = m_\tau^2/m_\rho^2$
- $\psi$: ángulo de helicidad del ρ (derivado de la fracción de energía x = E_ρ/E_τ)
- $\beta$: ángulo del π± en el frame del ρ respecto a la dirección del ρ

El peso de reweighting de una muestra con polarización $P_\text{SM}$ a
polarización $P_\text{new}$ es la razón de estas distribuciones:

$$w = \frac{f(z,\psi,\beta\,|\,P_\text{new})}{f(z,\psi,\beta\,|\,P_\text{SM})}$$

La polarización del SM $P_\text{SM}(\cos\theta_\tau)$ aparece en el
**denominador**.

---

## 4. El error

### Lo que hace el código

```python
gv_ga  = -1 + 4 * sin2theta_effective   # = -(1 - 4sin²θ) = -0.075
Ae_sm  = 2 * gv_ga / (1 + gv_ga**2)     # ≈ -0.1495  (signo incorrecto)
Atau_sm = Ae_sm
```

### Lo correcto

```python
gv_ga  = 1 - 4 * sin2theta_effective    # = +0.075
Ae_sm  = 2 * gv_ga / (1 + gv_ga**2)    # ≈ +0.1495  (correcto)
Atau_sm = Ae_sm
```

### Consecuencias numéricas

| Cantidad | Código (incorrecto) | Correcto |
|---|---|---|
| gv_ga | −0.0752 | +0.0752 |
| A_e = A_τ | −0.1495 | +0.1495 |
| P_SM (cos θ = 0) | **+0.150** | **−0.150** |
| weight_M1 (z = +0.9) | 1.332 | 1.507 |

Con el signo equivocado, el denominador del peso es **mayor** de lo que debería
en z > 0 (porque P_SM > 0 infla el término 1 + α·P_SM·z), reduciendo el
peso por debajo de su valor correcto. El resultado observable es un
**undershoot sistemático** de SM×weight en z > 0 al comparar con la muestra P1,
y un correspondiente **overshoot** en z < 0, confirmado experimentalmente en
los plots de debug (stages 1–5).

### Por qué explica también el intercambio de nombres

Con el signo equivocado, el parámetro `New_Atau = −1` produce
$P_\text{new} \approx +1$ y por tanto `weight_M1` empuja la distribución
hacia polarización positiva. Esto es lo contrario de lo que su nombre sugiere,
y coincide con la observación de que "`weight_M1` es el que reproduce la
muestra P1".

---

## 5. Origen probable del error

La variable `gv_ga` fue probablemente calculada como $2g_V$ (proporcional al
acoplamiento vectorial) sin dividir por $g_A$:

$$2g_V = 2\Big(-\tfrac{1}{2}+2\sin^2\theta_W\Big) = -1 + 4\sin^2\theta_W$$

Esta expresión es correcta para $2g_V$, pero se usó como si fuera $g_V/g_A$.
Como $g_A = -\frac{1}{2}$, el cociente real es:

$$\frac{g_V}{g_A} = -2g_V = 1 - 4\sin^2\theta_W$$

El signo negativo de $g_A$ no se tuvo en cuenta al hacer la división, un error
de bookkeeping habitual al derivar estos cocientes a mano. El error pasó
desapercibido durante tiempo porque $|A_\tau| \approx 0.15$ es pequeño y su
efecto sobre las distribuciones solo se hace evidente al comparar directamente
con muestras generadas a polarización fija.

---

## 6. Localización en el código

| Archivo | Línea | Contexto |
|---|---|---|
| `modules/weightsPol.py` | 16 | función `newAtau` (π, a₁, leptónico) |
| `modules/weightsPol.py` | 66 | función `newAtauRHO` (canal ρ completo) |
| `modules/weightsPol.py` | 168 | función `newAtauLep` (canal leptónico) |
| `modules/optimalVariabRho.py` | 68 | función `wVariab` (variable óptima ω y pesos gen) |

El error está presente tanto en el repositorio original
(`original_repos/ExamplesFCCFullSim`) como en la versión actual.

---

## 7. Referencias

| Contenido | Referencia |
|---|---|
| Acoplamientos Z y definición de A_f | PDG *Review of Particle Physics* (2024), cap. "Electroweak Model and Constraints on New Physics" |
| Medida de P_τ en LEP | ALEPH, DELPHI, L3, OPAL: *Phys. Rept.* **532** (2013) 119 |
| Fórmula de polarización τ del Z | Y.S. Tsai, *Phys. Rev.* **D4** (1971) 2821 |
| Distribución angular canal ρ | J.H. Kühn & A. Santamaria, *Z. Phys.* **C48** (1990) 445 |
| Implementación TAUOLA | S. Jadach, J.H. Kühn, Z. Was, *Comput. Phys. Commun.* **64** (1990) 275 |
| Desintegraciones τ en Pythia 8 | P. Ilten, *arXiv:1211.6730* (2012) |
| Medida reciente P_τ (CMS) | CMS Collaboration, *arXiv:2309.12408* (2023) |
