# Repesado de polarización tau — fórmulas implementadas

Documento de verificación: cada fórmula corresponde 1:1 al código en
`modules/weightsPol.py` y `RhoAnalysis/RhoHistFromTree_MDecs_parallel.py`.

Referencia teórica: J. Alcaraz, *Reweight_tautau.pdf*, mayo 2026.

---

## 1. Elementos comunes a todas las fórmulas

### Asimetría del electrón ($A_e$)
`weightsPol._compute_ae_sm`

$$A_e = \frac{2(g_V/g_A)}{1+(g_V/g_A)^2}, \qquad \frac{g_V}{g_A} = 1 - 4\sin^2\theta_\text{eff}$$

### Polarización del tau ($P$)
`weightsPol._compute_Ptau(costheta, Atau, Ae)`

$$P(\cos\theta_\tau;\, A_\tau) = -\frac{A_\tau(1+\cos^2\theta_\tau) + 2A_e\cos\theta_\tau}{1+\cos^2\theta_\tau + 2A_eA_\tau\cos\theta_\tau}$$

En todos los pesos:
- $P_\text{sm} = P(\cos\theta_\tau;\; A_\tau = A_e)$
- $P_\text{new} = P(\cos\theta_\tau;\; A_\tau = A_{\tau,\text{new}})$, con $A_{\tau,\text{new}} = +1$ (P1) ó $-1$ (M1)

---

## 2. Observables de spin ($H$)

### $H_V$ — canal rho/a1
`weightsPol._compute_H(meson_p4, tau_p4, decay_type=1)`

$$\xi = \frac{m_V^2}{m_\tau^2}, \quad x = \frac{E_V}{E_\tau}, \quad z_R = \frac{2x - 1 - \xi}{1 - \xi}$$

$$\alpha_V = \frac{m_\tau^2 - 2m_V^2}{m_\tau^2 + 2m_V^2}, \quad H_V = \alpha_V(m_\rho^\text{evento}) \cdot z_R$$

Para pión ($\text{type}=0$): $H_\pi = z_R$ (sin $\alpha_V$).

### $H_\ell$ — canal leptónico
`weightsPol._compute_H_lep(lep_p4, beam_E)`

$$H_\ell = \frac{1 + x_\ell - 8x_\ell^2}{5 + 5x_\ell - 4x_\ell^2}, \qquad x_\ell = \frac{E_\ell}{E_\text{beam}}$$

### $\omega$ — variable óptima del canal rho
`modules/optimalVariabRho.wVariab` — almacenado en árbol como `tau_omega`

$$r = \frac{m_\tau^2}{m_\rho^2}, \quad P_2(x) = \frac{3x^2-1}{2}$$

$$w_a = \Bigl(-2 + r + 2(1+r)\,P_2(\cos\psi)\,P_2(\cos\beta)\Bigr)\cos\theta_\rho$$

$$w_b = 3\sqrt{r}\;P_2(\cos\beta)\;\sin(2\psi)\,\sin\theta_\rho$$

$$w_c = 2 + r - 2(1-r)\,P_2(\cos\psi)\,P_2(\cos\beta)$$

$$\boxed{\omega = \frac{w_a + w_b}{w_c}}$$

**Equivalencia exacta:** $W(P) \equiv \tfrac{w_c}{3}(1 + P\cdot\omega)$, por tanto los pesos con $\omega$ son algebraicamente idénticos al cociente de densidades $W(P_\text{new})/W(P_\text{sm})$ calculado con la distribución angular completa.

---

## 3. Estrategias de repesado — fórmulas exactas

### 3.1 Per-tau con $H_V$ — `SM_HV`
`weightsPol.newAtau` — activado con `use_omega=False` (default)

$$\boxed{W_{H_V} = \frac{1 + P_\text{new} \cdot H_V}{1 + P_\text{sm} \cdot H_V}}$$

Histograma ROOT: `Omega_dec0_ALL_ALL_M1` (fichero estándar).

---

### 3.2 Per-tau con $\omega$ — `SM_omega`
`weightsPol.newAtauRhoOmega` — activado con `--omega-weights` (`use_omega=True`)

$$\boxed{W_\omega = \frac{1 + P_\text{new} \cdot \omega}{1 + P_\text{sm} \cdot \omega}}$$

Histograma ROOT: `Omega_dec0_ALL_ALL_M1` (fichero omegaW).

---

### 3.3 Correlación en modo **single-decay**
`process_tree_range_mdecs`, líneas 660–664

En modo single-decay, **no se llama a `_compute_joint_weights`**. El peso corr es el producto de los pesos per-tau de ambos hemisferios:

$$\boxed{W_\text{corr}^\text{single} = W^{(\tau_1)} \cdot W^{(\tau_2)}}$$

donde cada $W^{(i)}$ es $W_{H_V}$ o $W_\omega$ según el flag `use_omega` y el tipo del tau.

- `SM_HV_corr`: $W^{(1)} = W_{H_V}(\tau_1)$, $W^{(2)} = W_{H_V}(\tau_2)$
- `SM_omega_corr`: $W^{(1)} = W_\omega(\tau_1^\rho)$, $W^{(2)} = W_X(\tau_2)$ con $X=\omega$ si $\tau_2$ es rho, $X=H_V$ o $H_\ell$ si no

Histograma ROOT: `Omega_dec0_ALL_ALL_corr_M1`.

---

### 3.4 Correlación en modo **pair-decay** — fórmula Alcaraz joint
`weightsPol.newAtauJoint` — activado en modo pair con `compute_weights=True`

Para cualquier combinación de observables $h_1$, $h_2$:

$$\boxed{W_\text{joint} = \frac{1 + P_\text{new}(h_1 + h_2) + h_1 \cdot h_2}{1 + P_\text{sm}(h_1 + h_2) + h_1 \cdot h_2}}$$

El observable $h_i$ se obtiene en `_get_H_for_joint`:

| Tipo de tau $i$ | `use_omega=False` | `use_omega=True` |
|---|---|---|
| rho ($\text{decayID}=1$) | $H_V = \alpha_V z_R$ | $\omega$ (del árbol) |
| pión/a1 ($\text{decayID}=0,10$) | $H_V = \alpha_V z_R$ | $H_V = \alpha_V z_R$ |
| leptónico ($\text{decayID}=-11,-13$) | $H_\ell$ | $H_\ell$ |

**Convención de signo:** el signo del $\tau^+$ está absorbido en la estructura `sumHH = h_1 + h_2` (no $h_1 - h_2$); ambos $h_i$ se calculan con signo positivo natural y la fórmula tiene $+h_1 h_2$ (no $-h_1 h_2$). Esta convención proviene de la corrección de Alcaraz por la helicidad del antineutrino del $\tau^+$.

Histograma ROOT: `Omega_dec0_ALL_ALL_corr_M1` (solo disponible en modo pair).

---

## 4. Tabla resumen

| Nombre en plots | Observable $h$ | Modo | Fórmula | Histograma |
|---|---|---|---|---|
| `SM_HV` | $H_V$ | single | $\frac{1+P_\text{new}H_V}{1+P_\text{sm}H_V}$ | `_ALL_ALL_M1` estándar |
| `SM_omega` | $\omega$ | single | $\frac{1+P_\text{new}\omega}{1+P_\text{sm}\omega}$ | `_ALL_ALL_M1` omegaW |
| `SM_HV_corr` | $H_V$ | single | $W_{H_V}^{(1)}\cdot W_{H_V}^{(2)}$ | `_ALL_ALL_corr_M1` estándar |
| `SM_omega_corr` | $\omega$ / $H_V$ | single | $W_\omega^{(1)}\cdot W_X^{(2)}$ | `_ALL_ALL_corr_M1` omegaW |
| pair+H_V joint | $H_V$ | pair | $\frac{1+P(H_V+H_V')+H_VH_V'}{1+P_\text{sm}(\ldots)}$ | `_ALL_ALL_corr_M1` par |
| pair+$\omega$ joint | $\omega$ | pair | $\frac{1+P(\omega+\omega')+\omega\omega'}{1+P_\text{sm}(\ldots)}$ | `_ALL_ALL_corr_M1` par+omega |

---

## 5. Funciones relevantes en `weightsPol.py`

| Función | Rol |
|---|---|
| `newAtau(TauP4, MesonP4, Type, New_Atau)` | Per-tau hadrónico con $H_V$ |
| `newAtauLep(lepP4, lepTauP4, beamE, New_Atau)` | Per-tau leptónico con $H_\ell$ |
| `newAtauRhoOmega(TauP4, omega, New_Atau)` | Per-tau rho con $\omega$ |
| `newAtauJoint(TauP4, H, Hp, New_Atau)` | Joint genérico con $H$, $H'$ pre-calculados |
| `newAtauJoint_had_had(...)` | Joint had+had via `_compute_H` internamente |
| `newAtauJoint_had_lep(...)` | Joint had+lep via `_compute_H`/`_compute_H_lep` internamente |

---

## 6. Verificación de cierre y limitaciones conocidas (junio 2026)

Closure tests (chi²/ndf bin a bin sobre la distribución de ω, muestra SM repesada vs verdad P1/M1, canal ρ). Convención: peso `_M1` (New_Atau=−1) imita P1.

### 6.1 ω no es un observable óptimo perfecto — potencia analizadora ≈ 0.92

Si ω fuera el observable óptimo ideal y la muestra está totalmente polarizada (A_τ=±1 ⟹ P=∓1), la asimetría debería cumplir **A(ω) ≡ ω** (pendiente 1). Medido sobre las muestras P1/M1:

$$A(\omega) = \frac{f_{P1}(\omega) - f_{M1}(\omega)}{f_{P1}(\omega) + f_{M1}(\omega)} = 0.924\,\omega - 0.001$$

→ **ω tiene potencia analizadora del 92%** (dilución del 8%). Prueba directa: `f_P1(ω)` no se anula en ω→−1 (ni `f_M1` en ω→+1), cuando debería si ω fuera óptimo perfecto.

**Origen del 8%:** localizado en el test 6.1.1 — `wVariab` es óptima cerca del polo del ρ (a≈1.0) pero pierde poder analizador en la cola alta de masa. No es un efecto del límite mτ/Eτ→0 (que sería plano en masa) ni de reconstrucción (todo es gen).

**Síntoma observable en los plots:** el peso `(1+P·ω)/(1+P_sm·ω)` asume implícitamente potencia analizadora 1, por lo que es ~8% demasiado empinado en ω y sobre-corrige las colas:
- Plot **P1** (peso ∝ 1+ω): vacía de más la cola ω<0 → región desplazada hacia la derecha.
- Plot **M1** (peso ∝ 1−ω): vacía de más la cola ω>0 → región desplazada hacia la izquierda.

Consecuencia: la media del repesado per-tau **sobreestima ~9%** (P1: +0.256 vs verdad +0.232). Cuadra con la dilución del 8%.

**Estado:** no se corrige por ahora. Posible calibración futura: usar la potencia analizadora medida `a` en el peso, `W = (1+P_new·a·ω)/(1+P_sm·a·ω)`. **Pero ver test 6.1.1: `a` NO es global.**

#### 6.1.1 Test: el poder analizador depende de la masa del ρ (no es global)

Todos los histogramas `Omega_dec0` son **gen** (4-vectores exactos), así que ni la reconstrucción del π⁰ ni la resolución de detector intervienen. Para localizar el origen del 8% se midió la pendiente `a` de A(ω)=a·ω en bins de la masa invariante del ρ (`visM`):

| masa ρ (GeV) | poder analizador `a` |
|---|---|
| 0.40–0.60 | 1.000 |
| 0.60–0.70 | 0.991 |
| 0.70–0.80 (polo) | 0.996 |
| 0.80–0.90 | 0.985 |
| 0.90–1.05 | 0.943 |
| 1.05–1.50 | 0.658 |

**Conclusión:** cerca del polo del ρ, `a`≈1.0 — **ω es prácticamente óptima**. El poder se desploma en la cola alta de masa (a=0.66 por encima de 1.05 GeV). Como `a` depende fuertemente de mρ y **no** es plano, se **descarta el límite mτ/Eτ→0** como causa (sería plano en masa): el efecto está en cómo `wVariab` trata el ρ lejos del polo, donde la estructura de espín de la resonancia deja de describir bien el decay (masa media observada 0.810 GeV, σ=0.176, sesgada hacia arriba).

**Implicaciones:**
1. Una calibración con un solo factor global `a=0.92` **no es correcta** — habría que usar `a(mρ)`.
2. **Alternativa más simple: un corte de masa del ρ** alrededor del polo (p.ej. 0.65–0.85 GeV) recupera `a`≈0.99 sin calibrar nada. Precio: ~la mitad de estadística (la cola). La decisión es estadística (resolución vs sesgo).

Pendiente para confirmar (test 2, futuro): comparar el poder analizador de `wVariab` (algebraica) vs `wVariabDanvier` (covariante, $H^\mu=2(q\cdot N)q^\mu-q^2N^\mu$) para separar "defecto de fórmula" de "efecto físico irreducible".

### 6.2 La correlación: el producto sobre-corrige, la fórmula joint cierra mejor

Cierre sobre ω de un hemisferio en eventos ρ+ρ (target = verdad P1, media +0.234):

| Estrategia | Media | chi²/ndf |
|---|---|---|
| SM nominal | −0.021 | 0.00694 |
| per-tau (ω solo) | +0.256 | 0.00018 |
| producto W⁽¹⁾·W⁽²⁾ | +0.306 | 0.00068 ← peor |
| **joint Alcaraz** (1+P(ω+ω')+ωω') | **+0.244** | **0.00013** ← mejor |

- **El producto de pesos per-tau es incorrecto:** sobreestima la media a +0.306 (+31% sobre el shift real). Ambos τ comparten el **mismo ángulo de producción** → la misma P(z); el producto aplica la corrección de polarización dos veces de forma correlacionada.
- **La fórmula joint de Alcaraz cierra mejor incluso que el per-tau** (+0.244, chi² 0.00013). El término cruzado ω·ω' es la correlación de espín entre hemisferios y, al estar correctamente incluido, **cancela parte del overshoot** de la dilución del 8%.

**Correcciones aplicadas (junio 2026):**
1. En modo single-decay el `corr` usaba el producto (fallback). Como el otro hemisferio sí está disponible (`vars_other`), ahora usa `_compute_joint_weights` (fórmula joint de Alcaraz), igual que el modo pair.
2. `_compute_reco_joint_weights` (corr a nivel reco) no aceptaba `use_omega` y siempre usaba $H_V$. Ahora es omega-aware vía `_get_H_for_joint_reco` (ω reco vía `wVariabRECO` para ρ), consistente con `_recompute_reco_weights` per-tau.

### 6.3 Variable óptima en el canal leptónico

La formulación general del PDF (sección 3, eq. 15-16) es:

$$\frac{1}{N}\frac{dN}{d\omega} = f(\omega)(1+P\omega), \qquad W = \frac{1 + P'(\omega + \omega') + \omega\omega'}{1 + P(\omega + \omega') + \omega\omega'}$$

con ω, ω' las variables óptimas de cada pata. **Para el τ leptónico la variable óptima es $\omega_\ell = H_\ell(x_\ell)$**: la única información observable es $x_\ell = E_\ell/E_\text{beam}$ (los dos neutrinos no se reconstruyen), y la distribución (eq. 10) es $f(x_\ell)(1+P\cdot H_\ell)$, que comparada con eq. 15 identifica $H_\ell$ como la variable óptima. No existe una variable óptima "mejor" para el leptón análoga a la ω del ρ (que sí gana con cosβ, cosψ).

**Consecuencia:** el flag `--omega-weights` solo afecta al ρ. El leptón usa $H_\ell$ siempre (con y sin el flag), y eso es lo correcto: $H_\ell$ ya es su variable óptima. No hay cambio que aplicar para leptones.
