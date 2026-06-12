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
`modules/optimalVariabRho.wVariab` (gen) / `wVariabRECO` (reco) — almacenado en árbol como `tau1_omega`/`tau2_omega` (y duplicado en `tau1_optimalVar`/`tau2_optimalVar`).

$$r = \frac{m_\tau^2}{m_\rho^2}, \quad P_2(x) = \frac{3x^2-1}{2}$$

$$w_a = \Bigl(-2 + r + 2(1+r)\,P_2(\cos\psi)\,P_2(\cos\beta)\Bigr)\cos\theta_\rho$$

$$w_b = 3\sqrt{r}\;P_2(\cos\beta)\;\sin(2\psi)\,\sin\theta_\rho$$

$$w_c = 2 + r - 2(1-r)\,P_2(\cos\psi)\,P_2(\cos\beta)$$

$$\boxed{\omega = \frac{w_a + w_b}{w_c}}$$

**Equivalencia exacta:** $W(P) \equiv \tfrac{w_c}{3}(1 + P\cdot\omega)$, por tanto los pesos con $\omega$ son algebraicamente idénticos al cociente de densidades $W(P_\text{new})/W(P_\text{sm})$ calculado con la distribución angular completa.

#### Los tres ángulos de $\omega$

| Ángulo | Significado físico | Cómo se obtiene |
|---|---|---|
| $\theta_\rho$ (con $z\equiv\cos\theta_\rho$) | **Ángulo de producción/emisión del $\rho$** en la desintegración $\tau\to\rho\,\nu$: dirección del $\rho$ en el sistema de reposo del $\tau$ respecto a la línea de vuelo del $\tau$ en el lab. Porta la información de polarización longitudinal. | **Gen:** geométrico — se boostea el $\rho$ al reposo del $\tau$ ($-\vec\beta_\tau$ del 4-vector verdadero del $\tau$) y se toma el ángulo con $\vec p_\tau^{\,\text{lab}}$. **Reco:** analítico — $\cos\theta_\rho=\dfrac{2x\,m_\tau^2-m_\tau^2-m_\rho^2}{(m_\tau^2-m_\rho^2)\sqrt{1-4m_\tau^2/s}}$, con $x=E_\rho/E_\text{beam}$ (el $\tau$ no es observable). |
| $\psi$ (con $\cos\psi$) | **Ángulo helicidad-cinemático**, fija la relación entre la fracción de energía del $\rho$ y el marco de helicidad. No requiere geometría. | Analítico en gen y reco: $\cos\psi=\dfrac{x(m_\tau^2+m_\rho^2)-2m_\rho^2}{(m_\tau^2-m_\rho^2)\sqrt{x^2-4m_\rho^2/s}}$. Gen usa $x=E_\rho/E_\tau$; reco usa $x=E_\rho/E_\text{beam}$. |
| $\beta$ (con $\cos\beta$) | **Ángulo de desintegración $\rho\to\pi^\pm\pi^0$**: dirección del pión cargado en el reposo del $\rho$ respecto a la línea de vuelo del $\rho$. Codifica la helicidad del $\rho$ (interferencia transversa-longitudinal), entra como $P_2(\cos\beta)$. | Geométrico en gen y reco — se boostea el $\pi^\pm$ al reposo del $\rho$ y se mide el ángulo con $\vec p_\rho^{\,\text{lab}}$. Gen usa 4-vectores verdaderos; reco usa 4-vectores reconstruidos. |

En el lab cada $\tau$ tiene $E_\tau\simeq E_\text{beam}$, pero **gen usa la energía/dirección verdadera del $\tau$** mientras que **reco la infiere** (fracción de energía respecto al haz + cinemática de dos cuerpos), porque el $\nu$ se pierde. `wVariabRECO` devuelve solo $(\cos\theta_\rho,\cos\psi,\cos\beta,\omega)$ — **no** calcula pesos; estos se construyen aparte en `_recompute_reco_weights` con un $\tau$ proxy colineal ($\vec p_\tau\parallel$ visible, $E=E_\text{beam}$).

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

### 3.3 Correlación — fórmula Alcaraz joint (single-decay **y** pair)
`weightsPol.newAtauJoint` vía `_compute_joint_weights` — **SIEMPRE activo** (gen y reco, ambos modos), independiente de `--compute-weights`. La fórmula joint con término cruzado es el estándar por defecto del peso correlado.

Para cualquier combinación de observables $h_1$, $h_2$:

$$\boxed{W_\text{corr} = \frac{1 + P_\text{new}(h_1 + h_2) + h_1 \cdot h_2}{1 + P_\text{sm}(h_1 + h_2) + h_1 \cdot h_2}}$$

El observable $h_i$ se obtiene en `_get_H_for_joint`:

| Tipo de tau $i$ | `use_omega=False` | `use_omega=True` |
|---|---|---|
| rho ($\text{decayID}=1$) | $H_V = \alpha_V z_R$ | $\omega$ (del árbol) — **por defecto** |
| pión/a1 ($\text{decayID}=0,10$) | $H_V = \alpha_V z_R$ | $H_V = \alpha_V z_R$ |
| leptónico ($\text{decayID}=-11,-13$) | $H_\ell$ | $H_\ell$ |

Por defecto `use_omega=True` (ω para ρ); `--no-omega-weights` conmuta el ρ a $H_V$. El alias `--omega-weights` se eliminó (era el default).

**Modo single-decay:** aunque solo se histograma un hemisferio, el otro $\tau$ del evento sí está disponible (`vars_other`), así que el corr usa la misma fórmula joint con su término cruzado $h_1 h_2$ — no el producto de pesos per-tau.

**Combinaciones cubiertas:** had-had, had-lep **y lep-lep** usan la fórmula joint general (eq. 16) con su término cruzado $h_1 h_2$. El producto $W^{(\tau_1)}\cdot W^{(\tau_2)}$ **solo** se usa como último recurso para decays no soportados (p.ej. id $-2$).

**Convención de signo:** el signo del $\tau^+$ está absorbido en la estructura `sumHH = h_1 + h_2` (no $h_1 - h_2$); ambos $h_i$ se calculan con signo positivo natural y la fórmula tiene $+h_1 h_2$ (no $-h_1 h_2$). Esta convención proviene de la corrección de Alcaraz por la helicidad del antineutrino del $\tau^+$.

Histograma ROOT: `Omega_dec0_ALL_ALL_corr_M1`.

---

## 4. Tabla resumen

| Nombre en plots | Observable $h$ | Modo | Fórmula | Histograma |
|---|---|---|---|---|
| `SM_HV` | $H_V$ | single/pair | $\frac{1+P_\text{new}H_V}{1+P_\text{sm}H_V}$ | `_ALL_ALL_M1` estándar |
| `SM_omega` | $\omega$ | single/pair | $\frac{1+P_\text{new}\omega}{1+P_\text{sm}\omega}$ | `_ALL_ALL_M1` omegaW |
| `SM_HV_corr` | $H_V$ | single/pair | $\frac{1+P_\text{new}(H_V+H_V')+H_VH_V'}{1+P_\text{sm}(H_V+H_V')+H_VH_V'}$ | `_ALL_ALL_corr_M1` estándar |
| `SM_omega_corr` | $\omega$ / $H_V$ | single/pair | $\frac{1+P_\text{new}(h+h')+h\,h'}{1+P_\text{sm}(h+h')+h\,h'}$, $h=\omega$ para ρ | `_ALL_ALL_corr_M1` omegaW |

Todos los `*_corr` usan la **fórmula joint Alcaraz** (suma + producto cruzado, §3.3) tanto en single-decay como en pair, **siempre y por defecto** (incluido lep-lep, eq. 16), independiente de `--compute-weights`; el producto independiente solo es fallback para decays no soportados.

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

## 6. Variable óptima: nivel generador vs reconstruido

La diferencia esencial entre gen y reco es que **a nivel gen se dispone del 4-vector verdadero del $\tau$** (dirección y energía), mientras que **a nivel reco el $\tau$ no es observable** (el/los $\nu$ escapan): solo se mide el sistema visible y se usa la energía del haz como referencia.

### 6.1 Definición de la variable óptima por canal

La **variable óptima reco del ρ es la $\omega$ de `wVariabRECO`** (no la fracción de energía): es la misma fórmula angular de §2 pero con el ángulo de producción $\theta_\rho$ inferido analíticamente y los 4-vectores reconstruidos. La fracción de energía $2E_\text{vis}/E_\text{beam}-1$ es un observable *adicional* más simple, no la variable óptima.

| Canal (`decayID`) | Variable óptima **GEN** | Variable óptima **RECO** |
|---|---|---|
| **ρ** (1) | $\omega$ de `wVariab(tauP4, visP4, pionP4, beamE)` — fórmula angular completa (§2), boost al reposo del $\tau$ **verdadero** | $\omega$ de `wVariabRECO(recoMesonP4, recoPionP4, beamE)` — misma fórmula angular con $\theta_\rho$ analítico y 4-vectores reco. (Adicional, más simple: $2E_\text{vis}^\text{reco}/E_\text{beam}-1$) |
| **π** (0) | $E_\pi/E_\tau$ (fracción de energía del pión cargado respecto al $\tau$ verdadero) | $2\,E_\text{vis}^\text{reco}/E_\text{beam}-1$ |
| **a1** (10) | $E_{\pi^\pm}/E_\tau$ (fracción del pión cargado líder; no hay análisis angular completo del a1) | $2\,E_\text{vis}^\text{reco}/E_\text{beam}-1$ |
| **leptónico** ($-11,-13$) | $E_\ell/E_\tau$ (fracción de energía del leptón respecto al $\tau$ verdadero) | $2\,E_\text{vis}^\text{reco}/E_\text{beam}-1$ ($=x_\ell$ escalado). Contiene la misma información que la variable óptima leptónica $H_\ell(x_\ell)$ (§2): ambos dependen solo de $x_\ell=E_\ell/E_\text{beam}$, la única observable (los dos $\nu$ no se reconstruyen) |

`genOnlyRHOTree_MDecs_parallel.py` líneas 210–246 (gen); `RhoHistFromTree_MDecs_parallel.py` líneas 742 / 828 (`_optimal_x` reco). Para el ρ, `wVariabRECO` se invoca en `_recompute_reco_weights`/`_get_H_for_joint_reco`.

### 6.2 Diferencias clave gen → reco

1. **Denominador de la fracción de energía:** gen usa $E_\tau$ (energía verdadera del $\tau$); reco usa $E_\text{beam}$ (la del $\tau$ no es observable). Como $E_\tau\simeq E_\text{beam}$ en el pico de la Z, la aproximación es buena pero introduce dispersión por la radiación/cinemática del evento.
2. **Ángulo de producción $\theta_\rho$ (solo ρ):** gen lo obtiene **geométricamente** (boost del $\rho$ al reposo del $\tau$ verdadero); reco lo **infiere analíticamente** de $x=E_\rho/E_\text{beam}$ y la cinemática de dos cuerpos (sin dirección del $\tau$). Ver tabla de ángulos en §2.
3. **Dirección del $\tau$ para $P(z)$:** gen usa $\theta_\tau$ verdadero; reco usa un **proxy colineal** ($\tau\parallel$ visible, $E=E_\text{beam}$) en `_recompute_reco_weights`.

### 6.3 Contraste con el repo original (`original_repos/ExamplesFCCFullSim/analysisRHOTree.py`)

El análisis original (rho-céntrico, un canal por run vía `--decay`) **sí usa la $\omega$ de `wVariabRECO` como variable óptima reco** del ρ y la histograma directamente:

| Observable reco | Original | Línea |
|---|---|---|
| $\omega$ reco | `hOmega.Fill(w)`, con `(cos_theta,cos_psi,cos_beta,w)=wVariabRECO(recoMesonP4,recoPionP4,beamE)` | L428, L531 |
| $\cos\theta_\rho$, $\cos\psi$ reco | de `wVariabRECO` (analíticos) | L428, L494-495 |
| fracción $X$ | `RecoMeson_X.Fill(2·E_\text{meson}/E_\text{beam}-1)` (observable extra, más simple) | L510 |
| $E/E_\text{beam}$, $\cos\theta_\text{lab}$ | `RecoMesonEOverBeamE`, `RecoMesonCosTheta` | L491-492 |

**Pesos en los histogramas reco del original:** se aplican los pesos **gen** (`weight_P1/M1` vía `newAtauRHO` con cinemática gen, L524-525). Es decir, el original usa $\omega$ reco como **observable** y pesos **gen** — no recalcula pesos reco. Los bloques `genTauID==-13/-11/0/1/10` (L601-666) son clasificación de **fondo** del otro $\tau$, no variables óptimas distintas.

**Alineación del MDecs (junio 2026):** el histograma `Omega_Reco` del MDecs ahora rellena con la $\omega$ de `wVariabRECO`, igual que el original. La regla es `("Reco","Omega_Reco", v.get("_omega_reco", -999.0))`; el valor `_omega_reco` lo calcula el helper `_compute_omega_reco(tau_vars, beamE)` (solo para ρ reco, `recoTauID==1`; $-999$ en otros canales) vía `wVariabRECO(recoVisP4, recoPionP4, beamE)`, y se asigna en el bucle de eventos junto a `_optimal_x`. Antes rellenaba con el $\omega$ **gen** (`v["omega"]`); en producción `--only-gen` ambos coinciden (reco = espejo de gen), pero con reco real ya queda la verdadera variable óptima reconstruida.

---

## 7. Selección reco del canal ρ+leptón y cierre vs análisis legacy (junio 2026)

Pipeline reco real (full-sim, **no** `--only-gen`) para medir $A_\tau$/$A_e$/$\sin^2\theta_\text{eff}$ en el canal ρ+leptón, reproduciendo el primer análisis (`original_repos/.../analysisRHOTree.py`, fit legacy ene-2026 en `TauPolOutputs/Binned_histograms_full_lumi/`).

### 7.1 Cadena de selección

**Árbol** (`analysisRHOTree_MDecs_parallel.py`, muestras `ztt_plus` + `bhabha`, `--decay-modes 0 1 2 -11 -13`):
- Reconstrucción `extractTauDecays` (`TauDecays`+`NeutralRecover`): `TauPhotonPCut=0.35`, `NeutronCut=3`, `dRMax=0.4`, matching gen-reco `MatchedGenMinDR=1` (métrica θ-φ `dRAngle`).
- **Corte XOR de leptón** `--lepton-xor-p 10`: exactamente 1 tipo de leptón (e **XOR** µ) con $P>10$ GeV. *Difiere del legacy, que usaba **OR** (≥1 leptón $P>10$, L380); el XOR limpia más fondo (rechaza Bhabha de 2e).*

**Histogramas** (`RhoHistFromTree_MDecs_parallel.py`, pares `2 -11` y `2 -13`; dec0=ρ reco id 2, dec1=leptón):

| Corte | Flag | Variable | Rango | Legacy |
|---|---|---|---|---|
| mesón (ρ) | `--meson-cut` | `recoVisP` dec0 | $[2,\,45.57]$ GeV | sí |
| leptón | `--lepton-cut` | `recoVisP` dec1 | $[10,\,41.16]$ GeV | sí (OR) |
| separación | `--ang` | $dR=\sqrt{\Delta\theta^2+\Delta\phi^2}$ | $[3.06,\,5.0]$ | sí |
| aceptancia | `--cos-acceptance 0.95` | $\lvert\cos\theta_\rho^\text{vis}\rvert$ | $\le 0.95$ | sí (L378) |
| bordes ω | `--omega-border-cut` | $\cos\theta$/$\cos\psi$ reco (`wVariabRECO`) | descarta si clampea a $\pm1$ | sí (L437-438) |

- **Repesado**: $\omega$ por defecto (`newAtauRhoOmega`); el peso **gen** del ρ que usa el fit ya es $\omega$ completa (`wVariab`, no la $H_V$ simplificada). Flag `--no-omega-weights` para usar $H_V=\alpha_V z_R$.
- **Señal/fondo por verdad gen** (`decayID`): `SIGNAL_SIGNAL` = ρ reco ∧ ρ gen ambos lados; migración desglosada por tipo gen.
- *Veto legacy de µ@π/2 (θ∈(1.565,1.575), L382) **omitido**: obsoleto en MDecs (cero gen-muones fake en esa ventana; la reco nueva no tiene esa patología).*

**Binning + fit**: `makeCosBins_MDecs.py --bg-def rho_only` (señal = `SIGNAL_SIGNAL + SIGNAL_BG`, combina ρ+e ⊕ ρ+µ, escala $\text{lumi}\cdot\sigma/N_\text{gen}$, Bhabha como fondo externo nominal) → `fitPolAssym.py --bg-mode total --rebin 2`.

### 7.2 Porcentajes por configuración (modo `per-tau`, `rho_only`)

| Configuración | Señal | Migr. | Bhabha | Total | %S / %M / %B | Pureza Ztt* |
|---|---|---|---|---|---|---|
| **Legacy** (objetivo) | 581 567 | 36 517 | 27 058 | 645 142 | 90.15 / 5.66 / 4.19 | **94.09** |
| XOR (sin cosAcc/border) | 626 080 | 41 643 | 28 791 | 696 514 | 89.89 / 5.98 / 4.13 | 93.76 |
| XOR + cosAcc 0.95 | 602 758 | 38 577 | 21 409 | 662 744 | 90.95 / 5.82 / 3.23 | 93.98 |
| **XOR + cosAcc + bordes ω** | 580 066 | 35 372 | 18 456 | 633 894 | 91.51 / 5.58 / 2.91 | **94.25** |

\* Pureza Ztt = Señal / (Señal+Migración), excluyendo Bhabha.

### 7.3 Asimetrías (cierre físico)

Todas las configuraciones recuperan los valores de generación ($A=0.14955$, $\sin^2\theta_\text{eff}=0.2312$):

| Configuración | $A_\tau$ | $A_e$ | $\sin^2\theta_\text{eff}$ |
|---|---|---|---|
| Legacy (ene-2026) | $0.149544 \pm 0.000099$ | $0.149402 \pm 0.000144$ | $0.231219 \pm 0.000018$ |
| XOR | $0.149588 \pm 0.003282$ | $0.149390 \pm 0.004227$ | $0.231221 \pm 0.000537$ |
| XOR + cosAcc | $0.149590 \pm 0.003331$ | $0.149373 \pm 0.004330$ | $0.231223 \pm 0.000551$ |
| XOR + cosAcc + bordes ω | $0.149583 \pm 0.003227$ | $0.149371 \pm 0.004217$ | $0.231223 \pm 0.000536$ |

### 7.4 Conclusiones del cierre

1. **Señal/migración cierra.** Con los cortes cinemáticos del legacy (aceptancia $\lvert\cos\theta_\rho\rvert<0.95$ + bordes ω), la pureza interna Ztt pasa de 93.76% → 94.25%, **igualando/rebasando** el 94.09% legacy. La aceptancia recorta migración forward; los bordes ω quitan ρ de cinemática no física (≈12% migración).
2. **La física cierra** en todas las versiones (independiente de los cortes).
3. **Residuo: solo el Bhabha** (2.91% vs 4.19% legacy). Causa principal: el **XOR** (más estricto que el OR legacy) + reco más limpia (menos e→ρ falsos). El $N_\text{gen}$ **no** influye (el conteo escalado $\propto N_\text{raw}/N_\text{gen}$ y $N_\text{raw}\propto N_\text{gen}$ se cancelan). Reproducir el Bhabha legacy exigiría regenerar el árbol con OR en vez de XOR (no hecho; un Bhabha más bajo es un fondo más limpio).

Salidas: `Binned_histograms_MDecs/{cuts_xor, cuts_omega, cuts_omega_border}/`.
