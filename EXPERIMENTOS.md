# Registro de Experimentos — FARMSAR Sevilla 2017

## EXP-001 — Línea base backscatter
**Fecha:** 2024-11-18
**Estado:** ✅ Completado — 7/7 niveles
**Objetivo:** Reproducir y comparar con Di Martino et al. 2023

### Configuración
- Sensor: Sentinel-1 Sevilla 2017 (BXII)
- Features: VV + VH backscatter normalizado
- Clases activas: 16 (Rice excluido en split)
- Split: 50/50 por parcelas sin solape train/test
- Parcelas fiables: intersección CAE ∩ RF sobre TEST (umbral 75%)
- CAE: Conv1D bottleneck=1, 10 pasadas FARMSAR
- RF: 500 árboles, balanced_subsample, 4-fold CV por parcela
- Datasets infectados: 7 niveles × 10 réplicas = 70 experimentos

### Diferencias respecto al paper
1. Parcelas fiables extraídas sobre TEST (sin circularidad)
2. No se implementa SVM
3. Variante propia: Margen Relativo MSE (base=0.20, 2do=0.30)

### Resultados completos

#### CAE
| Ruido | Sin umb rec | Sin umb prec | Sin umb FP | Otsu rec | Otsu prec | Otsu FP | Otsu noDet | Margen rec | Margen prec | Margen FP | Margen noDet |
|-------|-------------|--------------|------------|----------|-----------|---------|------------|------------|-------------|-----------|--------------|
| 1%    | 0.91        | 0.42         | 124        | 0.80     | 0.95      | 4       | 20         | 0.91       | 0.59        | 63        | 7            |
| 5%    | 0.89        | 0.71         | 199        | 0.80     | 0.91      | 38      | 106        | 0.88       | 0.84        | 82        | 52           |
| 10%   | 0.92        | 0.84         | 189        | 0.80     | 0.95      | 39      | 225        | 0.90       | 0.91        | 82        | 100          |
| 15%   | 0.89        | 0.80         | 329        | 0.78     | 0.95      | 41      | 355        | 0.88       | 0.88        | 161       | 154          |
| 20%   | 0.90        | 0.86         | 273        | 0.78     | 0.98      | 24      | 514        | 0.89       | 0.92        | 125       | 205          |
| 25%   | 0.89        | 0.87         | 283        | 0.75     | 0.97      | 28      | 711        | 0.87       | 0.92        | 122       | 303          |
| 30%   | 0.87        | 0.86         | 352        | 0.73     | 0.95      | 50      | 873        | 0.85       | 0.91        | 159       | 379          |

#### RF
| Ruido | Sin umb rec | Sin umb prec | Sin umb FP | Otsu rec | Otsu prec | Otsu FP | Otsu noDet | Margen rec | Margen prec | Margen FP | Margen noDet |
|-------|-------------|--------------|------------|----------|-----------|---------|------------|------------|-------------|-----------|--------------|
| 1%    | 0.98        | 0.67         | 49         | 0.95     | 0.98      | 2       | 5          | 0.98       | 0.68        | 47        | 2            |
| 5%    | 0.98        | 0.91         | 56         | 0.95     | 0.99      | 6       | 29         | 0.97       | 0.91        | 55        | 15           |
| 10%   | 0.96        | 0.94         | 74         | 0.95     | 0.98      | 17      | 66         | 0.96       | 0.94        | 72        | 45           |
| 15%   | 0.96        | 0.94         | 103        | 0.95     | 0.98      | 27      | 78         | 0.96       | 0.94        | 100       | 60           |
| 20%   | 0.94        | 0.93         | 164        | 0.93     | 0.97      | 55      | 155        | 0.94       | 0.93        | 152       | 127          |
| 25%   | 0.92        | 0.91         | 236        | 0.91     | 0.97      | 63      | 265        | 0.91       | 0.92        | 217       | 241          |
| 30%   | 0.90        | 0.91         | 308        | 0.90     | 0.96      | 131     | 361        | 0.90       | 0.91        | 282       | 348          |

#### Comparativa con el artículo (Di Martino et al. 2023)
| Ruido | Art. CAE rec | Art. CAE prec | Art. RF rec | Art. RF prec |
|-------|--------------|---------------|-------------|--------------|
| 1%    | 0.92         | 0.71          | 0.95        | 0.56         |
| 5%    | 0.90         | 0.90          | 0.94        | 0.86         |
| 10%   | 0.88         | 0.92          | 0.90        | 0.91         |
| 15%   | 0.81         | 0.84          | 0.87        | 0.90         |
| 20%   | 0.71         | 0.77          | 0.83        | 0.89         |
| 25%   | 0.67         | 0.73          | 0.73        | 0.85         |
| 30%   | 0.59         | 0.71          | 0.60        | 0.77         |

### Observaciones

#### CAE vs Artículo
- CAE sin umbral supera al artículo en recall en niveles altos
  (30%: 0.87 vs 0.59, diferencia +0.28)
- CAE sin umbral tiene menor precisión en niveles bajos
  (1%: 0.42 vs 0.71) por mayor número de FP
- Causa probable: parcelas fiables extraídas sobre TEST separado →
  los CAE son menos específicos fuera de su conjunto de entrenamiento

#### Otsu
- Recall constante ≈ 0.80 en todos los niveles → confirma limitación
  de bimodalidad
- noDet crece dramáticamente con el ruido:
  10%: 225 → 30%: 873 parcelas no detectadas
- A 30% deja sin detectar el 24% de las parcelas infectadas
- Supera al artículo CAE+Otsu en recall en todos los niveles
  (30%: 0.73 vs 0.31)

#### Margen Relativo MSE (propuesta propia)
- Mejor balance recall/precisión que Otsu en todos los niveles
- Reduce FP un 55% respecto a sin umbral manteniendo recall -2%
  (30%: FP 352→159, recall 0.87→0.85)
- Superior a Otsu en recall en niveles altos
  (30%: 0.85 vs 0.73, diferencia +0.12)
- Al 10-30% iguala o supera la precisión del artículo CAE

#### RF
- Supera al artículo en recall en TODOS los niveles
  (30%: 0.90 vs 0.60, diferencia +0.30)
- RF sin umbral ya es competitivo con CAE+Margen MSE
- Margen Prob para RF es irrelevante (diferencia < 0.01 en todos los niveles)
- RF+Otsu mejora precisión pero reduce recall innecesariamente

#### Observaciones generales
- Cotton-Tomato: solapamiento SAR épocas 0-25 explica FP altos en CAE
- Tomato: pierde 81% de píxeles en purga iterativa
- Diferencia metodológica (TEST separado) explica el gap de precisión
  en niveles bajos y el mejor recall en niveles altos


