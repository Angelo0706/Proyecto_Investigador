# Registro de Experimentos — FARMSAR Sevilla 2017

## EXP-001 — Línea base backscatter
**Fecha:** 2024-11-18
**Estado:** ✅ En curso (3/7 niveles completados)
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

### Resultados parciales

| Ruido | Método       | Recall | Precisión | FP  | noDet |
|-------|--------------|--------|-----------|-----|-------|
| 1%    | CAE sin umb  | 0.91   | 0.42      | 124 | 7     |
| 1%    | CAE Otsu     | 0.80   | 0.95      | 4   | 20    |
| 1%    | CAE Margen   | 0.91   | 0.59      | 63  | 7     |
| 1%    | RF sin umb   | 0.98   | 0.67      | 49  | 2     |
| 1%    | RF Otsu      | 0.95   | 0.98      | 2   | 5     |
| 5%    | CAE sin umb  | 0.89   | 0.71      | 199 | 46    |
| 5%    | CAE Otsu     | 0.80   | 0.91      | 38  | 106   |
| 5%    | CAE Margen   | 0.88   | 0.84      | 82  | 52    |
| 5%    | RF sin umb   | 0.98   | 0.91      | 56  | 14    |
| 5%    | RF Otsu      | 0.95   | 0.99      | 6   | 29    |
| 10%   | CAE sin umb  | 0.92   | 0.84      | 189 | 76    |
| 10%   | CAE Otsu     | 0.80   | 0.95      | 39  | 225   |
| 10%   | CAE Margen   | 0.90   | 0.91      | 82  | 100   |
| 10%   | RF sin umb   | 0.96   | 0.94      | 74  | 45    |
| 10%   | RF Otsu      | 0.95   | 0.98      | 17  | 66    |

### Observaciones
- RF supera al artículo en recall y precisión sin umbral adicional
- Margen MSE mejor balance recall/precisión que Otsu en CAE
- Otsu mantiene recall=0.80 constante → limitación de bimodalidad confirmada
- Cotton-Tomato: solapamiento SAR épocas 0-25 explica FP altos en CAE
- Tomato: pierde 81% de píxeles en purga iterativa
- Margen Prob para RF es irrelevante (RF ya es preciso)

### Pendiente
- [ ] Resultados niveles 15%, 20%, 25%, 30%
- [ ] Tests estadísticos Wilcoxon CAE vs RF
- [ ] Figuras curvas recall/precisión vs nivel de ruido
- [ ] EXP-002: backscatter + coherencia

---


