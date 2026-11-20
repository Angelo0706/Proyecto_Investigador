# Proyecto_investigador
# FARMSAR — Detección de Anomalías en Censos Agrícolas

Implementación y extensión de la metodología FARMSAR 
para la detección y corrección automática de errores 
de etiquetado en censos agrícolas mediante teledetección SAR.

## Dataset
- Sensor: Sentinel-1 (VV + VH backscatter)
- Área: Sector BXII, Sevilla, España
- Año: 2017
- Clases: 16 cultivos
- Adquisiciones: 61 épocas temporales

## Metodología
- Autoencoders Convolucionales 1D especializados por clase
- Random Forest con validación cruzada por parcela
- Purga iterativa FARMSAR (10 pasadas)
- Propuesta propia: umbral por margen relativo MSE

## Referencia
Di Martino et al. (2023). FARMSAR: Fixing AgRicultural 
Mislabels using Sentinel-1 ARchive. Remote Sensing, 15, 35.
