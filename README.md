# VRC / HRV RRi Analyzer Pro v10.7

## Cambios v10.3

- Ajuste a la captura de Kubios Advanced Settings II:
  - SampEn/ApEn: m=2, r=0.2 x SD.
  - DFA alpha1: 4-12 beats.
  - DFA alpha2: 13-64 beats.
  - RQA/D2: embedding dimension=10.
  - RQA threshold=3.1623 x SD.
- Entropías con λ=500 verificable mediante columnas de control:
  - Entropy_lambda
  - Entropy_SD_ms
  - Entropy_r_ms
  - Entropy_N
- MSE: eliminado pseudoconteo A=0.5 introducido en v10.2 porque inflaba escalas altas.
- MSE1 sigue siendo igual a SampEn.


## v10.4
- Añade módulo de diagnóstico Kubios SampEn/MSE en la pestaña No lineales / MSE.
- Para cada escala MSE muestra:
  - N,
  - SD de escala,
  - SD de referencia,
  - r = 0.2 x SD,
  - B matches,
  - A matches,
  - A/B,
  - SampEn calculado,
  - estado: Calculado, A=0, B=0 o puntos insuficientes.
- Permite descargar el diagnóstico en CSV.


## v10.5
- El diagnóstico Kubios SampEn/MSE ahora también aparece dentro del informe automático.
- Para cada fase incluye tabla:
  Fase, Escala, N, r ms, B matches, A matches, A/B, SampEn app y Estado.


## v10.6
- Añade tres modos MSE cuando A=0:
  1. Clásico SampEn: A=0 -> no calculado.
  2. Pseudoconteo 0.5: A=0 -> A=0.5.
  3. Pseudoconteo 1.0: A=0 -> A=1.0.
- Añade selector en la barra lateral: `Modo MSE si A=0`.
- El diagnóstico SampEn/MSE muestra ahora las tres columnas:
  `MSE_clasico`, `MSE_A0_05`, `MSE_A0_1`.
- El informe automático incluye las tres alternativas para comparar con Kubios.


## v10.7
- Corrige la aparición del selector `Modo MSE si A=0` en la barra lateral real.
- El cálculo de ventanas ahora recibe el modo seleccionado.
- Corrige la cabecera del diagnóstico en el informe para mostrar:
  Clásico, A0=0.5 y A0=1.0.


## v10.8 corregida
- Corrige la ruta real de cálculo: `mse_common()` recibe `zero_policy=mse_zero_policy` dentro de `calculate_all()`.
- `mse_metrics()` y `sample_entropy_common()` usan el modo de la barra lateral si no se pasa explícitamente.
- `enforce_entropy_consistency()` ya no referencia una variable no definida y respeta el modo seleccionado.

## v10.9
- Añade cuarto modo MSE: `RCMSE / Composite Kubios-like`.
- Implementa RCMSE aproximado: coarse-graining con todos los offsets y suma de conteos A/B antes de calcular -ln(A/B).
- Añade columnas diagnósticas RCMSE:
  - RCMSE_offsets_validos
  - RCMSE_B_total
  - RCMSE_A_total
  - RCMSE_A/B
  - RCMSE
- El gráfico diagnóstico compara clásico, pseudoconteos y RCMSE.

## v11.0
- Añade ajuste SampEn/MSE con ventana de Theiler.
- Nuevo selector: `Exclusión temporal SampEn/MSE`.
- Permite probar Theiler 0-5 beats para acercar SampEn/MSE1 a Kubios cuando MSE1 ya difiere.
- La ventana Theiler se propaga a SampEn, MSE clásico, pseudoconteos, RCMSE y diagnóstico.


## v11.1
- Añade selector de radio r para SampEn/MSE: fijo λ500, por escala, o fijo RR sin λ.
- El modo se propaga a SampEn, MSE y RCMSE.

## v11.2
Añadidos métodos avanzados:
- Frecuencia: Lomb-Scargle, AR/Yule-Walker y análisis tiempo-frecuencia STFT tipo wavelet.
- No lineales: Hurst, Katz fractal dimension, Petrosian fractal dimension, Dispersion Entropy y MDE 1-20.


## v11.3
- Corrige NameError en `_sample_entropy_core` introducido en v11.2.
- Mantiene métodos avanzados de v11.2: Lomb-Scargle, AR, STFT, Hurst, KatzFD, PetrosianFD, DispEn y MDE.

## v11.4
- Añade Wavelet/STFT scalogram dentro de la pestaña HRV.
- El scalogram permite ver aparición/desaparición de HF, emergencia de LF y cambios transitorios.
- Añade explicaciones internas de definiciones, fórmulas e interpretación orientativa de:
  Lomb-Scargle, AR, STFT/wavelet, Hurst, KatzFD, PetrosianFD, DispEn y MDE.

## v11.5
- Añade Lyapunov_LLE con algoritmo de Rosenstein.
- Incluye definición, fórmula conceptual, valores orientativos e interpretación en la app.
- Lyapunov_LLE se añade al grupo Complejidad y a una tabla interpretativa por fase.

## v11.6
- El informe automático incorpora todas las métricas modernas:
  Lyapunov_LLE, Hurst, KatzFD, PetrosianFD, DispEn, MDE 1-20,
  Lomb-Scargle, AR y Wavelet/STFT.
- Añade definiciones, fórmulas/ideas, valores orientativos e interpretación en el informe.

## v11.7
- Añade métricas wavelet temporales:
  VLF/LF/HF_WAV_MEAN, VLF/LF/HF_WAV_SD, DOM_PCT, episodios, transiciones y entropías.
- Escalograma:
  VLF tiempo verde, LF tiempo azul, HF tiempo rosa, LF/HF tiempo rojo.
- Añade referencia e interpretación automática de métricas wavelet en tablas e informe.

## v11.8
- Cambia el cálculo de dominancia wavelet VLF/LF/HF.
- Antes se comparaba potencia absoluta.
- Ahora se normaliza cada banda por su media temporal antes de calcular dominancia:
  - VLF_n(t)=VLF(t)/mean(VLF)
  - LF_n(t)=LF(t)/mean(LF)
  - HF_n(t)=HF(t)/mean(HF)
- DOM_PCT, episodios, transiciones y entropía de bandas se calculan sobre bandas normalizadas.
- Las potencias VLF_WAV_MEAN/LF_WAV_MEAN/HF_WAV_MEAN y SD siguen siendo absolutas.

## v11.9
- Añade tablas de referencia / valor obtenido / interpretación dentro de la app.
- Añade columna Referencia/normalidad en el informe automático.
- Añade sección clínica completa para fase de referencia con:
  Métrica, referencia, valor obtenido e interpretación.

## v11.9.1
- Corrige error en la tabla referencia/valor/interpretación.
- `_interpret_metric()` ya no intenta convertir textos/configuraciones a `float`.
- Se filtran campos no fisiológicos como `MSE_zero_policy`, `MSE_radius_mode`, `DFA_alpha1_range`, etc.
- `_fmt_num()` tolera `None`, `nan` y textos.

## v12.0
- Sustituye la corrección de artefactos por un modo más cercano a Kubios/Lipponen-Tarvainen.
- Añade modo `kubios scientific`.
- Detección por:
  - mediana local robusta con umbrales Kubios,
  - ajuste por frecuencia cardíaca local,
  - dRR con umbral adaptativo basado en dispersión local de 90 latidos,
  - patrones NP/PN/NPN/PNP,
  - detección aproximada de latido perdido y latido extra.
- Corrección por reconstrucción missed/extra y cubic spline.


## v13.0 — Índices fisiológicos multivariados
- Añade IDX_Vagal, IDX_Amplitud, IDX_Complejidad, IDX_Rigidez, IDX_Adaptabilidad e IDX_Regulacion_Lenta (0-100).
- Añade clasificación automática explicable del perfil autonómico por fase.
- Añade índice de recuperación hacia basal cuando existen fases de recuperación.
- Nueva pestaña 9 con gráficos, tabla, referencias y descarga CSV.
- Exporta índices y referencias a Excel e incorpora los índices al informe automático.
- La predicción clínica a 7 días permanece desactivada hasta disponer de un modelo longitudinal entrenado y validado.

## v13.1 - Gradient Boosting supervisado
- Nueva pestaña `10) Gradient Boosting`.
- Entrenamiento de clasificación o regresión usando los seis índices fisiológicos v13.0.
- Imputación por mediana, partición entrenamiento/validación y validación cruzada.
- Métricas: accuracy equilibrada, F1 macro y matriz de confusión; o MAE y R² en regresión.
- Importancia explicable por permutación.
- Predicción sobre las fases cargadas en la app y exportación CSV.
- Descarga del modelo entrenado en formato joblib.
- La app exige desenlaces reales etiquetados y no inventa probabilidades clínicas.
