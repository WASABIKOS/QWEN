# CAT_CX_MODEL Release - v1.0.0

## Modelo de Clasificación NLP para Categorías TELCO

### Descripción
Este modelo utiliza un pipeline de Machine Learning para clasificar comentarios y feedback de clientes en categorías específicas del sector de telecomunicaciones (TELCO).

### Arquitectura del Modelo
- **Vectorización:** TF-IDF (Term Frequency-Inverse Document Frequency)
  - Máximo de características: 10,000
  - N-gramas: 1-2 (unigramas y bigramas)
  - Mínimo documento frecuencia: 2
  - Máximo documento frecuencia: 95%

- **Reducción de Dimensionalidad:** TruncatedSVD
  - Componentes: 100
  - Varianza explicada: variable según dataset

- **Clasificador:** Regresión Logística
  - Solver: LBFGS
  - Peso de clases: balanceado
  - Iteraciones máximas: 1000

### Características Principales
- Procesamiento de texto en español
- Lematización opcional con spaCy
- Balanceo de clases con SMOTE (opcional)
- Detección de baja confianza en predicciones
- Exportación de resultados no categorizados para reentrenamiento

### Métricas de Evaluación
El modelo reporta las siguientes métricas:
- Accuracy (Precisión general)
- F1-Score Macro (promedio no ponderado por clase)
- F1-Score Weighted (promedio ponderado por soporte de clase)
- Cross-Validation F1-Macro (5-Fold)

### Archivos Incluidos
- `CAT_CX_MODEL.pkl` - Modelo serializado listo para producción
- `metadata.json` - Información detallada del modelo y configuración

### Uso en Producción

```python
import joblib

# Cargar el modelo
modelo = joblib.load('CAT_CX_MODEL.pkl')

# Extraer componentes
vectorizer = modelo['vectorizer']
svd = modelo['svd']
clf = modelo['modelo']
label_encoder = modelo['label_encoder']
metadata = modelo['metadata']

# Preprocesar texto nuevo
def preprocesar(texto):
    # Aplicar mismas transformaciones que en entrenamiento
    texto_limpio = limpiar_texto(texto)
    return texto_limpio

# Predecir categoría
texto = "Mi internet no funciona desde ayer"
texto_procesado = preprocesar(texto)
X_vec = vectorizer.transform([texto_procesado])
X_svd = svd.transform(X_vec)
prediccion = clf.predict(X_svd)
categoria = label_encoder.inverse_transform(prediccion)[0]

# Obtener probabilidades
probs = clf.predict_proba(X_svd)[0]
confianza = max(probs)
```

### Dependencias
```
pandas >= 1.3.0
scikit-learn >= 1.0.0
joblib >= 1.1.0
numpy >= 1.21.0
openpyxl >= 3.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
spacy >= 3.0.0 (opcional, para lematización)
imbalanced-learn >= 0.8.0 (opcional, para SMOTE)
```

### Versiones
- **v1.0.0** - Versión inicial del modelo
  - Entrenado con datos históricos de categorías TELCO
  - Incluye 6 visualizaciones de evaluación
  - Sistema de detección de baja confianza implementado

### Mantenimiento y Mejora Continua
El modelo incluye un sistema de mejora continua:
1. Revisar archivo `NO_CATEGORIZADOS.xlsx` periódicamente
2. Completar columna `Categoria_Correcta` en casos de baja confianza
3. Incorporar nuevos datos etiquetados al dataset de entrenamiento
4. Reentrenar el modelo usando `1_ENTRENAR_MODELO.py`
5. Comparar métricas antes de desplegar nueva versión

### Contacto y Soporte
Para preguntas o soporte técnico, contactar al equipo de Data Science.

---
**Fecha de Release:** 2024
**Framework:** Scikit-Learn Pipeline
**Idioma:** Español (es_core_news_sm)
