# Plan de Pruebas: SimilaritySearchServer

Este documento describe la batería de pruebas planificada para validar la funcionalidad del CLI y de la API web de `SimilaritySearchServer` antes y durante la implementación. Las pruebas estarán basadas en bases de datos pseudo-sintéticas generadas a partir de textos del Proyecto Gutenberg.

## 1. Bases de Datos de Prueba

Se generarán 10 bases de datos correspondientes a 10 libros clásicos (ej. Frankenstein, Dracula, Moby Dick).
Para cada libro, los datos se dividirán en párrafos.

**Formato de cada registro:**
*   `id`: Identificador único del párrafo (ej. `doc_1`, `doc_2`).
*   `text`: Texto crudo del párrafo.
*   `vector`: Vector de 128 dimensiones que representa un histograma de los caracteres ASCII imprimibles presentes en el párrafo.
*   `meta`: Diccionario de metadatos del párrafo que incluye:
    *   `word_count` (int): Número de palabras en el párrafo.
    *   `verb_count` (int): Conteo de verbos de uso común (ej. *is, are, was, said, have, go*).
    *   `characters_mentioned` (list): Lista de nombres de personajes principales detectados en el párrafo (obtenidos extrayendo automáticamente las palabras capitalizadas más frecuentes a nivel del libro).

Estas bases de datos se exportarán en formato JSON Lines (`.jsonl`) dentro del directorio `test/data/` para ser fácilmente consumibles tanto por la CLI (mediante `build` y `searchbatch`) como por el servidor HTTP.

## 2. Escenarios de Prueba a Validar

### 2.1. Construcción de Índices (CLI)
*   **Comando:** `similarity-search build`
*   **Archivos de prueba:** Scripts que llamen al comando para cada una de las 10 bases de datos.
*   **Validaciones (Expected Results):**
    *   Creación exitosa del índice `SearchGraph` sobre los vectores densos (histogramas).
    *   Creación de un índice secundario invertido (`BM25InvertedFile` o `InvertedFile`) sobre la columna `text`.
    *   Extracción y almacenamiento correctos de `meta` en una Column Family (RocksDB) que soporte consultas.
    *   Generación de los archivos bajo la estructura `data/{dataset_uuid}/`.

### 2.2. Búsqueda y Filtrado (API Web y CLI)
*   **Búsqueda Métrica pura:** 
    *   **Acción:** Realizar búsquedas de k-vecinos más cercanos (k-NN) usando vectores de histogramas como queries.
    *   **Esperado:** Recuperar los `k` párrafos cuya distribución de caracteres sea más similar. Las métricas a usar serán L2 o Cosine.
*   **Búsqueda Lexical (BM25):**
    *   **Acción:** Buscar palabras clave como `"blood"`, `"ship"`, `"love"`.
    *   **Esperado:** Recuperar párrafos que contengan estas palabras ordenados por relevancia.
*   **Filtrado de Metadatos (Pre-filtro/Post-filtro):**
    *   **Acción:** Buscar vectores pero acotando a párrafos donde `word_count > 50` y `characters_mentioned` contenga `"Alice"`.
    *   **Esperado:** Los resultados devueltos por la búsqueda deben cumplir obligatoriamente con el predicado.
*   **Búsqueda Híbrida (Hybrid Search):**
    *   **Acción:** Búsqueda combinada que proporcione tanto el texto como el vector, haciendo join de los resultados usando Reciprocal Rank Fusion (RRF).

### 2.3. Tareas Pesadas (Jobs)
*   **Batch Search:** Validar `searchbatch` desde la CLI proporcionando un archivo con miles de queries (ej. usando 1 libro como queries sobre el índice de otro libro).
*   **allknn / closestpair:** Someter un Job para calcular el grafo exacto o pares más cercanos. Validar que la API web retorne `202 Accepted` e interactuar con `/api/v1/jobs/*` para ver el progreso.
*   **fft / neardup:** Validar rutinas sin-índice (`fft`) y rutinas de de-duplicación.

### 2.4. Telemetría y Límites
*   **Acción:** Saturar el servidor con múltiples consultas concurrentes de `search`.
*   **Esperado:** Comprobar que el 80% del pool dinámico responde, mientras que la ejecución de `allknn` usa el 20% reservado, sin bloquear el hilo principal.
*   **Token API:** Crear tokens y verificar que rechace llamadas a la API sin token (o con uno expirado/sin permisos).

## 3. Estructura de Scripts
*   `test/generate_datasets.jl`: (Independiente) Descarga los libros, genera vectores y metadatos, emite `test/data/*.jsonl`.
*   `test/run_tests.jl`: Batería de pruebas automatizadas en Julia. Arranca un servidor de prueba y consume los JSONL generados. Utilizará `Test` de Julia para hacer aserciones sobre las respuestas HTTP y los resultados CLI.
