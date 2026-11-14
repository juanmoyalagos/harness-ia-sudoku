# Generación de trazas con Qwen3.5-9B (para SFT / DPO del 2B)

Cómo se generaron trazas con `Qwen/Qwen3.5-9B` sobre `data/sudoku/train.jsonl` para entrenar
`Qwen/Qwen3.5-2B` con SFT y/o DPO, y cómo reproducirlo. Este documento cubre solo la
**generación** y la construcción de los datasets de SFT/DPO; el entrenamiento va después.

## Datos ya generados

Los datos resultantes vienen en `sudoku_data_traces.zip`, en la raíz del repositorio:

| Archivo | Contenido |
|---|---|
| `sudoku/train.jsonl` | 450 tareas (4x4, 6x6, 9x9 × easy/medium/hard, 50 c/u) |
| `sudoku/eval.jsonl` | 90 tareas (10 por tamaño y nivel) |
| `sudoku/test.jsonl` | 180 tareas (niveles fijos `n4` … `n50`, 20 por tamaño y nivel) |
| `traces/sft.jsonl` | 498 ejemplos SFT |
| `traces/dpo.jsonl` | 432 pares DPO (182 reales y 250 sintéticos) |

Para usarlos con las rutas por defecto de los scripts (`data/` está en `.gitignore`):

```sh
unzip sudoku_data_traces.zip -d data      # crea data/sudoku/ y data/traces/
```

Las particiones también se pueden regenerar de forma idéntica, sin el zip:

```sh
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku --test 20 --test-grid cells
```

Las respuestas crudas del 9B (los archivos de `baseline.py run`) no se incluyen; solo el
resultado de `build_sft_dpo.py`. Para regenerarlas hay que seguir el procedimiento de abajo.

## Decisiones

| Tema | Decisión |
|---|---|
| Modelo profesor | `Qwen/Qwen3.5-9B` |
| Modo de razonamiento | `--no-thinking` (sin modo think) |
| `--max-output-tokens` | 4096 |
| Tareas | `train.jsonl`: 4x4 (easy, medium, hard) y 6x6 (easy, medium). Sin 6x6 hard ni 9x9 |
| Muestreo | Predeterminado de Qwen sin thinking: temperatura 0.7, top_p 0.8 |
| Muestras por tarea | `n = 8` |
| Few-shot | 2 shots, formato `steps` (elegido tras el piloto, ver Resultados) |

`train.jsonl` tiene 50 tareas por (tamaño, nivel), así que el subconjunto son **250 tareas**
(150 de 4x4 y 100 de 6x6). Con `n = 8` son 2000 muestras. Todas las tareas tienen solución
única demostrada y se resuelven por casillas con un único candidato, por lo que el
verificador (`reward`) es exacto.

## ¿Few-shot o no?

Recomendación: **generar con 2 shots en formato `steps`, pero guardar el prompt sin shots**
(destilación del prompt).

- **A favor de los shots.** Sin ejemplos, el 9B sin thinking puede responder con la rejilla
  directa o con un razonamiento libre de formato variable. Con shots responde
  `Step k: rXcY = v` más la rejilla y termina en `<answer>`. Esa traza es corta, uniforme y
  aprendible por un modelo de 2B, y reproduce el formato que ya usa `make_shots`.
- **El prompt de entrenamiento no lleva shots.** El 2B se evalúa con el prompt base. Cada
  ejemplo de SFT/DPO es `prompt de train.jsonl → completion`, de modo que el 2B aprende a
  producir el estilo de los shots sin necesitar los ejemplos en contexto. Las trazas siguen
  cumpliendo el contrato de `parse_answer`, porque los shots no cambian el formato de
  respuesta.
- **En contra.** Los shots alargan el prompt (límite 16384 tokens, ya automático con
  `--shots`) y hacen más lenta la generación. También puede copiarse el estilo de los shots
  aunque sea subóptimo.
- **Piloto antes de decidir.** Correr 0 y 2 shots sobre 5 tareas por nivel con `n = 8` y
  comparar con `baseline.py summary`: tasa de aprobación (pass@1), tasa de truncamiento,
  media de tokens. Se usa 2 shots salvo que 0 shots apruebe igual o más y con trazas de
  formato igualmente limpio. El formato `compact` es una alternativa más corta si las trazas
  de 6x6 `medium` truncan a 4096; `seen` es más largo.

## Opciones de `baseline.py run` usadas

- `--sizes 4 6 --skip 6:hard` deja solo el subconjunto de la tabla de decisiones
  (`--skip SIZE:LEVEL` excluye pares tamaño/nivel).
- `--shard I/K` (con `0 <= I < K`) corre solo una de cada `K` tareas, empezando por la `I`.
  Sirve para repartir la corrida entre varias GPUs o procesos; cada shard necesita su propio
  `--out`.
- Si una corrida se interrumpe, repetir el mismo comando la reanuda. Reanudar sobre el mismo
  archivo con otros ajustes se rechaza a propósito (`RUN_KEYS`).

## Procedimiento

1. **Entorno.** Python 3.13, entorno virtual e instalación de dependencias
   (ver `README.md`):
   ```sh
   python -m venv .venv && source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
   ```
   El 9B en bf16 ocupa unos 18 GB de pesos. Las corridas originales usaron una GPU NVIDIA
   de 48 GB por proceso; con menos memoria, bajar `--batch-size`. Los pesos se descargan de
   Hugging Face la primera vez (`HF_HOME` define dónde se guardan).
2. **Datos.** Descomprimir el zip o regenerar las particiones (ver "Datos ya generados").
3. **Piloto**, un directorio de salida por variante:
   ```sh
   COMMON="--tasks data/sudoku/train.jsonl --model-id Qwen/Qwen3.5-9B --no-thinking \
     --max-output-tokens 4096 --sizes 4 6 --skip 6:hard --per-level 5 --n 8"
   python Laboratorio/Qwen3.5/baseline.py run $COMMON --out results/pilot/0shot
   python Laboratorio/Qwen3.5/baseline.py run $COMMON --shots 2 --out results/pilot/2shot
   python Laboratorio/Qwen3.5/baseline.py summary results/pilot/*/*.jsonl
   ```
   Del piloto sale también el tiempo por muestra, para dimensionar la corrida completa. Las
   dos variantes son independientes y se pueden correr en paralelo.
4. **Corrida completa (r0)**: 2 shots, formato `steps`. En un solo proceso:
   ```sh
   COMMON="--tasks data/sudoku/train.jsonl --model-id Qwen/Qwen3.5-9B --no-thinking \
     --max-output-tokens 4096 --sizes 4 6 --skip 6:hard --n 8 --shots 2"
   python Laboratorio/Qwen3.5/baseline.py run $COMMON --out results/traces/r0
   ```
   O repartida en 5 shards, por ejemplo uno por GPU. Cada shard toma una de cada 5 tareas,
   así que todos reciben las mismas 10 tareas por (tamaño, nivel) y tardan lo mismo:
   ```sh
   for I in 0 1 2 3 4; do
     CUDA_VISIBLE_DEVICES=$I python Laboratorio/Qwen3.5/baseline.py run $COMMON \
       --shard $I/5 --out results/traces/r0/shard$I &
   done
   wait
   ```
   Como referencia, en una GPU NVIDIA L40S tomó ≈2,4 min por tarea (≈10 GPU-horas en total,
   ≈2 h por shard). Si un shard se interrumpe, repetir su comando lo reanuda.
5. **Revisar la cobertura** con `summary`: por (tamaño, nivel) la tasa de aprobación, la tasa
   de truncamiento y la media de tokens. Esto indica cuántas tareas tienen aprobadas (SFT) y
   cuántas tienen aprobadas y fallidas a la vez (pares DPO).
   ```sh
   python Laboratorio/Qwen3.5/baseline.py summary results/traces/r0/*.jsonl      # un proceso
   python Laboratorio/Qwen3.5/baseline.py summary results/traces/r0/shard*/*.jsonl   # shards
   ```
6. **Rondas extra si falta cobertura** (típicamente 6x6 medium): volver a correr con otro
   `--seed` y, si se quiere diversidad de shots, otro `--shot-seed`, siempre en un `--out`
   nuevo. La ronda r1 usada aquí fue:
   ```sh
   python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku/train.jsonl \
     --model-id Qwen/Qwen3.5-9B --no-thinking --max-output-tokens 4096 --n 8 --shots 2 \
     --sizes 6 --skip 6:hard --temperature 1.0 --seed 1 --shot-seed 1 \
     --out results/traces/r1
   ```
7. **Construir los datasets** (ver la sección siguiente).

## Qué queda registrado por muestra

`task_id`, `size`, `level`, `empty_cells`, `sample`, `reward`, `reward_parts` (format,
clues_frac, fill, solved, exact), `finish_reason`, `generated_tokens`, `completion` y todos
los ajustes de la corrida. El prompt no se duplica: se recupera de `train.jsonl` por
`task_id`. Los shots tampoco se guardan; se reconstruyen con `--shot-seed`, pero no entran
en los datos de entrenamiento.

## Resultados

### Piloto (5 tareas por nivel, `n = 8`)

| Tablero / nivel | pass@1 0 shots | pass@1 2 shots | truncadas 0 / 2 shots | tokens medios 0 / 2 shots |
|---|---|---|---|---|
| 4x4 easy | 1.00 | 1.00 | 0% / 0% | 1295 / 795 |
| 4x4 medium | 1.00 | 1.00 | 0% / 0% | 1554 / 1321 |
| 4x4 hard | 1.00 | 0.93 | 0% / 8% | 2211 / 1991 |
| 6x6 easy | 0.88 | 0.85 | 12% / 10% | 2644 / 2223 |
| 6x6 medium | 0.47 | 0.60 | 52% / 40% | 3508 / 3217 |

En 6x6 medium pass@8 es 0.60 sin shots y 1.00 con 2 shots. Con 2 shots las respuestas son
más cortas y truncan menos, así que se eligió **2 shots, formato `steps`**. Cada celda tiene
solo 5 tareas (40 muestras): los resultados son indicativos. Tiempo medido: ≈2,4 min por
tarea (1,0 min en 4x4 easy hasta 3,5 min en 6x6 medium).

### Corrida completa (r0: 250 tareas, 2000 muestras, 5 shards)

| Tablero / nivel | pass@1 | pass@8 | truncadas | tokens medios |
|---|---|---|---|---|
| 4x4 easy | 0.99 | 1.00 | 0% | 771 |
| 4x4 medium | 0.98 | 1.00 | 1% | 1113 |
| 4x4 hard | 0.96 | 1.00 | 4% | 1772 |
| 6x6 easy | 0.87 | 1.00 | 10% | 2127 |
| 6x6 medium | 0.64 | 0.96 | 34% | 3053 |

- **SFT:** 1775 de 2000 muestras son correctas (reward 1.0); 248 de 250 tareas tienen al
  menos una (las 2 sin ninguna son 6x6 medium).
- **DPO:** 95 tareas tienen muestras correctas y fallidas a la vez (74 de ellas en 6x6);
  153 tareas aprobaron todas las muestras (129 en 4x4, 20 en 6x6 easy) y no dan pares.
- **Los fallos son sobre todo truncamientos:** de 225 muestras fallidas, 194 (86%) se
  cortaron a 4096 tokens y solo 31 son rejillas erróneas o formato roto. Los rechazados de
  DPO enseñarían sobre todo "ser más corto".
- **Sesgo a lo fácil:** de las 1775 correctas, ≈1170 son 4x4 y ≈600 son 6x6.

### Ronda extra r1 (100 tareas 6x6, 800 muestras, temperatura 1.0, `--seed 1`, `--shot-seed 1`)

El objetivo era conseguir más pares DPO con rechazados de rejilla errónea.

| Tablero / nivel | pass@1 | truncadas | tokens medios |
|---|---|---|---|
| 6x6 easy | 0.91 | 9% | 2140 |
| 6x6 medium | 0.63 | 36% | 3166 |

La temperatura 1.0 no cambió la tasa de aprobación ni la de truncamiento respecto a r0
(0.87 / 0.64 y 10% / 34%), así que casi no hay más rejillas erróneas: r1 solo aportó 6
rechazados de ese tipo (30 → 36 en total con cap 2).

## Construcción de los datasets

`Laboratorio/Qwen3.5/build_sft_dpo.py` lee los resultados (archivos o directorios, que
recorre de forma recursiva), recupera el prompt de `train.jsonl` por `task_id` (sin shots) y
escribe `sft.jsonl` y `dpo.jsonl`. Rechaza muestras que no sean del split `train`.

```sh
python Laboratorio/Qwen3.5/build_sft_dpo.py --results results/traces/r0 results/traces/r1 \
    --tasks data/sudoku/train.jsonl --out data/traces --synthetic-per-task 1
```

- **SFT** (`prompt`, `completion`): muestras con `reward == 1.0` y sin truncar; hasta
  `--max-sft-per-task` (2) completions distintas por tarea, las más cortas primero.
- **DPO** (`prompt`, `chosen`, `rejected`, `rejected_truncated`, `rejected_synthetic`): hasta
  `--max-pairs-per-task` (2) pares por tarea sin repetir completions. `chosen` es la correcta
  más corta. `rejected` prefiere rejillas erróneas que terminaron (la más cercana a la
  correcta primero) y solo después truncadas.
- **Pares sintéticos** (`--synthetic-per-task N`, por defecto 0): N pares extra por tarea cuyo
  `rejected` es la propia traza correcta con **una celda originalmente vacía cambiada en la
  rejilla del `<answer>` final** (`rejected_synthetic = true`). Es el rechazado de rejilla
  errónea que el 9B casi no produce (sus fallos son casi todos truncamientos). Cada uno se
  comprueba con `reward` (formato correcto y `solved = 0`) y la semilla depende de
  `task_id`, así que el resultado es determinista. `chosen` rota entre las correctas
  distintas, empezando por las que no usaron los pares reales de esa tarea.
  - **Limitación:** son rechazados fuera de política. El razonamiento previo (que termina en
    "All constraints satisfied") se deja tal cual y solo la rejilla final es incorrecta, de
    modo que el par enseña sobre todo a copiar fielmente la última rejilla verificada. Los
    campos `rejected_synthetic` y `rejected_truncated` permiten filtrarlos o pesarlos.
- **Resultado con r0 y r1 juntos y 1 par sintético por tarea:** 498 ejemplos SFT (100 por
  nivel, 98 en 6x6 medium) y **432 pares DPO**:

  | | 4x4 easy | 4x4 medium | 4x4 hard | 6x6 easy | 6x6 medium | total |
  |---|---|---|---|---|---|---|
  | reales | 3 | 6 | 16 | 65 | 92 | 182 |
  | sintéticos | 50 | 50 | 50 | 50 | 50 | 250 |

  De los 182 pares reales, 36 tienen un rechazado con rejilla errónea y 146 uno truncado.
  Solo con r0 y sin sintéticos eran 492 ejemplos SFT y 148 pares reales (30 con rejilla
  errónea).
- Los archivos quedan en `data/traces/` (ignorado por git; la versión usada está en
  `sudoku_data_traces.zip`). Hay tests en `test_build_sft_dpo.py`.

## Siguientes pasos

1. **SFT** del 2B con `data/traces/sft.jsonl` (el formato de entrenamiento depende de la
   librería elegida; los campos son strings simples).
2. **Pares DPO:** ya generados con rechazados sintéticos (ver arriba). Subir
   `--max-pairs-per-task` a 3 da 52 pares reales más, casi todos con rechazado truncado.
3. **Rechazados on-policy del 2B tras el SFT** (opcional, mejor calidad que los sintéticos):
   muestrear el 2B ya entrenado sobre las mismas tareas y emparejar sus fallos (sobre todo
   rejillas erróneas) con las trazas correctas del 9B. Es la fuente de mejor calidad para DPO
   y el 2B es mucho más rápido que el 9B.
4. **Descartado:** otra ronda del 9B para más rechazados erróneos. r1 (temperatura 1.0) apenas
   aportó 6 y el 9B falla casi siempre por truncamiento.
5. **Evaluación:** el 2B se evalúa con `baseline.py` sobre `eval.jsonl` con los mismos
   ajustes que su baseline. Medir aparte 6x6 hard y 9x9 para ver si generaliza o retrocede
   en lo que no vio.

## Riesgos

- **Desbalance (confirmado).** 4x4 aprueba casi todo y 6x6 medium el 64%; las trazas
  correctas se concentran en lo fácil. Se mitiga limitando las completions por tarea (el
  SFT queda en ≈100 por nivel) y, si hace falta, generando un split `train` más grande con
  `dataset.py`.
- **Truncamiento a 4096 tokens (confirmado).** Afecta al 34% de 6x6 medium y al 10% de 6x6
  easy, y explica el 86% de los fallos. Una alternativa es el formato `compact` (no
  probado).
- **Mismo generador que el test.** Las semillas impiden compartir puzzles entre splits, y se
  verificó que ningún prompt ni puzzle de `train.jsonl` aparece en `eval.jsonl` ni en
  `test.jsonl`.
