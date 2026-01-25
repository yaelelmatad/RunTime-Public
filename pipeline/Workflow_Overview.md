# Runtime Data Engineering Workflow

This directory contains the sequential stages of the **Runtime** data pipeline. Each notebook transforms raw athletic records into the structured Runtime grammar used by the Transformer model.

## Sequential Stages

### [Stage 01] Data Acquisition (`01_Data_Acquisition.ipynb`)
- **Availability note**: The full Stage 01 acquisition/pulling step (and raw-data retrieval instructions) is **not provided in this public repo** to prevent abuse (e.g., automated scraping / bulk pulling of underlying results). The rest of the pipeline can be reviewed, and the modeling code can be run end-to-end, using the included sample shards in `data/samples/`. Interested parties can reach out to the authors/maintainers for additional details as appropriate.
- **Input**: Raw NYRR race result parquets/JSONs.
- **Process**: Normalizing schemas across different years and sources.
- **Output**: Unified race result dataset.

### [Stage 02] Weather Extraction (`02_Weather_Extraction.ipynb`)
- **Input**: Unified race results + external weather API data.
- **Process**: Mapping race dates and locations to historical meteorological conditions (temperature, humidity, wind).
- **Output**: Weather-hydrated race records.

### [Stage 03] Runner Career Grouping (`03_Runner_Career_Grouping.ipynb`)
- **Input**: Hydrated race records.
- **Process**: Heuristic-based deduplication and grouping. This stage converts individual race entries into chronological careers for unique athletes.
- **Output**: Grouped runner careers.

### [Stage 04] Weather Grammar Creation (`04_Weather_Grammar_Creation.ipynb`)
- **Input**: Weather-hydrated records.
- **Process**: Defining the vocabulary and binning strategy for weather conditions (categorical and numerical).
- **Output**: Weather condition grammar bins (`.pickle`).

### [Stage 05] Distance Grammar Creation (`05_Distance_Grammar_Creation.ipynb`)
- **Input**: Unified race dataset.
- **Process**: Defining the vocabulary for race distances and mapping them to unique distance tokens.
- **Output**: Race distance grammar bins (`.pickle`).

### [Stage 06] Pace Grammar Creation (`06_Pace_Grammar_Creation.ipynb`)
- **Input**: Grouped runner careers.
- **Process**: Implementing the **Pace Binning** strategy and tokenizing temporal deltas.
- **Output**: Pace grammar lookup tables and statistics (`.pickle`).

### [Stage 07] Unified Grammar Integration (`07_Unified_Grammar_Integration.ipynb`)
- **Input**: Output from stages 04, 05, and 06.
- **Process**: Merging individual grammar definitions into a unified Runtime grammar.
- **Output**: Final unified grammar mapping.

### [Stage 08] Hydration and Tokenization (`08_Hydration_and_Tokenization.ipynb`)
- **Input**: Grouped careers + Unified grammar.
- **Process**: Mapping every race in every career into the final **11-token block sequence**.
- **Output**: Token-hydrated career sequences.

### [Stage 09] Final Dataset Generation (`09_Final_Dataset_Generation.ipynb`)
- **Input**: Token-hydrated careers.
- **Process**: Sharding the dataset into multiple training splits (`runners_split_XXX.pkl.gz`).
- **Output**: Production-ready training data.

---

## Design Principles
1. **Irregular Intervals**: Unlike traditional time-series preprocessing that resamples to a fixed frequency, this pipeline preserves the exact temporal distance between events as discrete grammar tokens.
2. **Context-Heavy Blocks**: Each observation is packaged with its environment (weather) and runner state (age) to ensure the Transformer has a complete snapshot of the causal conditions.
