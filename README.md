# Indian Real Estate Price Pipeline — End-to-End Documentation

**Scrape → Structure → Analyze → Predict (Rent Calculator)**

> **Goal.** Build a fully reproducible pipeline that scrapes listings from popular Indian real-estate portals, constructs a clean, analysis-ready dataset, performs exploratory analysis and outlier handling, and trains models to power a **price (monthly rent) estimator**.

---

## Contents

- [Project structure](#project-structure)
- [Data sources & scope](#data-sources--scope)
- [Architecture overview](#architecture-overview)
- [Notebook 1 — URL inventory (Selenium + infinite scroll)](#notebook-1--url-inventory-selenium--infinite-scroll)
- [Notebook 2 — Detail scraping & dataset creation](#notebook-2--detail-scraping--dataset-creation)
- [Notebook 3 — Cleaning, standardization, and EDA](#notebook-3--cleaning-standardization-and-eda)
- [Notebook 4 — Modeling & rent calculator tool](#notebook-4--modeling--rent-calculator-tool)
- [Feature schema (data dictionary)](#feature-schema-data-dictionary)
- [Design choices & rationale](#design-choices--rationale)
- [Results snapshot (to be added)]
- [How to run (reproducibility)](#how-to-run-reproducibility)
- [Extending to other portals/cities](#extending-to-other-portalscities)
- [Limitations / TODO](#limitations--todo)
- [Troubleshooting](#troubleshooting)
- [Ethics, ToS, and rate limits](#ethics-tos-and-rate-limits)
- [Appendix: key functions (signatures & notes)](#appendix-key-functions-signatures--notes)

---

## Project structure

Four notebooks form the pipeline:

1. **`Realtor url extraction.ipynb`** — Collect **listing URLs** from location search pages (handles **infinite scroll**).
2. **`realtor-dataset-creation.ipynb`** — Visit each URL; **extract structured attributes** into a **wide table**.
3. **`realtor_data_analysis.ipynb`** — **Clean** & **normalize** fields; perform **EDA**; trim outliers.
4. **`realtor_tool.ipynb`** — Build **ML models** and expose a **`predict_rent(...)`** helper (the “rent calculator”).

**Intermediate & final artifacts**

- `realtor_urls.csv` — inventory of `(Location, URL)` pairs.
- `property_data.json` — raw scraped items (checkpoint).
- `processed_urls.json` — set of completed URLs (for resume/retry).
- `realtor_dataset.csv` — normalized, union-schema dataset (raw → wide table).
- `realtor_dataset_copy.csv` — cleaned, analysis/modeling copy.

    .
    ├── Realtor url extraction.ipynb
    ├── realtor-dataset-creation.ipynb
    ├── realtor_data_analysis.ipynb
    └── realtor_tool.ipynb

---

## Data sources & scope

- **Sources**: Popular Indian real-estate portals (dynamic pages with infinite scroll + JS-rendered details).
- **Geos**: Any Indian city/neighborhood with a location search page.
- **Target variable**: **Monthly `Rent` (INR)**.
- **Canonical features**:  
  Categorical → `Location`, `No. of Bedroom`, `Parking`, `Furnishing Status`, `Gated Security`  
  Numeric → `Total SqFt`, `Age of Building`  
  *(Additional keys captured opportunistically from detail pages; see schema.)*

---

## Architecture overview

High-level flow (ASCII):

    [Search Pages] --(Selenium + infinite scroll)--> [All Listing Cards]
           |                                           |
           +------ extract <a href> (URLs) -----------+
                                                       v
                                         [realtor_urls.csv]
                                                       |
                                                       v
                                           [Detail Page Scraper]
                                                       |
                                  key/values + metrics, retries, checkpoints
                                                       v
                                            [property_data.json]
                                                       |
                                union of keys + normalization (wide schema)
                                                       v
                                           [realtor_dataset.csv]
                                                       |
                                       cleaning + EDA + outliers trimmed
                                                       v
                                         [realtor_dataset_copy.csv]
                                                       |
                                         preprocessing + model training
                                                       v
                                         Rent Estimator: predict_rent()

---

## Notebook 1 — URL inventory (Selenium + infinite scroll)

**Purpose.** Build a complete list of listing URLs for each target location page.

**Highlights**
- Detects total **`propertyCount`** on the page.
- Performs **infinite scroll** until the number of loaded `<article>` cards ≥ `propertyCount` or a stall detector triggers.
- Extracts each card’s `<a href>`, de-duplicates, and writes to `realtor_urls.csv`.

**Key behaviors**
- `WebDriverWait` + `ExpectedConditions` ensure DOM readiness before counting.
- **Attempt counter** (stall detection) with small random sleeps to mimic human scrolling.

**Output**: `realtor_urls.csv` with columns: `Location`, `URL`.

---

## Notebook 2 — Detail scraping & dataset creation

**Purpose.** Visit each listing URL and extract structured attributes; accumulate **all observed keys** into a wide table.

**Anti-ban & reliability**
- **User-Agent rotation** by batch.
- **Retry loop** per URL (e.g., 3 attempts).
- **Session batching** and periodic **checkpointing**:
  - `property_data.json` — append raw dicts
  - `processed_urls.json` — maintain a set for resume

**Extractor (typical targets)**
- **Rent** — strings like `₹27,000/M` → normalized integer.
- **Total SqFt**, **Deposit** — numeric parses (commas stripped).
- **Key/value panels** — e.g., Bedrooms, Parking, Furnishing, Gated Security, Floor/Total Floors, Age of Building.
- **Livability Score**, **Transit Score** — portal-provided (0–10).

**Schema building**
- After scraping, compute the **union of keys** across all items.
- Build a wide `DataFrame`; sparse attributes → `NaN`.
- Save to **`realtor_dataset.csv`**.

---

## Notebook 3 — Cleaning, standardization, and EDA

**Purpose.** Make the dataset analysis-ready, with consistent numerics/categoricals and trimmed tails.

**Utilities**
- `extract_rent_value(text) -> int|NaN` — robust numeric parse for `Rent`.
- `process_floor(text) -> numeric/levels` — map “Ground”, “1 out of 5”, etc., to numeric representation.
- `trim_extremes(df, column, lower_percent, upper_percent)` — quantile-based outlier filter.
- Interactive summaries:
  - **By `Location`**
  - **By (`Location`, `No. of Bedroom`)**

**Outputs**
- Cleaned copy **`realtor_dataset_copy.csv`** for modeling.
- Optional derived fields (e.g., `Cost_per_SqFt = Rent / Total SqFt` when appropriate).

> **Deposit parsing note**: replace any placeholder logic (e.g., non-null→0) with proper numeric parsing and **impute** missing values (e.g., median by `Location` × `Bedroom`).

---

## Notebook 4 — Modeling & rent calculator tool

**Target.** `Rent` (integer, INR/month)

**Features (production subset)**
- **Numeric**: `Total SqFt`, `Age of Building`
- **Categorical**: `Location`, `No. of Bedroom`, `Parking`, `Furnishing Status`, `Gated Security`

**Preprocessing**
- Numeric: `SimpleImputer(strategy="mean")`
- Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
- Combined via `ColumnTransformer` in an sklearn `Pipeline`.

**Models tried**
- **RandomForestRegressor** (baseline ensemble)
- **GradientBoostingRegressor** (stronger baseline)
- **Keras Dense Regressor** (MAE loss, Adam; uses dense matrix from the preprocessor)

**Train/test**
- `train_test_split(test_size=0.2, random_state=0)`
- **Primary metric**: **MAE** (scale-preserving, robust to outliers).  
  *(Add RMSE/R² as secondary metrics if desired.)*

**Runtime helper (signature)**

    def predict_rent(input_data):
        """
        Input: pd.DataFrame or dict with columns:
          Location, No. of Bedroom, Parking, Furnishing Status, Gated Security, Total SqFt, Age of Building
        Output: np.ndarray of predicted monthly rents (INR).
        """

**Example**

    example = {
      "Location": "HSR Layout",
      "No. of Bedroom": "2 Bedroom",
      "Parking": "Bike and Car",
      "Furnishing Status": "Fully Furnished",
      "Gated Security": "Yes",
      "Total SqFt": 1200,
      "Age of Building": 5
    }
    predict_rent(example)

---

## Feature schema (data dictionary)

| Column               | Type      | Notes / Examples                                                                  |
|---------------------|-----------|------------------------------------------------------------------------------------|
| `URL`               | string    | Source listing URL                                                                 |
| `Location`          | category  | Area / neighborhood                                                                |
| `No. of Bedroom`    | category  | e.g., *1 Bedroom*, *2 Bedroom*                                                     |
| `Total SqFt`        | number    | Built-up area (sq. ft.), numeric                                                   |
| `Rent`              | integer   | Monthly rent (INR), cleaned                                                        |
| `Deposit`           | integer   | Security deposit (INR), cleaned/imputed                                            |
| `Age of Building`   | number    | Years (approx.)                                                                    |
| `Parking`           | category  | e.g., *Bike and Car*, *Car Only*, *None*                                           |
| `Furnishing Status` | category  | *Unfurnished*, *Semi*, *Fully*                                                     |
| `Gated Security`    | category  | *Yes* / *No*                                                                       |
| `Floor`             | text/num  | Raw descriptor (normalized via `process_floor`)                                    |
| `Total Floors`      | number    | Building floors total                                                              |
| `Livability Score`  | number    | 0–10 (portal-provided)                                                             |
| `Transit Score`     | number    | 0–10 (portal-provided)                                                             |
| `...`               | mixed     | Additional keys discovered during scraping (union schema → sparse columns as NaN)  |

---

## Design choices & rationale

1. **Selenium over requests/BS4**  
   Listing & detail pages are **SPA/infinite-scroll** with **lazy loading**; Selenium ensures the rendered DOM is complete before harvesting.

2. **Quantile trimming for outliers**  
   Real-estate prices have **long tails**. Non-parametric quantile filters are simple, interpretable, and robust.

3. **One-Hot for categoricals**  
   - Tree ensembles handle OHE well; no leakage from target encoding.  
   - For NNs, OHE + dense layers is a clean baseline.  
   - `handle_unknown="ignore"` allows **unseen categories at inference** without crashes.

4. **MAE as the primary metric**  
   INR-scale error is directly interpretable; less sensitive to high-rent outliers than MSE/RMSE.

---

## Results snapshot (fill after a run)

> Replace the `TBD` values with actual metrics from your latest execution.

**Dataset summary**
- Rows: **TBD**
- Cities/Locations: **TBD**
- Time window: **TBD**

**Model performance (test set, MAE in INR/month)**

| Model                     | MAE (↓) | RMSE | R²  |
|---------------------------|---------|------|-----|
| RandomForestRegressor     | **TBD** | TBD  | TBD |
| GradientBoostingRegressor | **TBD** | TBD  | TBD |
| Keras Dense Regressor     | **TBD** | TBD  | TBD |

**Example prediction (sanity)**

    Input: 2BHK • 1200 sqft • Fully Furnished • Parking: Bike+Car • Gated • HSR Layout • Age: 5y
    Predicted: ₹ TBD / month

---

## How to run (reproducibility)

**1) Environment**

    python -V           # ≥ 3.9 recommended
    pip install pandas numpy selenium scikit-learn tensorflow matplotlib seaborn regex joblib
    # Install Chrome + matching ChromeDriver, or:
    pip install webdriver-manager

**2) URL inventory**  
Open **`Realtor url extraction.ipynb`**, set `(location, search_url)` seeds, run all → produces `realtor_urls.csv`.

**3) Detail scraping**  
Run **`realtor-dataset-creation.ipynb`** in batches. It will:
- Rotate **User-Agents**
- Retry on failures
- Periodically save **`property_data.json`**, **`processed_urls.json`**
- Produce **`realtor_dataset.csv`**

**4) Cleaning & EDA**  
Open **`realtor_data_analysis.ipynb`**, point to `realtor_dataset.csv`, apply cleaners/outlier trimming; export **`realtor_dataset_copy.csv`**.

**5) Modeling & tool**  
Open **`realtor_tool.ipynb`**, set file path to the **cleaned copy**, train models, and use `predict_rent(...)`.

> **Tip**: Persist the final sklearn pipeline via `joblib.dump(pipeline, "rent_rf.pkl")`. For Keras, use `model.save("rent_nn.keras")`. Save the **preprocessor** together with the model.

---

## Extending to other portals/cities

- **DOM selectors**: Duplicate the extractor and update `By.CSS_SELECTOR` / `By.XPATH` to match the new site (e.g., 99acres, MagicBricks, Housing, NoBroker). Keep **output keys** consistent.
- **Schema evolution**: The union-schema builder auto-adds new columns. Add cleaners in Notebook 3 for any new fields.
- **Scale & resilience**: Add **proxy pools**, **exponential backoff**, and **distributed crawling** if scaling beyond a few cities.

---

## Limitations / TODO

- **Deposit parsing**: Replace any placeholder logic with full numeric parsing; impute by `Location × Bedroom`.
- **Time awareness**: If scraping spans months, add **date features** and consider **time-aware splits** for evaluation.
- **Cross-validation**: Add K-fold or **blocked CV** by location/time for robust generalization estimates.
- **Feature scaling for NN**: Consider `StandardScaler` for numeric features before Keras.
- **Model registry/UI**: Add a small **Streamlit** app for interactive predictions; store artifacts with versions.
- **`pd.concat` vs `df.append`**: Ensure modern, efficient DataFrame building.

---

## Troubleshooting

- **Scroll stalls before reaching `propertyCount`**  
  Increase `max_attempts` or `SCROLL_PAUSE_TIME`; try a small up-scroll (height − 200) between full scrolls.

- **Categories unseen at inference**  
  Ensure the exact **column names** exist; with `handle_unknown="ignore"`, unseen labels are safely dropped at transform.

- **Driver/Chrome mismatch**  
  Use `webdriver-manager` to auto-resolve driver versions.

- **Frequent 403 / blocks**  
  Slow down, rotate UAs less predictably, add small random `sleep`, and consider proxies.

---

## Ethics, ToS, and rate limits

- Always check each portal’s **robots.txt** and **Terms of Service**.
- Throttle requests, avoid heavy parallelization, and cache intermediate data.
- Use scraped content only for **research/prototyping** unless you have explicit permission.

---

## Appendix: key functions (signatures & notes)

**Notebook 1 — URL inventory**

    def get_property_count(driver) -> int: ...
    def scroll_until_property_count(driver, property_count, max_attempts=10) -> list[WebElement]: ...
    def get_property_urls(articles, count) -> list[str]: ...
    def scrape_urls(locations_and_urls, rate_limit=10) -> pd.DataFrame: ...

**Notebook 2 — Detail scraping**

    def configure_driver(user_agent: str):
        """Build a Chrome driver with UA override and sane defaults."""

    def extract_data(driver) -> dict:
        """
        Scrape the current detail page; parse Rent/Deposit/SqFt and key/value panels.
        Resilient with try/except on missing nodes.
        """

    def process_url(url: str, retries: int = 3) -> dict|None: ...
    def save_intermediate_data(data: list[dict], processed_urls: set[str]) -> None: ...
    def load_processed_urls() -> set[str]: ...
    def process_urls(url_list: list[str], urls_per_session=10, save_interval=5) -> list[dict]: ...

**Notebook 3 — Cleaning & EDA**

    def extract_rent_value(text: str) -> int|float|None: ...
    def process_floor(text: str) -> int|float|None: ...
    def trim_extremes(df: pd.DataFrame, column: str, lower_percent=0.01, upper_percent=0.01) -> pd.DataFrame: ...
    def display_location_stats(df: pd.DataFrame, column: str) -> pd.DataFrame: ...
    def display_location_and_room_stats(df: pd.DataFrame, column: str) -> pd.DataFrame: ...

**Notebook 4 — Modeling & tool**

    # sklearn ColumnTransformer + Pipeline for numeric/categorical features.
    # RandomForestRegressor / GradientBoostingRegressor baselines.
    # Keras dense regressor with MAE loss and Adam optimizer.

    def predict_rent(input_data: dict|pd.DataFrame) -> np.ndarray: ...

---
