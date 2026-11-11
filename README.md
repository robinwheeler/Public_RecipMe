# RecipMe

**A constraint-aware retrieval-augmented generation (RAG) system for recipe generation.**

RecipMe is designed for home cooks with dietary requirements. It combines rule-based (symbolic) constraint filtering with neural retrieval, ranking and generation, so that allergen and dietary restrictions are enforced at the retrieval stage rather than left to the language model. The project also evaluates how well an LLM-as-a-judge agrees with human evaluation of generated recipes.

> **Status:** proof of concept. RecipMe is **not** a safe, final product. Always check generated recipes yourself before cooking, especially if you have allergies. See [Limitations](#limitations).

---

## Contents

- [How it works](#how-it-works)
- [Repository contents](#repository-contents)
- [Data](#data)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results summary](#results-summary)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## How it works

RecipMe runs as a five-stage pipeline: representation, retrieval, ranking, generation and evaluation.

### 1. Representation
- The `epi_r` dataset is cleaned: nutrition columns are converted to numbers, missing values set to zero, and low-calorie or incomplete recipes (e.g. sauces) removed to encourage safe eating and reduce noise.
- Non-ingredient tags (locations, holidays, equipment, etc.) are removed using an exclusion list.
- Each recipe is consolidated into a single string: `Title | nutrition (macros) | Features: ingredient/dish-type tags`.
- These strings are embedded with the Sentence-BERT model `multi-qa-MiniLM-L6-cos-v1` and cached to Google Drive.
- A vocabulary of known ingredients (`known_ingredients.json`) is built once and cached.

### 2. Retrieval
- The user enters a natural-language query (ingredients, dish type, nutritional goals), then separately states dietary restrictions.
- Restrictions are expanded via a manual allergen map (e.g. `gluten` → wheat, barley, rye, bread, flour…) and any recipe containing a prohibited term is filtered out.
- If no recipes remain, the full dataset is used instead (the notebook flags this rather than failing silently).
- The query is embedded and cosine similarity retrieves the **top 20** candidates.

### 3. Ranking
1. **Multi-criterion scoring (MCS)** boosts or penalises candidates based on keywords in the query:
   - taste words (`delicious`, `flavourful`…) boost high-rated recipes
   - `protein`, `muscle`, `bulking` boost high-protein recipes
   - `diet`, `low calorie`, `healthy` penalise high-calorie recipes
2. A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) computes fine-grained relevance.
3. Scores are combined: `0.4 × MCS + 0.6 × cross-encoder`.
4. The top 3 are re-ranked by **ingredient overlap** with the query, with the combined score used as a tie-breaker. The winner is carried forward.

### 4. Generation
- The title of the selected recipe is matched against `RecipeNLG` titles (cosine similarity) to retrieve the 3 most similar recipes.
- These are used as few-shot examples in a structured prompt to `meta-llama/Llama-3.2-1B-Instruct`, which writes the final recipe (title, ingredients, step-by-step directions) using the selected recipe's ingredients and respecting the dietary exclusions.

### 5. Evaluation
- Generated recipes are scored (1–5) on five criteria: **fluency, relevance, correctness/safety, structure, taste**.
- Scoring is done by both an LLM-as-a-judge and the human user.
- Users can also rewrite the recipe; original/improved pairs are saved, creating a corpus that could be used for fine-tuning.

---

## Repository contents

| File | Description |
|---|---|
| `RecipMe.ipynb` | The RecipMe model: cleaning, embedding, retrieval, ranking, generation and evaluation. Google Colab notebook. |
| `analysis_of_data.ipynb` | Script that analyses RecipMe's results (human vs LLM agreement). Google Colab notebook. |
| `epi_r.csv` | **Clipped** version of the Epicurious dataset (full version on Kaggle). |
| `RecipeNLG.csv` | **Clipped** version of the RecipeNLG dataset (full version on Kaggle). |
| `human_evaluated_recipes.json` | 102 previously evaluated recipes with human scores on the 5 criteria, plus human-improved versions of each recipe. |
| `llm_evaluated_recipes.json` | The same recipes scored by the LLM-as-a-judge on the 5 criteria. |

---

## Data

The two source datasets were clipped to fit size limits. For better results, download the full versions:

- **epi_r.csv** – [Epicurious recipes](https://www.kaggle.com/datasets/hugodarwood/epirecipes): titles, nutritional metadata (rating, calories, protein, fat, sodium) and binary ingredient/feature tags.
- **RecipeNLG.csv** – [RecipeNLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg): natural-language recipes with ingredients and directions, used for few-shot generation.

---

## Setup

RecipMe is built for **Google Colab** and Google Drive.

1. **Upload the data files to the root of your Google Drive ("My Drive")** so they follow the path `/content/drive/MyDrive/` once mounted:
   - `epi_r.csv`
   - `RecipeNLG.csv`
   - `human_evaluated_recipes.json`
   - `llm_evaluated_recipes.json`
2. **Create an `Embeddings` folder** at `/content/drive/MyDrive/Embeddings/`. Cached embeddings are stored here:
   - `paragraph_embeddings.pt`
   - `title_embeddings.pt`
   - `parsed_recipe_dataset.pkl`
3. **Open `RecipMe.ipynb` in Google Colab** (and `analysis_of_data.ipynb` for the analysis).
4. **Install dependencies** if they are not already available in your runtime:
   ```bash
   pip install pandas numpy torch transformers sentence-transformers scipy
   ```
5. **Hugging Face access for Llama.** `meta-llama/Llama-3.2-1B-Instruct` is a gated model. Request access on its [Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and log in within Colab (e.g. `huggingface-cli login` or a Colab secret token).
6. A GPU runtime is recommended (Runtime → Change runtime type).

> The first run is slow: it builds the ingredient vocabulary and embeds the datasets. Subsequent runs load the cached files from Drive.

---

## Usage

Run the notebook cells top to bottom:

1. **Mount Drive and clean data.** Authorise Google Drive access when prompted.
2. **Embed recipes** (cached after the first run).
3. **Enter your query.** You will be prompted twice:
   ```
   Please tell me any ingredients you would like to use, the type of dish you would like
   and any nutritional goals you want to adhere to:
   > high protein chicken and rice, something delicious

   Please state all dietary preferences you have (e.g. peanut, gluten-free, vegan). If none, leave blank or say no:
   > dairy, nuts
   ```
   - Separate multiple restrictions with commas.
   - Supported restriction keywords: `nuts`, `peanut`, `gluten`, `dairy`, `soy`, `egg`, `shellfish`, `fish`, `sesame`, `corn`, `mustard`, `sulfites`, `vegan`, `vegetarian`. Unrecognised terms are used as literal exclusion words.
   - Useful query keywords for nutritional goals: `delicious` / `flavourful` (favour high ratings), `protein` / `muscle` / `bulking` (favour high protein), `diet` / `low calorie` / `healthy` (favour lower calories).
4. **Generate the recipe.** The final recipe is printed.
5. **Evaluate (optional).** The LLM judge scores the recipe automatically, then you are asked for 1–5 scores for fluency, relevance, correctness/safety, structure and taste, and invited to paste an improved version of the recipe (finish with an empty line). Results are appended to the JSON files on your Drive under an auto-incremented `recipe_id`.

### Example output

```
Grilled Shrimp with Roasted Garlic-Herb Sauce

Ingredients:
- 1 lb. large shrimp, peeled and deveined
- 4 cloves garlic, minced
...
Directions:
1. Preheat grill to medium-high heat.
...
```

---

## Evaluation

Run `analysis_of_data.ipynb` in Colab (with the evaluation JSON files in `MyDrive`) to compute agreement between human and LLM-as-a-judge scores using:

- **Pearson** correlation (linear agreement)
- **Spearman** rank correlation (rank-order consistency)
- **Kendall's τ** (pairwise ranking agreement)

computed with `scipy.stats` for each of the five criteria.

Both the human and the LLM judge use the same five criteria:

| Criterion | Meaning |
|---|---|
| Fluency | Clarity and readability |
| Relevance | Matches the query and ingredients |
| Correctness | Safe and logically correct cooking steps |
| Structure | Clear ingredients list and step-by-step format |
| Taste | Likelihood of tasting good |

---

## Results summary

- **Constraint compliance:** strong within the scope of the allergen mapping. Violations occurred in a minority of cases, mostly with composite or semantically complex ingredients (e.g. soy sauce, sourdough, praline), which reflects gaps in symbolic coverage rather than model reasoning.
- **LLM vs human agreement:** weak to modest. Correlations were low across all criteria (Spearman ≈ 0.09–0.23), highest for relevance and near zero for structure. The LLM-as-a-judge is not a reliable substitute for human evaluation in this domain.

| Metric | Pearson | Spearman | Kendall's τ |
|---|---|---|---|
| Fluency | 0.0526 | 0.0952 | 0.0899 |
| Relevance | 0.1545 | 0.2292 | 0.2134 |
| Correctness | 0.1690 | 0.1437 | 0.1271 |
| Structure | 0.0113 | 0.0250 | 0.0225 |
| Taste | 0.0903 | 0.1081 | 0.0949 |

These figures come from a limited sample (n≈100) and should be interpreted cautiously.

---

## Limitations

- **Allergen safety is only as complete as the manual mapping.** The map is a proof of concept and misses composite ingredients. Do not rely on it for real allergy management.
- **Small generator.** The 1B-parameter Llama model can hallucinate ingredient substitutions, misalign with the desired dish (ingredient adherence is prioritised over dish type), and produce structural or duplication errors, all delivered confidently.
- **Minimum calorie filter** removes low-calorie recipes by design.
- **Small evaluation sample** and a lightweight LLM judge (which can occasionally fail to produce valid JSON).
- Keyword-based MCS and the allergen map would need expanding to scale.

---

## Future work

- Ablation study (e.g. removing MCS or ingredient overlap).
- Expand symbolic constraints with knowledge graphs of food and ingredient substitutions.
- Post-generation validation and structure enforcement.
- Direct preference optimisation (DPO) fine-tuning using original vs human-improved recipe pairs, or LLM pairwise ranking.
- Persistent user profiles for allergens and nutritional goals.

---

## Acknowledgements

- **Datasets:** [Epicurious (epi_r)](https://www.kaggle.com/datasets/hugodarwood/epirecipes) and [RecipeNLG](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) (Bień et al., 2020).
- **Models:** [`multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1), [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
- Developed as a report for INST0100, Department of Information Studies, University College London.
