# Clean Profile Creation Documentation

## Workflow

![Workflow](wf.png)


## Overview

The profiling module builds statistical profiles for dataset columns using only clean (non-error) data. These profiles capture syntactic patterns, semantic characteristics, and contextual relationships.


## Core Data Structure: ColumnProfile

The `ColumnProfile` dataclass stores statistics for a single column:

### Basic Metadata
- `table_id`: Table identifier
- `column_idx`: Column index
- `column_name`: Column name

### Cardinality Statistics
- `uniqueness`: Ratio of unique values to total clean cells
  ```
  uniqueness = |distinct_values| / |clean_cells|
  ```

### Value Distribution
- `value_histogram`: Frequency distribution of actual values (only for non-unique columns)
- Numeric statistics (for numeric columns):
  - `min_value`, `max_value`: Range bounds
  - `q1`, `q2`, `q3`: Quartiles (25th, 50th, 75th percentiles)
  - `mean_value`: Arithmetic mean
  - `std_value`: Standard deviation

### Pattern Analysis
- `mask_histogram`: Unicode category patterns → frequency
- `inferred_data_type`: DBMS-specific type (`tinyint`, `decimal`, `varchar(255)`, etc.)

### Length Statistics
- `length_min`, `length_max`: Character length bounds
- `length_median`: Median character length
- `length_mu`: Mean character length
- `length_sigma`: Standard deviation of character lengths

### Numeric Format Constraints
- `max_digits`: Maximum number of digits observed
- `max_decimals`: Maximum decimal places observed
- `has_negatives`: Whether negative numbers are present
- `has_scientific`: Whether scientific notation is used

### Semantic Profiles
- `clean_value_embeddings`: Sentence transformer embeddings for semantic similarity
- `clean_char_ngrams`: Character 2-grams
- `functional_dependencies`: FDs where this column is the right-hand side

## Deviation Scores

### Pattern Unusualness Score
The system calculates how unusual a value's pattern is:

```
pattern_unusualness = 1 - P(pattern | clean_data)

where P(pattern | clean_data) = count(pattern) / total_patterns
```

### Length Deviation Score
Measures how far a value's length deviates from the clean distribution:

```
length_deviation = min(|length - μ_length| / (3 * σ_length), 1.0)

where μ_length = mean length of clean values
      σ_length = standard deviation of clean lengths
```

### Value Unusualness Score
For categorical columns, measures how rare a specific value is:

```
value_unusualness = 1 - P(value | clean_data)

where P(value | clean_data) = count(value) / total_clean_values
```

### Character N-gram Overlap
Measures string similarity based on character subsequences:

```
ngram_overlap = |ngrams(dirty_value) ∩ ngrams(clean_values)| / |ngrams(dirty_value)|
```

### Minimum Edit Distance
Uses Levenshtein distance for string similarity:

```
min_edit_distance = min(levenshtein(dirty_value, clean_value) for clean_value in domain)
normalized_distance = min_edit_distance / max(len(dirty_value), len(clean_value))
```

### Functional Dependency Violation Score
Measures violations of discovered functional dependencies where this column is the right-hand side:

```
fd_violation_score = max(confidence_i × violation_i for all FDs involving this column)

where:
- confidence_i = strength of functional dependency i (from TANE algorithm)
- violation_i = 1 if FD is violated, 0 if satisfied
- FD violation occurs when: LHS_values → expected_RHS ≠ actual_RHS
```

### Minimum Embedding Distance
Measures semantic distance to clean domain values using sentence transformers:

min_embedding_distance = min(cosine_distance(emb_dirty, emb_clean_i) for all clean embeddings)

where:
- cosine_distance = 1 - cosine_similarity
- cosine_similarity = (A · B) / (||A|| × ||B||)
- emb_dirty = embedding of dirty value
- emb_clean_i = embedding of clean value i


### Examples 

- Before - Influnece Based Method 

| Table ID | Column | Row | Dirty Value | Clean Value | Error Type |
|----------|--------|-----|-------------|-------------|------------|
| 8cb42c4711339949a412581261ee0b33 | 10 | 121 | *(empty)* | WI | Missing Value - Semantic |
| ea276d3ae1a300422acd31920fbebc7b | 3 | 53 | *(empty)* | 8:00 a.m. | Missing Value - Semantic |
| ea276d3ae1a300422acd31920fbebc7b | 4 | 756 | *(empty)* | 1:32 p.m. | Missing Value - Semantic |
| ea276d3ae1a300422acd31920fbebc7b | 5 | 53 | *(empty)* | 11:55 a.m. | Missing Value - Semantic |
| ea276d3ae1a300422acd31920fbebc7b | 6 | 756 | *(empty)* | 5:34 p.m. | Missing Value - Semantic |
| ca123f33da87365bc92d637016370eaf | 7 | 17 | 7.0 | 7 | Representation |
| ca123f33da87365bc92d637016370eaf | 1 | 60 | 2014 2015 2016 | 2015 | REP/ Semantic |
| 8b42a1c9b8f9fde869f83c954b3d463b | 13 | 157 | yxs | yes | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 11 | 413 | acuxe care hospixals | acute care hospitals | Random |
| 8cb42c4711339949a412581261ee0b33 | 9 | 1430 | Portland OR | Portland | Semantic / Rep |
| 8b42a1c9b8f9fde869f83c954b3d463b | 17 | 411 | 1xx% | 100% | Random |
| ca123f33da87365bc92d637016370eaf | 8 | 1312 | 21,203 | 21203 | REP |
| 8b42a1c9b8f9fde869f83c954b3d463b | 12 | 332 | proxrietary | proprietary | Random |
| ca123f33da87365bc92d637016370eaf | 0 | 426 | DÃ©jÃ\xa0 Vu | Deja Vu | Unicode char error |
| 8b42a1c9b8f9fde869f83c954b3d463b | 2 | 532 | baptist medical cexter south | baptist medical center south | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 14 | 256 | xneumonia | pneumonia | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 16 | 532 | pxeumoxia patiexts assessed... | pneumonia patients assessed... | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 18 | 532 | 87 patientx | 87 patients | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 19 | 213 | al_sxip-inf-2 | al_scip-inf-2 | Random |
| 8b42a1c9b8f9fde869f83c954b3d463b | 15 | 101 | hfx1 | hf-1 | Random |


- Now - Dirtiness Profiles - Priority-based sampling 

| Table ID | Column | Row | Dirty Value | Ground Truth | Error Type |
|----------|--------|-----|-------------|--------------|------------|
| 8cb42c4711339949a412581261ee0b33 | 10 | 20 | *(empty)* | CA | Missing value - Semantic |
| 8cb42c4711339949a412581261ee0b33 | 4 | 1 | 12.0 oz. | 12 | REP |
| ca123f33da87365bc92d637016370eaf | 8 | 279 | 14,503 | 14503 | REP |
| ca123f33da87365bc92d637016370eaf | 8 | 591 | 8,179 | 8179 | REP |
| ea276d3ae1a300422acd31920fbebc7b | 4 | 73 | 3:04 p.m. | 2:55 p.m. | Semantic |
| ca123f33da87365bc92d637016370eaf | 8 | 118 | 129,025 | 129025 | REP |
| 8cb42c4711339949a412581261ee0b33 | 6 | 0 | N/A | *(empty)* | REP |
| ca123f33da87365bc92d637016370eaf | 7 | 17 | 7.0 | 7 | REP |
| 8cb42c4711339949a412581261ee0b33 | 5 | 100 | 0.055999999999999994% | 0.056 | REP |
| ea276d3ae1a300422acd31920fbebc7b | 6 | 514 | 12:12 p.m. | 11:56 a.m. | Semantic |
| 8b42a1c9b8f9fde869f83c954b3d463b | 11 | 280 | acutexcarexhospitals | acute care hospitals | Random |
| ea276d3ae1a300422acd31920fbebc7b | 3 | 991 | Not Available | 8:00 a.m. | Semantic |
| 8cb42c4711339949a412581261ee0b33 | 5 | 44 | 0.055% | 0.055 | Rep |
| 8b42a1c9b8f9fde869f83c954b3d463b | 10 | 384 | 25x47x7xx0 | 2514717110 | Random |
| ea276d3ae1a300422acd31920fbebc7b | 5 | 1665 | 10:15 p.m. | 10:30 p.m. | Semantic |
| 8b42a1c9b8f9fde869f83c954b3d463b | 13 | 418 | xo | no | Random |
| ca123f33da87365bc92d637016370eaf | 6 | 4467 | 1 hr. 32 min. | 81 min | Semantic |
| ea276d3ae1a300422acd31920fbebc7b | 6 | 95 | 3:04 p.m. | 3:16 p.m. | Semantic |
| 8b42a1c9b8f9fde869f83c954b3d463b | 17 | 411 | 1xx% | 100% | Random |
| 62ed597b338095651312245f2063ef2a | 6 | 532 | *(empty)* | -1 | REP |
| ca123f33da87365bc92d637016370eaf | 8 | 4779 | 7,611 | 7611 | REP |


- Now - Dirtiness Profiles - Representative sampling 

| Table ID | Column | Row | Dirty Value | Ground Truth | Error Type |
|----------|--------|-----|-------------|--------------|------------|
| ca123f33... | 14 | 4806 | Action & Adventure,Science Fiction ... | Action,Sci-Fi | Semantic |
| 8cb42c47... | 9 | 1558 | Longmont CO | Longmont | Semantic / REP|
| ea276d3a... | 4 | 1495 | 11:41 a.m. | 11:40 a.m. | Semantic |
| 62ed597b... | 8 | 416 | 2/1/13 | 1/13/02 | REP |
| 62ed597b... | 8 | 749 | 5/1/11 | 1/11/05 | REP |
| 62ed597b... | 7 | 532 | *(empty)* | -1 | Missing Value |
| 8b42a1c9... | 15 | 580 | px-5c | pn-5c | Random |
| ca123f33... | 3 | 5978 | Sep 8, 2007 Wide | 22 February 2008 (USA) | Semantic |
| 8cb42c47... | 5 | 608 | 0.07% | 0.07 | REP |
| ca123f33... | 11 | 3634 | 7.0 | 7 | REP |
| 8b42a1c9... | 7 | 559 | xl | al | Random |
| ea276d3a... | 6 | 1025 | Dec 02 Contact Airline | 12:21 a.m. | Semantic |
| ea276d3a... | 3 | 50 | *(empty)* | 7:10 a.m. | Missing Value |
| ea276d3a... | 4 | 128 | *(empty)* | 2:48 p.m. | Missing Value |
| ea276d3a... | 6 | 1179 | 7:55 p.m. | 8:03 p.m. | Semantic |
| ea276d3a... | 5 | 1747 | Fri Dec 2 5:11 a.m. | 5:11 a.m. | REP |
| ea276d3a... | 4 | 384 | 2:04 p.m. | 2:10 p.m. | Semantic |
| 8cb42c47... | 4 | 0 | 12.0 oz | 12 | REP |
| 8cb42c47... | 4 | 1682 | 12.0 oz. | 12 | REP |
| 8cb42c47... | 6 | 0 | N/A | *(empty)* | REP |
| 8cb42c47... | 5 | 6 | 0.045% | 0.045 | REP |