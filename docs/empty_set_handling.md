# Empty Prediction Set Handling

## Overview

This document explains how conformal prediction methods handle empty prediction sets in our implementation.

## The Problem

When the calibrated threshold `tau` is very strict, it's possible that no class meets the inclusion criterion, resulting in an empty prediction set. This presents a dilemma:

1. **Keep empty**: Most honest but results in 0% coverage for those samples
2. **Force predictions**: Maintains coverage but artificially improves metrics

## Our Solution

We provide configurable behavior via `empty_set_handling` in the config:

### 1. `keep_empty` (Default - Recommended)
```yaml
empty_set_handling: 'keep_empty'
```
- **Behavior**: Empty sets remain empty
- **Coverage**: 0% for those samples
- **Set size**: 0
- **Pros**: Most honest, reveals true method performance
- **Cons**: May drastically reduce overall coverage

### 2. `add_most_likely`
```yaml
empty_set_handling: 'add_most_likely'
```
- **Behavior**: Add the highest probability class
- **Coverage**: May be correct if highest class is true class
- **Set size**: 1 (minimum possible)
- **Pros**: Maintains some coverage
- **Cons**: Artificially improves metrics, not truly "conformal"

### 3. `add_top_k`
```yaml
empty_set_handling: 'add_top_k'
empty_set_top_k: 3
```
- **Behavior**: Add top-k highest probability classes
- **Coverage**: Higher chance of including true class
- **Set size**: k
- **Pros**: Conservative approach, likely maintains coverage
- **Cons**: Inflates set sizes

## Metrics Reporting

When empty sets occur, we report:
- `empty_sets`: Count of empty prediction sets
- `empty_set_percentage`: Percentage of test samples with empty sets
- `average_set_size`: **Excludes empty sets** (honest evaluation)
- `average_set_size_with_empty`: Includes empty sets (for reference)

### Important Note on Average Set Size

The primary `average_set_size` metric **excludes empty sets** from the calculation. This provides the most honest evaluation because:
- Empty sets (size 0) would artificially lower the average
- We want to know the average size of *actual* prediction sets
- Empty sets represent complete failure and should be tracked separately

Example:
- 100 samples: 90 with size 2, 10 with size 0 (empty)
- `average_set_size` = 2.0 (calculated from 90 non-empty sets)
- `average_set_size_with_empty` = 1.8 (includes the 10 zeros)
- `empty_set_percentage` = 10%

## Why This Matters

1. **Fair Comparison**: Methods that produce empty sets shouldn't get "free" size-1 sets
2. **True Performance**: Reveals when methods fail catastrophically
3. **Research Integrity**: Honest reporting of limitations

## Recommendations

1. **For Research**: Use `keep_empty` to see true performance
2. **For Production**: Consider `add_most_likely` or `add_top_k` to maintain usability
3. **For Analysis**: Always report empty set statistics separately

## Example Configuration

```yaml
# In your dataset config (e.g., cifar100.yaml)
empty_set_handling: 'keep_empty'  # For honest evaluation

# Or for production use:
# empty_set_handling: 'add_top_k'
# empty_set_top_k: 5
```

## Implementation Details

The handling is implemented in `base_conformal_scorers.py` in the `create_prediction_sets` method of each scorer. Empty sets are detected and handled according to the configuration before returning the final prediction sets.