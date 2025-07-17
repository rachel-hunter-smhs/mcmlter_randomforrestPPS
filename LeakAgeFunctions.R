suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
  library(glue)
  library(corrplot)
})

root_dir <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
processed_dir <- path(root_dir, "processed")
diag_dir <- path(root_dir, "validationDiagnostics")

# Function to find leaky features
find_leaky_features <- function(dataset_name, target_name, threshold = 0.8) {
  
  # Load the training data
  train_file <- path(processed_dir, glue("{dataset_name}_train.csv"))
  if (!file_exists(train_file)) {
    return(tibble(dataset = dataset_name, target = target_name, issue = "file_not_found"))
  }
  
  train_data <- read_csv(train_file, show_col_types = FALSE)
  
  if (!(target_name %in% names(train_data))) {
    return(tibble(dataset = dataset_name, target = target_name, issue = "target_not_found"))
  }
  
  # Get numeric columns only
  numeric_data <- train_data %>% select(where(is.numeric))
  
  if (!(target_name %in% names(numeric_data))) {
    return(tibble(dataset = dataset_name, target = target_name, issue = "target_not_numeric"))
  }
  
  # Calculate correlations with target, handling constant features
  target_vals <- numeric_data[[target_name]]
  other_cols <- numeric_data %>% select(-all_of(target_name))
  
  # Check for constant features and remove them
  non_constant_features <- other_cols %>%
    summarise(across(everything(), ~ sd(.x, na.rm = TRUE))) %>%
    pivot_longer(everything(), names_to = "feature", values_to = "std_dev") %>%
    filter(!is.na(std_dev) & std_dev > 0) %>%
    pull(feature)
  
  if (length(non_constant_features) == 0) {
    return(tibble(dataset = dataset_name, target = target_name, issue = "all_features_constant"))
  }
  
  # Calculate correlations only for non-constant features
  filtered_cols <- other_cols %>% select(all_of(non_constant_features))
  
  correlations <- filtered_cols %>%
    summarise(across(everything(), ~ {
      corr_val <- cor(.x, target_vals, use = "pairwise.complete.obs")
      if (is.na(corr_val)) 0 else corr_val
    })) %>%
    pivot_longer(everything(), names_to = "feature", values_to = "correlation") %>%
    mutate(abs_correlation = abs(correlation)) %>%
    arrange(desc(abs_correlation))
  
  # Find suspicious features
  suspicious <- correlations %>%
    filter(!is.na(abs_correlation) & abs_correlation > threshold) %>%
    mutate(
      dataset = dataset_name,
      target = target_name,
      suspicion_level = case_when(
        abs_correlation > 0.95 ~ "EXTREME - Almost certainly leakage",
        abs_correlation > 0.9 ~ "HIGH - Very likely leakage", 
        abs_correlation > 0.8 ~ "MODERATE - Investigate further",
        TRUE ~ "LOW"
      )
    )
  
  return(suspicious)
}

# Function to check for obvious leakage patterns
check_leakage_patterns <- function(dataset_name, target_name) {
  
  train_file <- path(processed_dir, glue("{dataset_name}_train.csv"))
  if (!file_exists(train_file)) return(tibble())
  
  train_data <- read_csv(train_file, show_col_types = FALSE)
  
  if (!(target_name %in% names(train_data))) return(tibble())
  
  feature_names <- names(train_data)
  target_val <- train_data[[target_name]]
  
  # Remove rows with missing target values
  valid_rows <- !is.na(target_val)
  if (sum(valid_rows) == 0) return(tibble())
  
  train_data <- train_data[valid_rows, ]
  target_val <- target_val[valid_rows]
  
  patterns <- tibble()
  
  # Check for features that are exact matches or simple transforms
  for (col in feature_names) {
    if (col == target_name) next
    
    col_val <- train_data[[col]]
    if (!is.numeric(col_val)) next
    
    # Remove missing values for comparison
    valid_pairs <- !is.na(col_val) & !is.na(target_val)
    if (sum(valid_pairs) < 10) next  # Need at least 10 valid pairs
    
    col_clean <- col_val[valid_pairs]
    target_clean <- target_val[valid_pairs]
    
    # Skip if either is constant
    if (sd(col_clean) == 0 || sd(target_clean) == 0) next
    
    # Check for exact matches
    if (all(abs(col_clean - target_clean) < 1e-10)) {
      patterns <- bind_rows(patterns, tibble(
        dataset = dataset_name,
        target = target_name,
        feature = col,
        pattern = "EXACT_MATCH",
        description = "Feature is identical to target"
      ))
    }
    
    # Check for simple scaling (but avoid division by zero)
    if (all(target_clean != 0)) {
      ratios <- col_clean / target_clean
      if (sd(ratios, na.rm = TRUE) < 1e-10) {
        patterns <- bind_rows(patterns, tibble(
          dataset = dataset_name,
          target = target_name,
          feature = col,
          pattern = "SIMPLE_SCALE",
          description = "Feature is target multiplied by constant"
        ))
      }
    }
    
    # Check for log transforms (only if positive values)
    if (all(col_clean > 0) && all(target_clean > 0)) {
      log_corr <- tryCatch({
        cor(log(col_clean), log(target_clean), use = "complete.obs")
      }, error = function(e) NA)
      
      if (!is.na(log_corr) && log_corr > 0.95) {
        patterns <- bind_rows(patterns, tibble(
          dataset = dataset_name,
          target = target_name,
          feature = col,
          pattern = "LOG_TRANSFORM",
          description = "Feature is log-transformed target"
        ))
      }
    }
    
    # Check for ratios involving target-like terms
    if (str_detect(col, "total|live|dead") && str_detect(target_name, "total|live|dead")) {
      patterns <- bind_rows(patterns, tibble(
        dataset = dataset_name,
        target = target_name,
        feature = col,
        pattern = "SUSPICIOUS_NAME",
        description = "Feature name suggests it might be derived from target"
      ))
    }
  }
  
  return(patterns)
}

# Analyze your problematic datasets
problematic_pairs <- tibble(
  dataset = c("bee_et_rowbind_shared", "bee_et_rowbind_shared", "bee_et_rowbind_shared",
              "mcmlter-soil-bee-20250304", "mcmlter-soil-bee-20250304", "mcmlter-soil-bee-20250304",
              "mcmlter-soil-et-20250305", "mcmlter-soil-et-20250305", "mcmlter-soil-et-20250305"),
  target = c("total", "total_dead", "total_live", 
             "total", "total_dead", "total_live",
             "total", "total_dead", "total_live")
)

message("Analyzing correlations for leaky features...")
all_correlations <- map2_dfr(problematic_pairs$dataset, problematic_pairs$target, find_leaky_features)

message("Checking for obvious leakage patterns...")
all_patterns <- map2_dfr(problematic_pairs$dataset, problematic_pairs$target, check_leakage_patterns)

# Summary of worst offenders
worst_features <- all_correlations %>%
  filter(abs_correlation > 0.95) %>%
  arrange(desc(abs_correlation)) %>%
  select(dataset, target, feature, correlation, abs_correlation, suspicion_level)

# Save results
write_csv(all_correlations, path(diag_dir, "feature_correlations.csv"))
write_csv(all_patterns, path(diag_dir, "leakage_patterns.csv"))
write_csv(worst_features, path(diag_dir, "worst_leaky_features.csv"))

# Print summary
message("\n=== LEAKAGE ANALYSIS SUMMARY ===")
message(glue("Total features analyzed: {nrow(all_correlations)}"))
message(glue("Features with correlation > 0.95: {sum(all_correlations$abs_correlation > 0.95, na.rm = TRUE)}"))
message(glue("Features with correlation > 0.9: {sum(all_correlations$abs_correlation > 0.9, na.rm = TRUE)}"))
message(glue("Features with correlation > 0.8: {sum(all_correlations$abs_correlation > 0.8, na.rm = TRUE)}"))

if (nrow(worst_features) > 0) {
  message("\n=== TOP LEAKY FEATURES ===")
  worst_features %>%
    slice_head(n = 10) %>%
    pwalk(~ message(glue("{..1}/{..2}: '{..3}' (r={round(..4, 3)}) - {..6}")))
}

if (nrow(all_patterns) > 0) {
  message("\n=== LEAKAGE PATTERNS DETECTED ===")
  all_patterns %>%
    count(pattern, sort = TRUE) %>%
    pwalk(~ message(glue("{..1}: {..2} features")))
}

message(glue("\nDetailed results saved to: {diag_dir}"))
message("\nNEXT STEPS:")
message("1. Check 'worst_leaky_features.csv' for the most problematic features")
message("2. Remove or fix these features in your data preparation")
message("3. Re-run your models after cleaning")
message("4. Look for features that are calculated FROM your target variables")