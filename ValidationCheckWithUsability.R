suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
  library(glue)
})

root_dir <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
processed_dir <- path(root_dir, "processed")
results_dir <- path(root_dir, "results_tables")
diag_dir <- path(root_dir, "validationDiagnostics")


classification_metrics <- read_csv(path(results_dir, "classification_diagnostics_table.csv"), show_col_types = FALSE)
regression_metrics <- read_csv(path(results_dir, "regression_diagnostics_table.csv"), show_col_types = FALSE)


clf_numeric_cols <- intersect(names(classification_metrics), c("Accuracy", "Balanced_Accuracy", "AUC", "F1"))
if (length(clf_numeric_cols) > 0) {
  classification_metrics <- classification_metrics %>%
    mutate(across(all_of(clf_numeric_cols), as.numeric))
}

reg_numeric_cols <- intersect(names(regression_metrics), c("RMSE", "MAE", "R2"))
if (length(reg_numeric_cols) > 0) {
  regression_metrics <- regression_metrics %>%
    mutate(across(all_of(reg_numeric_cols), as.numeric))
}

find_balance <- function(df, target_name, threshold) {
  min(mean(df[[target_name]] >= threshold), mean(df[[target_name]] < threshold))
}

find_corr_leak <- function(df, target_name) {
  num_df <- df %>% select(where(is.numeric))
  if (!(target_name %in% names(num_df))) return(0)
  cor_vec <- suppressWarnings(abs(cor(num_df[[target_name]], num_df %>% select(-all_of(target_name)), use = "pairwise.complete.obs")))
  max(cor_vec, na.rm = TRUE)
}

detailed_evaluate <- function(ds, tg) {
  split_files <- tibble(
    split = c("train", "validation", "test"),
    f = path(processed_dir, glue("{ds}_{split}.csv"))
  ) %>% filter(file_exists(f))
  
  if (nrow(split_files) < 3) {
    return(tibble(
      dataset = ds,
      target = tg,
      issue = "missing_splits",
      detail = "Not all train/validation/test splits found"
    ))
  }
  
  data_list <- split_files %>% mutate(data = map(f, ~ read_csv(.x, show_col_types = FALSE)))
  train_df <- data_list$data[[1]]
  valid_df <- data_list$data[[2]]
  test_df <- data_list$data[[3]]
  
  if (!(tg %in% names(train_df))) {
    return(tibble(
      dataset = ds,
      target = tg,
      issue = "missing_target",
      detail = glue("Target '{tg}' not found in data")
    ))
  }
  
  
  n_train <- nrow(train_df)
  n_valid <- nrow(valid_df)
  n_test <- nrow(test_df)
  
  p80 <- quantile(train_df[[tg]], 0.8, na.rm = TRUE)
  min_balance <- min(
    find_balance(train_df, tg, p80),
    find_balance(valid_df, tg, p80),
    find_balance(test_df, tg, p80)
  )
  
  corr_leak <- find_corr_leak(train_df, tg)
  
  reg_row <- regression_metrics %>% filter(dataset == ds, target == tg)
  clf_row <- classification_metrics %>% filter(dataset == ds, target == tg)
  
  reg_gap <- if (nrow(reg_row) == 2) abs(diff(reg_row$RMSE)) / mean(reg_row$RMSE) else NA_real_
  
  
  issues <- tibble()
  
 
  if (nrow(reg_row) == 2 && any(reg_row$R2 > 0.99, na.rm = TRUE)) {
    max_r2 <- max(reg_row$R2, na.rm = TRUE)
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "perfect_r2",
      detail = glue("R² = {round(max_r2, 4)} (> 0.99) - likely data leakage")
    ))
  }
  
  
  if (!is.na(reg_gap) && reg_gap > 0.1) {
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "rmse_gap",
      detail = glue("RMSE gap = {round(reg_gap, 3)} (> 0.1) - overfitting")
    ))
  }
  
  
  if (corr_leak > 0.95) {
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "correlation_leak",
      detail = glue("Max correlation = {round(corr_leak, 3)} (> 0.95) - data leakage")
    ))
  }
  
  
  if (min_balance < 0.1) {
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "class_imbalance",
      detail = glue("Min class balance = {round(min_balance, 3)} (< 0.1) - severe imbalance")
    ))
  }
  
  if (n_train < 100 || n_test < 50) {
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "small_sample",
      detail = glue("Train: {n_train}, Test: {n_test} - small sample size")
    ))
  }
  
  if (nrow(clf_row) == 2) {
    if ("Accuracy" %in% names(clf_row)) {
      max_acc <- max(clf_row$Accuracy, na.rm = TRUE)
      if (max_acc > 0.98) {
        issues <- bind_rows(issues, tibble(
          dataset = ds,
          target = tg,
          issue = "perfect_accuracy",
          detail = glue("Max accuracy = {round(max_acc, 3)} (> 0.98) - likely overfitting")
        ))
      }
    }
  }
  
  
  if (nrow(reg_row) == 2 && any(reg_row$R2 > 0.95 & reg_row$R2 <= 0.99, na.rm = TRUE)) {
    high_r2 <- max(reg_row$R2[reg_row$R2 > 0.95 & reg_row$R2 <= 0.99], na.rm = TRUE)
    issues <- bind_rows(issues, tibble(
      dataset = ds,
      target = tg,
      issue = "high_r2",
      detail = glue("R² = {round(high_r2, 4)} (0.95-0.99) - suspiciously high")
    ))
  }
  
  
  if (nrow(issues) == 0) {
    issues <- tibble(
      dataset = ds,
      target = tg,
      issue = "none",
      detail = "No issues detected"
    )
  }
  
  return(issues)
}


pairs <- bind_rows(
  classification_metrics %>% distinct(dataset, target),
  regression_metrics %>% distinct(dataset, target)
) %>% distinct()


message("Running detailed diagnostics...")
detailed_issues <- map2_dfr(pairs$dataset, pairs$target, detailed_evaluate)


issue_summary <- detailed_issues %>%
  filter(issue != "none") %>%
  count(issue, sort = TRUE) %>%
  mutate(
    description = case_when(
      issue == "perfect_r2" ~ "R² > 0.99 - Data leakage",
      issue == "correlation_leak" ~ "Feature correlation > 0.95 - Data leakage",
      issue == "rmse_gap" ~ "RMSE gap > 10% - Overfitting",
      issue == "class_imbalance" ~ "Class balance < 10% - Severe imbalance",
      issue == "perfect_accuracy" ~ "Accuracy > 98% - Overfitting",
      issue == "high_r2" ~ "R² 0.95-0.99 - Suspiciously high",
      issue == "small_sample" ~ "Small sample size",
      TRUE ~ issue
    )
  )


severity_counts <- detailed_issues %>%
  group_by(dataset, target) %>%
  summarise(
    has_severe = any(issue %in% c("perfect_r2", "correlation_leak", "rmse_gap", "class_imbalance")),
    has_moderate = any(issue %in% c("small_sample", "perfect_accuracy", "high_r2")),
    has_none = all(issue == "none"),
    .groups = "drop"
  ) %>%
  summarise(
    unusable = sum(has_severe),
    caution = sum(has_moderate & !has_severe),
    usable = sum(has_none),
    total = n()
  )


write_csv(detailed_issues, path(diag_dir, "detailed_issues.csv"))
write_csv(issue_summary, path(diag_dir, "issue_summary.csv"))
write_csv(severity_counts, path(diag_dir, "severity_counts.csv"))


message("\n=== DIAGNOSTIC SUMMARY ===")
message(glue("Total models analyzed: {nrow(pairs)}"))
message(glue("Unusable (severe issues): {severity_counts$unusable}"))
message(glue("Caution (moderate issues): {severity_counts$caution}"))
message(glue("Usable (no issues): {severity_counts$usable}"))

message("\n=== TOP ISSUES ===")
issue_summary %>% 
  slice_head(n = 5) %>%
  pwalk(~ message(glue("{..1}: {..2} models - {..3}")))

message("\n=== SAMPLE PROBLEMATIC MODELS ===")
sample_issues <- detailed_issues %>%
  filter(issue != "none") %>%
  group_by(issue) %>%
  slice_head(n = 2) %>%
  ungroup()

sample_issues %>%
  pwalk(~ message(glue("{..1}/{..2}: {..3} - {..4}")))

message(glue("\nDetailed results saved to: {diag_dir}"))
message("Check 'detailed_issues.csv' for full breakdown of each model's problems.")