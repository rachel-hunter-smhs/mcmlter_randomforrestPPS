suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
})

root_dir    <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
results_dir <- path(root_dir, "results_tables")
issues_dir  <- path(root_dir, "validationDiagnostics")
dir_create(issues_dir, recurse = TRUE)


classification <- read_csv(path(results_dir, "classification_diagnostics_table.csv"), show_col_types = FALSE)
regression     <- read_csv(path(results_dir, "regression_diagnostics_table.csv"),     show_col_types = FALSE)


split_complete <- classification %>%
  count(dataset, target, name = "n_class_models") %>%
  filter(n_class_models != 1) %>%
  mutate(issue = "wrong_number_of_class_models")


metric_range <- regression %>%
  mutate(
    train_rmse = as.numeric(train_rmse),
    test_rmse  = as.numeric(test_rmse),
    train_r2   = as.numeric(train_r2),
    test_r2    = as.numeric(test_r2)
  ) %>%
  filter(
    train_rmse <= 0 |
      test_rmse  <= 0 |
      train_r2   < 0 |
      train_r2   > 1 |
      test_r2    < 0 |
      test_r2    > 1
  ) %>%
  mutate(issue = "invalid_regression_metrics")


paired_sets <- full_join(
  classification %>% distinct(dataset, target) %>% mutate(source = "class"),
  regression     %>% distinct(dataset, target) %>% mutate(source = "reg"),
  by = c("dataset", "target")
) %>%
  group_by(dataset, target) %>%
  filter(n() != 2) %>%
  ungroup() %>%
  mutate(issue = "missing_in_class_or_reg")

# Combine all issues
extra_issues <- bind_rows(split_complete, metric_range, paired_sets)

if (nrow(extra_issues) > 0) {
  write_csv(extra_issues, path(issues_dir, "extra_validation_issues.csv"))
  message("Extra issues detected; see extra_validation_issues.csv")
} else {
  message("No additional issues detected")
}
