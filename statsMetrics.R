suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
  library(glue)
  library(viridis)
})

root_dir          <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
class_dir         <- path(root_dir, "classification")
reg_dir           <- path(root_dir, "regression")
plot_tables_dir   <- path(root_dir, "plot_tables")
plot_results_dir  <- path(root_dir, "plot_results")

walk(c(plot_tables_dir, plot_results_dir), \(d) {
  if (dir_exists(d)) dir_delete(d)
  dir_create(d, recurse = TRUE)
})

read_diag <- \(f) readr::read_csv(f, show_col_types = FALSE) %>%
  mutate(dataset = path_file(path_dir(path_dir(f))),
         target  = path_file(path_dir(f))) %>%
  relocate(dataset, target)

collect_diag <- \(base, pattern) {
  files <- dir_ls(base, recurse = TRUE, glob = pattern)
  if (length(files) == 0) tibble() else map_dfr(files, read_diag)
}

clf_diag <- collect_diag(class_dir, "*clf_metrics_*.csv")
reg_diag <- collect_diag(reg_dir,   "*reg_metrics_*.csv")

numericize <- \(df, cols)
if (nrow(df) == 0) df else df %>% mutate(across(any_of(cols), \(x) suppressWarnings(as.numeric(x))))

if (nrow(clf_diag) > 0) {
  clf_diag <- numericize(clf_diag, c("Accuracy","Kappa","Sensitivity","Specificity",
                                     "Precision","F1","Balanced_Accuracy","AUC"))
  write_csv(clf_diag, path(plot_tables_dir, "classification_metrics.csv"))
}

if (nrow(reg_diag) > 0) {
  reg_diag <- numericize(reg_diag, c("RMSE","MAE","MAPE","R2"))
  write_csv(reg_diag, path(plot_tables_dir, "regression_metrics.csv"))
}

plot_metric <- \(df, metric, prefix) {
  if (!(metric %in% names(df))) return()
  dm <- df %>% filter(!is.na(.data[[metric]]))
  if (nrow(dm) == 0) return()
  p <- ggplot(dm, aes(fct_rev(fct_inorder(interaction(dataset, target, sep = " / "))),
                      .data[[metric]])) +
    geom_col(fill = viridis(1)) +
    coord_flip() +
    labs(title = glue("{metric} by dataset and target"),
         x = "Dataset / target",
         y = metric) +
    theme_minimal(base_size = 13) +
    theme(axis.text.y = element_text(size = 8))
  ggsave(path(plot_results_dir, glue("{prefix}_{metric}.png")), p, width = 8, height = 6, dpi = 300)
}

plot_overview <- \(df, metrics, prefix, file_out) {
  dl <- df %>%
    pivot_longer(all_of(metrics), names_to = "metric", values_to = "value") %>%
    filter(!is.na(value))
  if (nrow(dl) == 0) return()
  p <- ggplot(dl, aes(fct_rev(fct_inorder(interaction(dataset, target, sep = " / "))),
                      value, fill = metric)) +
    geom_col(position = "dodge") +
    coord_flip() +
    scale_fill_viridis_d() +
    labs(title = glue("{prefix} metrics overview"),
         x = "Dataset / target",
         y = "Metric value",
         fill = "Metric") +
    theme_minimal(base_size = 13) +
    theme(axis.text.y = element_text(size = 8))
  ggsave(path(plot_results_dir, file_out), p, width = 10, height = 8, dpi = 300)
}

if (nrow(clf_diag) > 0) {
  clf_all   <- c("Accuracy","Kappa","Sensitivity","Specificity",
                 "Precision","F1","Balanced_Accuracy","AUC")
  clf_part1 <- c("Accuracy","Balanced_Accuracy","Kappa","AUC")
  clf_part2 <- c("Sensitivity","Specificity","Precision","F1")
  walk(clf_all, \(m) plot_metric(clf_diag, m, "clf"))
  plot_overview(clf_diag, clf_all,   "Classification",        "clf_overview_all.png")
  plot_overview(clf_diag, clf_part1, "Classification part 1", "clf_overview_part1.png")
  plot_overview(clf_diag, clf_part2, "Classification part 2", "clf_overview_part2.png")
}

if (nrow(reg_diag) > 0) {
  reg_mets <- c("RMSE","MAE","MAPE","R2")
  walk(reg_mets, \(m) plot_metric(reg_diag, m, "reg"))
  plot_overview(reg_diag, reg_mets, "Regression", "reg_overview.png")
}
