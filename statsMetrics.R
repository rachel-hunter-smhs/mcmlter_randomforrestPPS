suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
  library(glue)
  library(viridis)
})

root_dir   <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_full"
results_dir <- path(root_dir, "results_tables")
plots_dir   <- path(root_dir, "results_plots")
dir_create(results_dir, recurse = TRUE)
dir_create(plots_dir,   recurse = TRUE)
dir_ls(results_dir, glob = "*.csv") %>% walk(file_delete)
dir_ls(plots_dir,   glob = "*.png") %>% walk(file_delete)

read_metrics <- function(f) {
  d <- read_csv(f, show_col_types = FALSE, progress = FALSE)
  if (all(c("Metric", "Value") %in% names(d))) {
    d <- d %>%
      filter(!is.na(Metric)) %>%
      mutate(Metric = str_replace_all(Metric, "\\s+", "_")) %>%
      group_by(Metric) %>%
      summarise(Value = first(na.omit(Value)), .groups = "drop") %>%
      pivot_wider(names_from = Metric, values_from = Value)
  }
  d
}

collect_metrics <- function(pattern) {
  files <- dir_ls(root_dir, recurse = TRUE, type = "file", glob = pattern)
  if (length(files) == 0) return(tibble())
  map_dfr(files, function(f) {
    read_metrics(f) %>%
      mutate(dataset = path_file(path_dir(path_dir(f))),
             target  = path_file(path_dir(f))) %>%
      relocate(dataset, target)
  })
}

numericize <- function(df) df %>% mutate(across(-c(dataset, target), ~ suppressWarnings(as.numeric(.x))))

reg <- numericize(collect_metrics("*reg_metrics_*.csv")) %>% filter(if_any(-c(dataset, target), ~ !is.na(.x)))
clf <- numericize(collect_metrics("*clf_metrics_*.csv")) %>% filter(if_any(-c(dataset, target), ~ !is.na(.x)))

if (nrow(reg) > 0) write_csv(reg, path(results_dir, "regression_metrics_table.csv"))
if (nrow(clf) > 0) write_csv(clf, path(results_dir, "classification_metrics_table.csv"))

plot_metric <- function(df, metric, prefix) {
  tryCatch({
    d <- df %>% filter(!is.na(.data[[metric]]))
    if (nrow(d) == 0) return(invisible())
    p <- ggplot(
      d,
      aes(fct_rev(fct_inorder(interaction(dataset, target, sep = " / "))), .data[[metric]])
    ) +
      geom_col(fill = viridis(1)) +
      coord_flip() +
      labs(
        title = glue("{metric} Across Datasets / Targets"),
        x     = "Dataset / Target",
        y     = metric
      ) +
      theme_minimal(base_size = 12) +
      theme(axis.text.y = element_text(size = 8))
    ggsave(path(plots_dir, glue("{prefix}_{metric}.png")), p, width = 8, height = 6, dpi = 300)
  }, error = function(e) {
    message(glue("Error plotting {metric}: {e$message}"))
  })
}

if (nrow(reg) > 0) walk(names(reg)[-(1:2)],  ~ plot_metric(reg, .x, "reg"))
if (nrow(clf) > 0) walk(names(clf)[-(1:2)], ~ plot_metric(clf, .x, "clf"))
