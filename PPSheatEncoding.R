library(tidyverse)
library(ppsr)
library(fs)
library(glue)
library(ggplot2)
library(viridis)
library(lubridate)

log_msg <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(glue("{timestamp} - {msg}"))
}

files <- c(
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-bee-20250304.csv",
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-et-20250305.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_rowbind_shared.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_full_join.csv"
)
out_dir <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/pps_output/ppsNoSampleID"
dir_create(out_dir)

get_mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}

one_hot_encode <- function(df) {
  cats <- df %>% select(where(is.character)) %>% select(where(~ n_distinct(.) > 1))
  nums <- df %>% select(where(is.numeric))
  if (ncol(cats) > 0) {
    names(cats) <- make.names(names(cats))
    model.matrix(~ . - 1, data = cats) %>% as_tibble() %>% bind_cols(nums, .)
  } else {
    nums
  }
}

for (fpath in files) {
  base  <- tools::file_path_sans_ext(basename(fpath))
  label <- case_when(
    str_detect(base, "soil-bee")    ~ "BEE Dataset",
    str_detect(base, "soil-et")     ~ "ET Dataset",
    base == "bee_et_rowbind_shared" ~ "Row-bind Combined",
    base == "bee_et_full_join"      ~ "Full-join Combined",
    TRUE                            ~ base
  )
  log_msg(glue("Reading {fpath}"))
  df_raw <- tryCatch(read_csv(fpath, show_col_types = FALSE, guess_max = 100000), error = function(e) NULL)
  if (is.null(df_raw)) next
  dt_cols <- names(df_raw)[sapply(df_raw, inherits, c("POSIXct","POSIXlt")) | str_detect(names(df_raw), "date|time")]
  for (col in dt_cols) {
    parsed <- parse_date_time(df_raw[[col]], orders = c("Ymd HMS","Ymd HM","ymd HMS","ymd HM","Ymd","ymd"), quiet = TRUE)
    df_raw <- df_raw %>% mutate(!!paste0(col, "_year") := year(parsed), !!paste0(col, "_month") := month(parsed), !!paste0(col, "_day") := day(parsed), !!paste0(col, "_hour") := hour(parsed)) %>% select(-all_of(col))
  }
  na_ratio <- colSums(is.na(df_raw)) / nrow(df_raw)
  df_raw   <- df_raw %>% select(names(na_ratio[na_ratio <= 0.5])) %>% mutate(across(where(is.numeric), ~ replace_na(., median(., na.rm = TRUE)))) %>% mutate(across(where(is.character), ~ replace_na(., get_mode(.)))) %>% select(-matches("(?i)^sample_id"))
  df_enc   <- one_hot_encode(df_raw)
  n_row    <- nrow(df_enc)
  n_col    <- ncol(df_enc)
  log_msg(glue("Prepared: {n_row}×{n_col}"))
  if (n_row < 2 || n_col < 2) next
  log_msg("Computing PPS")
  pps_mat <- tryCatch(score_matrix(df_enc, seed = 1), error = function(e) NULL)
  if (is.null(pps_mat)) next
  pps_df <- as.data.frame(pps_mat) %>% rownames_to_column("x") %>% pivot_longer(-x, names_to = "y", values_to = "pps") %>% filter(x != y)
  write_csv(pps_df,      file.path(out_dir, paste0("pps_longNoID_", base, ".csv")))
  write_csv(pps_df %>% arrange(desc(pps)), file.path(out_dir, paste0("pps_sortedNoID_", base, ".csv")))
  mat    <- pps_df %>% pivot_wider(names_from = x, values_from = pps) %>% column_to_rownames("y") %>% as.matrix()
  sym    <- mat
  sym[is.na(sym)] <- t(mat)[is.na(mat)]
  ord    <- hclust(dist(sym))$order
  labs   <- rownames(sym)[ord]
  plot_df <- as.data.frame(as.table(sym)) %>% rename(y = Var1, x = Var2, pps = Freq) %>% mutate(x = factor(x, levels = labs), y = factor(y, levels = rev(labs)))
  g <- ggplot(plot_df, aes(x, y, fill = pps)) +
    geom_tile() +
    scale_x_discrete(expand = expansion(add = c(0, 0))) +
    scale_y_discrete(expand = expansion(add = c(0, 0))) +
    scale_fill_viridis_c(option = "D", na.value = "grey90") +
    coord_equal() +
    theme_minimal(base_size = 14) +
    theme(
      axis.title.x = element_text(size = 16),
      axis.title.y = element_text(size = 16),
      axis.text.x  = element_text(angle = 45, hjust = 1, vjust = 1, size = 12, margin = ggplot2::margin(t = 4, r = 0, b = 0, l = 0)),
      axis.text.y  = element_text(size = 12, margin = ggplot2::margin(t = 0, r = -5, b = 0, l = 0)),
      axis.ticks   = element_blank(),
      panel.grid   = element_blank(),
      plot.title   = element_text(size = 18, face = "bold"),
      plot.margin  = ggplot2::margin(t = 1, r = 1, b = 0.5, l = 1, unit = "cm")
    ) +
    labs(title = glue("{label} PPS Heatmap"), x = "Predictor", y = "Target", fill = "PPS")
  ggsave(filename = file.path(out_dir, paste0("pps_heatmap_", base, ".png")), plot = g, width = 14, height = 14, units = "in", dpi = 300)
  log_msg(glue("Finished {fpath}"))
}
log_msg("All files processed")
