library(tidyverse)
library(ppsr)
library(fs)
library(glue)
library(ggplot2)
library(viridis)

log_msg <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  message(glue("{timestamp} - {msg}"))
}

root_dir <- "C:/Users/rache/OneDrive/Documents/mcmurdo"
base_out <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/pps_output"
out_dir  <- file.path(base_out, "ppsNoSampleID")
dir_create(out_dir)

# Clear output directory
existing_files <- dir_ls(out_dir, recurse = FALSE)
if (length(existing_files) > 0) file_delete(existing_files)  

files <- c(
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-bee-20250304.csv",
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-et-20250305.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_rowbind_shared.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_full_join.csv"
)


get_mode <- function(x) {
  ux <- unique(na.omit(x))
  ux[which.max(tabulate(match(x, ux)))]
}


one_hot_encode <- function(df) {
  cats <- df %>% select(where(is.character)) %>% select(where(~ n_distinct(.) > 1))
  nums <- df %>% select(where(is.numeric))
  if (ncol(cats) > 0) {
    names(cats) <- make.names(names(cats))
    dummies <- model.matrix(~ . - 1, data = cats) %>% as_tibble()
    bind_cols(nums, dummies)
  } else nums
}

for (path in files) {
  base <- tools::file_path_sans_ext(basename(path))
  label <- case_when(
    str_detect(base, "soil-bee") ~ "BEE Dataset",
    str_detect(base, "soil-et")  ~ "ET Dataset",
    base == "bee_et_rowbind_shared" ~ "Row-bind Combined",
    base == "bee_et_full_join"      ~ "Full-join Combined",
    TRUE ~ base
  )
  log_msg(glue("Reading {basename(path)}"))
  df_raw <- tryCatch(
    read_csv(path, show_col_types = FALSE, guess_max = 100000),
    error = function(e) { log_msg(glue("Failed {basename(path)}: {e$message}")); NULL }
  )
  if (is.null(df_raw)) next
  
  # date and time is split into component parts to help standardize it
  dt_cols <- names(df_raw)[sapply(df_raw, inherits, what = c("POSIXct","POSIXlt")) | str_detect(names(df_raw), "date|time")]
  for (col in dt_cols) {
    parsed <- parse_date_time(df_raw[[col]], orders = c("Ymd HMS","Ymd HM","ymd HMS","ymd HM","Ymd","ymd"), quiet = TRUE)
    df_raw <- df_raw %>%
      mutate(
        !!paste0(col, "_year")  := year(parsed),
        !!paste0(col, "_month") := month(parsed),
        !!paste0(col, "_day")   := day(parsed),
        !!paste0(col, "_hour")  := hour(parsed)
      ) %>%
      select(-all_of(col))
  }
  
  
  na_ratio <- colSums(is.na(df_raw)) / nrow(df_raw)
  df_raw <- df_raw %>% select(names(na_ratio[na_ratio <= 0.5]))
  df_raw <- df_raw %>%
    mutate(across(where(is.numeric), ~ replace_na(., median(., na.rm = TRUE)))) %>%
    mutate(across(where(is.character), ~ replace_na(., get_mode(.))))
  
  
  df_raw <- df_raw %>% select(-matches("(?i)^sample_id"))
  
  
  df_enc <- one_hot_encode(df_raw)
  n_row <- nrow(df_enc); n_col <- ncol(df_enc)
  log_msg(glue("Prepared: {n_row} rows x {n_col} cols"))
  if (n_row < 2 || n_col < 2) { log_msg("Skipping, insufficient data"); next }
  
  log_msg("Computing PPS")
  pps_mat <- tryCatch(
    score_matrix(df_enc, seed = 1),
    error = function(e) { log_msg(glue("PPS failed: {e$message}")); NULL }
  )
  if (is.null(pps_mat)) next
  
  pps_df <- as.data.frame(pps_mat) %>%
    rownames_to_column("x") %>%
    pivot_longer(-x, names_to = "y", values_to = "pps") %>%
    filter(x != y)
  
 
  write_csv(pps_df, file.path(out_dir, paste0("pps_longNoID_", base, ".csv")))
  write_csv(pps_df %>% arrange(desc(pps)), file.path(out_dir, paste0("pps_sortedNoID_", base, ".csv")))
  
  top <- pps_df %>% slice_max(pps, n = 1)
  low <- pps_df %>% slice_min(pps, n = 1)
  
  cap <- glue(
    "This heatmap visualizes PPS among {n_col} features across {n_row} samples in '{label}'. ",
    "The feature '{top$x}' is the strongest predictor for '{top$y}' with a PPS of {round(top$pps,2)}. ",
    "The feature '{low$x}' is the weakest predictor for '{low$y}' with a PPS of {round(low$pps,2)}. ",
    "Hierarchical clustering groups similar features for easier interpretation."
  )
  title <- glue("{label} PPS Heatmap")
  
  mat <- pps_df %>% pivot_wider(names_from = x, values_from = pps) %>% column_to_rownames("y") %>% as.matrix()
  sym <- mat; sym[is.na(sym)] <- t(mat)[is.na(sym)]
  ord <- hclust(dist(sym))$order; labs <- rownames(sym)[ord]
  plot_df <- as.data.frame(as.table(sym)) %>%
    rename(y=Var1, x=Var2, pps=Freq) %>%
    mutate(
      x = factor(x, levels = labs),
      y = factor(y, levels = rev(labs))
    )
  
  g <- ggplot(plot_df, aes(x = x, y = y, fill = pps)) +
    geom_tile(width = 1.5, height = 1.5) +  # slightly smaller boxes
    scale_x_discrete(expand = expansion(add = c(0, 0))) +
    scale_y_discrete(expand = expansion(add = c(0, 0))) +
    scale_fill_viridis_c(option = "D", na.value = "grey90") +
    coord_equal() +
    theme_minimal(base_size = 14) +
    theme(
      axis.title.x    = element_text(size = 16, margin = margin(t = 10)),   # more gap above x-axis title
      axis.title.y    = element_text(size = 16, margin = margin(r = 10)),   # more gap left of y-axis title
      axis.text.x     = element_text(angle = 45, vjust=1, hjust=1, size=12, margin = margin(t = 5)),  # slight gap from tiles
      axis.text.y     = element_text(size = 12, margin = margin(r = 5)),  # slight gap from tiles
      axis.ticks      = element_blank(),   # remove ticks
      panel.grid      = element_blank(),
      plot.title      = element_text(size = 18, face = "bold"),
      plot.caption    = element_text(hjust = 0, size = 12, margin = margin(t = 2)),  # tighter to caption
      plot.margin     = margin(1, 1, 0.5, 1, "cm")  # less bottom space
    ) +
    labs(x = "Predictor", y = "Target", fill = "Predictive Power Score", title = title, caption = cap)
  
  
  ggsave(filename = file.path(out_dir, paste0("pps_heatmap_", base, ".png")), plot = g, width = 14, height = 14, units = "in", dpi = 300)
  log_msg(glue("Finished {basename(path)}"))
}

log_msg("All files processed")