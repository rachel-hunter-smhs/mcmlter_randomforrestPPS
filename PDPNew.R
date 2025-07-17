suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(pdp)
  library(randomForest)
  library(pROC)
  library(PRROC)
  library(viridis)
  library(fs)
})

root           <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
processed_dir  <- file.path(root, "processed")

make_pdp <- function(model, train_df, feature, ylab_txt, title_txt, out_csv, out_png,
                     prob_flag = FALSE, which_class = NULL) {
  pd <- partial(model,
                pred.var        = feature,
                train           = train_df,
                prob            = prob_flag,
                which.class     = which_class,
                grid.resolution = 50) |>
    as_tibble()
  write_csv(pd, out_csv)
  p <- ggplot(pd, aes_string(x = feature, y = "yhat")) +
    geom_line(color = viridis::viridis(1)) +
    labs(title = title_txt, x = feature, y = ylab_txt) +
    theme_minimal(base_size = 14)
  ggsave(out_png, p, width = 8, height = 6, dpi = 300)
}

get_top_numeric_features <- function(imp_df, train_df, n = 10) {
  importance_col <- intersect(c("MeanDecreaseGini", "importance", "IncNodePurity"), names(imp_df))[1]
  if (is.null(importance_col)) return(character(0))
  top_feats <- imp_df |> arrange(desc(.data[[importance_col]])) |> slice_head(n = n) |> pull(feature)
  intersect(top_feats, names(select_if(train_df, is.numeric)))
}

class_root <- file.path(root, "classification")
if (dir.exists(class_root)) {
  for (ds_dir in dir_ls(class_root, recurse = FALSE, type = "directory")) {
    ds <- tools::file_path_sans_ext(path_file(ds_dir))
    train_csv <- file.path(processed_dir, paste0(ds, "_train.csv"))
    if (!file.exists(train_csv)) next
    train_df <- read_csv(train_csv, show_col_types = FALSE)
    
    for (target_dir in dir_ls(ds_dir, recurse = FALSE, type = "directory")) {
      target     <- path_file(target_dir)
      model_path <- file.path(target_dir, paste0(ds, "_", target, "_clf.rds"))
      if (!file.exists(model_path)) model_path <- file.path(target_dir, "model.rds")
      
      imp_path <- file.path(target_dir, paste0(ds, "_", target, "_clf_importance.csv"))
      if (!file.exists(imp_path)) imp_path <- file.path(target_dir, "importance.csv")
      
      if (!file.exists(model_path) || !file.exists(imp_path)) next
      
      model <- readRDS(model_path)
      imp   <- read_csv(imp_path, show_col_types = FALSE)
      feats <- get_top_numeric_features(imp, train_df)
      
      for (feat in feats) {
        pdp_csv <- file.path(target_dir, paste0("pdp_", feat, "_", target, ".csv"))
        pdp_png <- file.path(target_dir, paste0("pdp_", feat, "_", target, ".png"))
        make_pdp(model, train_df, feat, "P(High)",
                 paste0("Partial Dependence: ", ds, " / ", target, " / ", feat),
                 pdp_csv, pdp_png, prob_flag = TRUE, which_class = "High")
      }
    }
  }
}

reg_root <- file.path(root, "regression")
if (dir.exists(reg_root)) {
  for (ds_dir in dir_ls(reg_root, recurse = FALSE, type = "directory")) {
    ds <- tools::file_path_sans_ext(path_file(ds_dir))
    train_csv <- file.path(processed_dir, paste0(ds, "_train.csv"))
    if (!file.exists(train_csv)) next
    train_df <- read_csv(train_csv, show_col_types = FALSE)
    
    for (target_dir in dir_ls(ds_dir, recurse = FALSE, type = "directory")) {
      target     <- path_file(target_dir)
      model_path <- file.path(target_dir, paste0(ds, "_", target, "_reg.rds"))
      if (!file.exists(model_path)) model_path <- file.path(target_dir, "model.rds")
      
      imp_path <- file.path(target_dir, paste0(ds, "_", target, "_reg_importance.csv"))
      if (!file.exists(imp_path)) imp_path <- file.path(target_dir, "importance.csv")
      
      if (!file.exists(model_path) || !file.exists(imp_path)) next
      
      model <- readRDS(model_path)
      imp   <- read_csv(imp_path, show_col_types = FALSE)
      feats <- get_top_numeric_features(imp, train_df)
      
      for (feat in feats) {
        pdp_csv <- file.path(target_dir, paste0("pdp_", feat, "_", target, ".csv"))
        pdp_png <- file.path(target_dir, paste0("pdp_", feat, "_", target, ".png"))
        make_pdp(model, train_df, feat, paste0("Predicted ", target),
                 paste0("Partial Dependence: ", ds, " / ", target, " / ", feat),
                 pdp_csv, pdp_png)
      }
    }
  }
}

message("PDP generation complete")
