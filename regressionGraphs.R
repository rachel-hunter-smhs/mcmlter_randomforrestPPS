suppressPackageStartupMessages({
  library(tidyverse)
  library(pdp)
  library(fastshap)
  library(caret)
  library(pROC)
  library(viridis)
})

root_path   <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_full"
processed   <- file.path(root_path, "processed")
targets_all <- c("total", "total_live", "total_dead")
set.seed(2025)

dataset_tag <- function(p) basename(dirname(dirname(p)))
target_tag  <- function(p) basename(dirname(p))

class_models <- list.files(file.path(root_path, "classification"),
                           pattern = "_clf\\.rds$", recursive = TRUE, full.names = TRUE)
reg_models   <- list.files(file.path(root_path, "regression"),
                           pattern = "model\\.rds$", recursive = TRUE, full.names = TRUE)

plots_made <- 0

flattify <- function(df) {
  df %>% mutate(across(where(is.list), ~ map_chr(.x, as.character)))
}

pdp_ice <- function(mod, feat, X, outdir, subtitle, ylab, pf = NULL) {
  pd  <- (if (is.null(pf))
    partial(mod, feat, train = X)
    else
      partial(mod, feat, train = X, pred.fun = pf)) |>
    as_tibble() |>
    flattify()
  
  ggsave(file.path(outdir, sprintf("pdp_%s.png", feat)),
         ggplot(pd, aes(x = !!sym(feat), y = yhat)) +
           geom_line(color = viridis(1)) +
           labs(title = paste("Partial dependence:", feat),
                subtitle = subtitle, caption = "Average marginal response",
                x = feat, y = ylab) +
           theme_minimal(base_size = 14),
         width = 8, height = 5, dpi = 300)
  
  ice <- (if (is.null(pf))
    partial(mod, feat, train = X, ice = TRUE)
    else
      partial(mod, feat, train = X, pred.fun = pf, ice = TRUE)) |>
    as_tibble() |>
    flattify() |>
    mutate(id = row_number())           # guarantee id column
  
  ggsave(file.path(outdir, sprintf("ice_%s.png", feat)),
         ggplot(ice, aes(x = !!sym(feat), y = yhat, group = id)) +
           geom_line(alpha = 0.25) +
           labs(title = paste("ICE curves:", feat),
                subtitle = subtitle, caption = "Individual trajectories",
                x = feat, y = ylab) +
           theme_minimal(base_size = 14),
         width = 8, height = 5, dpi = 300)
  
  write_csv(pd,  file.path(outdir, sprintf("pdp_data_%s.csv", feat)))
  write_csv(ice, file.path(outdir, sprintf("ice_data_%s.csv", feat)))
}

process_one <- function(model_file, mode) {
  dir      <- dirname(model_file)
  ds       <- dataset_tag(model_file)
  tgt      <- target_tag(model_file)
  traincsv <- file.path(processed, sprintf("%s_train.csv", ds))
  if (!file.exists(traincsv)) return()
  
  train <- read_csv(traincsv, show_col_types = FALSE)
  X     <- train %>% select(-any_of(targets_all))
  
  mod   <- readRDS(model_file)
  impf  <- file.path(dir, "importance.csv")
  if (!file.exists(impf)) return()
  imp   <- read_csv(impf, show_col_types = FALSE)
  
  metric <- intersect(c("%IncMSE","IncMSE","MeanDecreaseAccuracy",
                        "MeanDecreaseGini","IncNodePurity"),
                      names(imp))
  if (!length(metric)) metric <- setdiff(names(imp), "feature")[1]
  metric <- metric[1]
  
  top3 <- imp |> arrange(desc(.data[[metric]])) |> slice_head(n = 3) |> pull(feature)
  
  if (mode == "clf") {
    pf   <- function(object, newdata) predict(object, newdata, type = "prob")[,"High"]
    wrap <- pf
    ylab <- "Pr(High)"
  } else {
    pf   <- NULL
    wrap <- function(object, newdata) predict(object, newdata)
    ylab <- "Predicted value"
  }
  
  subtxt <- sprintf("%s | target: %s", ds, tgt)
  walk(top3, ~ pdp_ice(mod, .x, X, dir, subtxt, ylab, pf))
  
  shap <- explain(mod, X = X, nsim = 100, pred_wrapper = wrap) |> as_tibble()
  write_csv(shap, file.path(dir, "shap_values.csv"))
  
  shap_mean <- shap |>
    summarise(across(all_of(top3), ~ mean(abs(.x), na.rm = TRUE))) |>
    pivot_longer(everything(), names_to = "feature", values_to = "mean_abs")
  write_csv(shap_mean, file.path(dir, "shap_summary.csv"))
  
  ggsave(file.path(dir, "shap_summary.png"),
         ggplot(shap_mean, aes(reorder(feature, mean_abs), mean_abs, fill = mean_abs)) +
           geom_col() + coord_flip() + scale_fill_viridis(option = "D") +
           labs(title = "Mean |SHAP| (top 3)", subtitle = subtxt,
                caption = "Higher bars → stronger influence",
                x = "Feature", y = "Mean |SHAP|") +
           theme_minimal(base_size = 14),
         width = 8, height = 5, dpi = 300)
  
  if (mode == "clf") {
    p80 <- quantile(train[[tgt]], 0.8, na.rm = TRUE)
    for (spl in c("validation","test")) {
      path <- file.path(processed, sprintf("%s_%s.csv", ds, spl))
      if (!file.exists(path)) next
      dat  <- read_csv(path, show_col_types = FALSE)
      lab  <- factor(if_else(dat[[tgt]] >= p80, "High", "Low"),
                     levels = c("Low","High"))
      pr   <- predict(mod, dat %>% select(names(X)), type = "prob")[,"High"]
      roc_ <- roc(lab, pr, levels = c("Low","High"))
      ggsave(file.path(dir, sprintf("roc_%s.png", spl)),
             ggplot(aes(1-roc_$specificities, roc_$sensitivities)) +
               geom_line(color = viridis(1), linewidth = 1) +
               geom_abline(linetype = "dashed") +
               labs(title = "ROC", subtitle = sprintf("%s | %s | %s", ds, tgt, spl),
                    caption = paste("AUC", round(auc(roc_),3)),
                    x = "FPR", y = "TPR") +
               theme_minimal(base_size = 14),
             width = 6, height = 6, dpi = 300)
    }
  }
  plots_made <<- plots_made + 1
}

walk(class_models, process_one, mode = "clf")
walk(reg_models,   process_one, mode = "reg")

cat("plots written:", plots_made, "\n")
