suppressPackageStartupMessages({
  library(tidyverse)
  library(randomForest)
  library(caret)
  library(pROC)
  library(PRROC)
  library(janitor)
  library(fastDummies)
  library(fs)
  library(viridis)
  library(glue)
  library(SHAPforxgboost)
  library(shapr)
  library(fastshap)
})

root          <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
processed_dir <- path(root, "processed")
class_root    <- path(root, "classification")
reg_root      <- path(root, "regression")

log_skip  <- function(p) message("exists   :", p)
log_write <- function(p) message("created  :", p)

write_csv_safe <- function(d, p) {
  if (file_exists(p)) log_skip(p) else { write_csv(d, p); log_write(p) }
}

save_plot <- function(o, p, w = 7, h = 6) {
  if (file_exists(p)) log_skip(p) else { ggsave(p, o, width = w, height = h, dpi = 300); log_write(p) }
}

scat <- function(d, x, y, t, p) {
  save_plot(
    ggplot(d, aes(.data[[x]], .data[[y]])) +
      geom_point(alpha = .4, size = 1, color = viridis(1)) +
      geom_smooth(method = "lm", linewidth = .8, se = FALSE, color = viridis(1)) +
      labs(title = t, x = "Observed", y = "Predicted") +
      theme_minimal(base_size = 14),
    p
  )
}

residplot <- function(d, t, p) {
  save_plot(
    ggplot(d, aes(fitted, residual)) +
      geom_point(alpha = .4, size = 1, color = viridis(1)) +
      geom_smooth(method = "loess", linewidth = .8, se = FALSE, color = viridis(1)) +
      labs(title = t, x = "Fitted", y = "Residual") +
      theme_minimal(base_size = 14),
    p
  )
}

histplot <- function(v, t, p) {
  save_plot(
    ggplot(tibble(value = v), aes(value)) +
      geom_histogram(bins = 30, fill = viridis(1), color = "black") +
      labs(title = t, x = "Probability of High", y = "Count") +
      theme_minimal(base_size = 14),
    p
  )
}

lineplot <- function(d, x, y, t, xl, yl, p) {
  save_plot(
    ggplot(d, aes(.data[[x]], .data[[y]])) +
      geom_line(linewidth = .8, color = viridis(1)) +
      labs(title = t, x = xl, y = yl) +
      theme_minimal(base_size = 14),
    p
  )
}

barplot <- function(d, t, xl, yl, p) {
  save_plot(
    ggplot(d, aes(reorder(feature, importance), importance)) +
      geom_col(fill = viridis(1)) +
      coord_flip() +
      labs(title = t, x = xl, y = yl) +
      theme_minimal(base_size = 14),
    p
  )
}

heatplot <- function(d, t, p) {
  save_plot(
    ggplot(d, aes(row, col, fill = value)) +
      geom_tile() +
      scale_fill_viridis_c(limits = c(-1, 1)) +
      labs(title = t, fill = "r") +
      theme_minimal(base_size = 14) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title  = element_blank()
      ),
    p, 10, 10
  )
}

oobplot <- function(v, t, p) {
  save_plot(
    ggplot(tibble(tree = seq_along(v), error = v), aes(tree, error)) +
      geom_line(linewidth = .8, color = viridis(1)) +
      labs(title = t, x = "Tree", y = "OOB Error") +
      theme_minimal(base_size = 14),
    p
  )
}


create_shap_plot <- function(model, data, target, model_type, title, output_path) {
  tryCatch({
    
    if (nrow(data) > 100) {
      set.seed(2025)
      sample_idx <- sample(nrow(data), 100)
      data_sample <- data[sample_idx, ]
    } else {
      data_sample <- data
    }
    
    feature_data <- data_sample %>% 
      select(-any_of(c("total", "total_live", "total_dead", "label", "target_val", "class_obs")))
    
    
    if (ncol(feature_data) > 0 && any(sapply(feature_data, is.numeric))) {
      
      if (model_type == "classification") {
        pred_func <- function(x) {
          preds <- predict(model, x, type = "prob")
          if (ncol(preds) >= 2) preds[, 2] else preds[, 1]
        }
      } else {
        pred_func <- function(x) predict(model, x)
      }
      
      shap_values <- fastshap::explain(
        model, 
        X = feature_data,
        pred_wrapper = pred_func,
        nsim = 50
      )
      
      shap_df <- as.data.frame(shap_values)
      shap_importance <- shap_df %>%
        summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) %>%
        pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") %>%
        arrange(desc(mean_abs_shap)) %>%
        slice_head(n = 20)
      
      shap_plot <- ggplot(shap_importance, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
        geom_col(fill = viridis(1)) +
        coord_flip() +
        labs(
          title = paste0("SHAP Feature Importance - ", title),
          x = "Feature",
          y = "Mean |SHAP value|"
        ) +
        theme_minimal(base_size = 14)
      
      save_plot(shap_plot, output_path)
      
      shap_csv_path <- str_replace(output_path, "\\.png$", "_values.csv")
      write_csv_safe(shap_importance, shap_csv_path)
      
    } else {
      message("Skipping SHAP plot - no numeric features available")
    }
  }, error = function(e) {
    message(paste("SHAP plot failed:", e$message))
  })
}

ensure_reg <- function(ds, target, reg, df, tag, dirp) {
  pred_df <- df %>% select(-any_of(c("total", "total_live", "total_dead")))
  
  obs   <- df[[target]]
  preds <- predict(reg, pred_df)
  res   <- obs - preds
  
  mape <- mean(abs(res / if_else(obs == 0, NA_real_, obs)), na.rm = TRUE) * 100
  
  met <- tibble(
    RMSE = sqrt(mean(res^2, na.rm = TRUE)),
    MAE  = mean(abs(res), na.rm = TRUE),
    MAPE = mape,
    R2   = cor(obs, preds, use = "complete.obs")^2
  )
  
  write_csv_safe(met, path(dirp, paste0("reg_metrics_", tag, ".csv")))
  
  scat(tibble(obs, preds), "obs", "preds",
       paste0("Predicted vs Observed (", tag, ") – ", ds, " / ", target),
       path(dirp, paste0("pred_vs_actual_", tag, ".png")))
  
  residplot(tibble(fitted = preds, residual = res),
            paste0("Residuals vs Fitted (", tag, ") – ", ds, " / ", target),
            path(dirp, paste0("residuals_vs_fitted_", tag, ".png")))
}

ensure_clf <- function(ds, target, clf, df, tag, dirp, p80) {
  dfp <- df %>%
    filter(!is.na(.data[[target]])) %>%
    mutate(class_obs = factor(if_else(.data[[target]] >= p80, "High", "Low")))
  
  pred_df <- dfp %>% select(-any_of(c("total", "total_live", "total_dead", "class_obs")))
  
  probs_all <- predict(clf, pred_df, type = "prob")[, "High"]
  keep_idx  <- !is.na(probs_all)
  probs     <- probs_all[keep_idx]
  dfp       <- dfp[keep_idx, , drop = FALSE]
  
  preds <- factor(if_else(probs >= .5, "High", "Low"), levels = c("Low", "High"))
  
  cm <- confusionMatrix(preds, dfp$class_obs, positive = "High")
  
  sens <- cm$byClass["Sensitivity"]
  spec <- cm$byClass["Specificity"]
  prec <- cm$byClass["Pos Pred Value"]
  f1   <- 2 * prec * sens / (prec + sens)
  bal  <- (sens + spec) / 2
  
  met <- tibble(
    Accuracy          = cm$overall["Accuracy"],
    Kappa             = cm$overall["Kappa"],
    Sensitivity       = sens,
    Specificity       = spec,
    Precision         = prec,
    F1                = f1,
    Balanced_Accuracy = bal,
    AUC               = auc(roc(dfp$class_obs, probs, levels = c("Low", "High")))
  )
  
  write_csv_safe(met, path(dirp, paste0("clf_metrics_", tag, ".csv")))
  
  histplot(probs,
           paste0("Predicted Probabilities (", tag, ") – ", ds, " / ", target),
           path(dirp, paste0("prob_hist_", tag, ".png")))
  
  cm_tbl <- as_tibble(as.table(cm$table))
  colnames(cm_tbl) <- c("Prediction", "Reference", "Count")
  
  cmplot <- function(tb, t, p) {
    save_plot(
      ggplot(tb, aes(Prediction, Reference, fill = Count)) +
        geom_tile() +
        geom_text(aes(label = Count), color = "white", size = 4) +
        scale_fill_viridis_c() +
        labs(title = t, x = "Predicted", y = "Actual", fill = "Count") +
        theme_minimal(base_size = 14),
      p
    )
  }
  
  cmplot(cm_tbl,
         paste0("Confusion Matrix (", tag, ") – ", ds, " / ", target),
         path(dirp, paste0("conf_matrix_", tag, ".png")))
  
  roc_obj <- roc(dfp$class_obs, probs, levels = c("Low", "High"))
  roc_df  <- tibble(fpr = rev(roc_obj$specificities), tpr = rev(roc_obj$sensitivities))
  
  write_csv_safe(roc_df, path(dirp, paste0("roc_curve_", tag, ".csv")))
  
  lineplot(roc_df, "fpr", "tpr",
           paste0("ROC Curve (", tag, ") – ", ds, " / ", target),
           "FPR", "TPR",
           path(dirp, paste0("roc_", tag, ".png")))
  
  pr <- pr.curve(
    scores.class0 = probs[dfp$class_obs == "High"],
    scores.class1 = probs[dfp$class_obs == "Low"],
    curve = TRUE
  )
  
  pr_df <- tibble(recall = pr$curve[, 1], precision = pr$curve[, 2])
  
  write_csv_safe(pr_df, path(dirp, paste0("pr_curve_", tag, ".csv")))
  
  lineplot(pr_df, "recall", "precision",
           paste0("PR Curve (", tag, ") – ", ds, " / ", target),
           "Recall", "Precision",
           path(dirp, paste0("pr_curve_", tag, ".png")))
}

for (ds_dir in dir_ls(reg_root, recurse = FALSE, type = "directory")) {
  ds <- tools::file_path_sans_ext(path_file(ds_dir))
  message(glue("Processing regression dataset: {ds}"))
  
  
  train_file <- path(processed_dir, paste0(ds, "_train.csv"))
  valid_file <- path(processed_dir, paste0(ds, "_validation.csv"))
  test_file  <- path(processed_dir, paste0(ds, "_test.csv"))
  
  if (!file_exists(train_file) || !file_exists(valid_file) || !file_exists(test_file)) {
    message(glue("  Skipping {ds} - missing data files"))
    next
  }
  
  train_df <- read_csv(train_file, show_col_types = FALSE)
  valid_df <- read_csv(valid_file, show_col_types = FALSE)
  test_df  <- read_csv(test_file, show_col_types = FALSE)
  
  for (reg_dir in dir_ls(ds_dir, recurse = FALSE, type = "directory")) {
    target <- path_file(reg_dir)
    message(glue("  Processing regression target: {target}"))
    
    
    model_path <- path(reg_dir, "model.rds")
    if (!file_exists(model_path)) {
      model_path <- path(reg_dir, paste0(ds, "_", target, "_reg.rds"))
    }
    
    if (!file_exists(model_path)) {
      message(glue("    Skipping - no model file found"))
      next
    }
    
    reg <- readRDS(model_path)
    
    ensure_reg(ds, target, reg, valid_df, "validation", reg_dir)
    ensure_reg(ds, target, reg, test_df,  "test", reg_dir)
    
  
    if (!file_exists(path(reg_dir, "oob_error.png"))) {
      if ("finalModel" %in% names(reg) && !is.null(reg$finalModel$mse)) {
        oobplot(reg$finalModel$mse, paste0("OOB MSE – ", ds, " / ", target), path(reg_dir, "oob_error.png"))
      } else if (!is.null(reg$mse)) {
        oobplot(reg$mse, paste0("OOB MSE – ", ds, " / ", target), path(reg_dir, "oob_error.png"))
      }
    }
    
   
    imp_csv <- path(reg_dir, "importance.csv")
    if (!file_exists(imp_csv)) {
    
      imp_csv <- path(reg_dir, paste0(ds, "_", target, "_reg_importance.csv"))
    }
    
    if (file_exists(imp_csv)) {
      imp_df <- read_csv(imp_csv, show_col_types = FALSE)
  
      if (ncol(imp_df) >= 2) {
        colnames(imp_df)[1:2] <- c("feature", "importance")
      }
      
      barplot(
        slice_max(imp_df, importance, n = 20),
        paste0("Feature Importance – ", ds, " / ", target),
        "Feature", "IncNodePurity",
        path(reg_dir, "importance_plot.png")
      )
      
      corr_png <- path(reg_dir, "correlation_heatmap.png")
      if (!file_exists(corr_png)) {
        top_feats <- slice_max(imp_df, importance, n = 20) %>% pull(feature)
        num_feats <- intersect(top_feats, names(select_if(train_df, is.numeric)))
        
        if (length(num_feats) > 1) {
          corr_mat <- cor(select(train_df, all_of(num_feats)), use = "pairwise.complete.obs")
          
          corr_long <- corr_mat %>%
            as.data.frame() %>%
            rownames_to_column("row") %>%
            pivot_longer(-row, names_to = "col", values_to = "value")
          
          heatplot(corr_long,
                   paste0("Correlation Heatmap – ", ds, " / ", target),
                   corr_png)
        }
      }
      
      shap_path <- path(reg_dir, "shap_importance.png")
      if (!file_exists(shap_path)) {
        create_shap_plot(reg, test_df, target, "regression", 
                         paste0(ds, " / ", target), shap_path)
      }
    }
  }
}

for (ds_dir in dir_ls(class_root, recurse = FALSE, type = "directory")) {
  ds <- tools::file_path_sans_ext(path_file(ds_dir))
  message(glue("Processing classification dataset: {ds}"))
  
  
  train_file <- path(processed_dir, paste0(ds, "_train.csv"))
  valid_file <- path(processed_dir, paste0(ds, "_validation.csv"))
  test_file  <- path(processed_dir, paste0(ds, "_test.csv"))
  
  if (!file_exists(train_file) || !file_exists(valid_file) || !file_exists(test_file)) {
    message(glue("  Skipping {ds} - missing data files"))
    next
  }
  
  train_df <- read_csv(train_file, show_col_types = FALSE)
  valid_df <- read_csv(valid_file, show_col_types = FALSE)
  test_df  <- read_csv(test_file, show_col_types = FALSE)
  
  for (clf_dir in dir_ls(ds_dir, recurse = FALSE, type = "directory")) {
    target <- path_file(clf_dir)
    message(glue("  Processing classification target: {target}"))
    
  
    model_path <- path(clf_dir, "model.rds")
    if (!file_exists(model_path)) {
      model_path <- path(clf_dir, paste0(ds, "_", target, "_clf.rds"))
    }
    
    if (!file_exists(model_path)) {
      message(glue("    Skipping - no model file found"))
      next
    }
    
    clf <- readRDS(model_path)
    
   
    p80 <- quantile(train_df[[target]], 0.75, na.rm = TRUE)
    
    ensure_clf(ds, target, clf, valid_df, "validation", clf_dir, p80)
    ensure_clf(ds, target, clf, test_df,  "test", clf_dir, p80)
    
    if (!file_exists(path(clf_dir, "oob_error.png"))) {
      if ("finalModel" %in% names(clf) && !is.null(clf$finalModel$err.rate)) {
        oobplot(clf$finalModel$err.rate[, "OOB"],
                paste0("OOB Error – ", ds, " / ", target),
                path(clf_dir, "oob_error.png"))
      } else if (!is.null(clf$err.rate)) {
        oobplot(clf$err.rate[, "OOB"],
                paste0("OOB Error – ", ds, " / ", target),
                path(clf_dir, "oob_error.png"))
      }
    }
    
    imp_csv <- path(clf_dir, "importance.csv")
    if (!file_exists(imp_csv)) {
      
      imp_csv <- path(clf_dir, paste0(ds, "_", target, "_clf_importance.csv"))
    }
    
    if (file_exists(imp_csv)) {
      imp_df <- read_csv(imp_csv, show_col_types = FALSE)
      
      if (ncol(imp_df) >= 2) {
        colnames(imp_df)[1:2] <- c("feature", "importance")
      }
      
      barplot(
        slice_max(imp_df, importance, n = 20),
        paste0("Feature Importance – ", ds, " / ", target),
        "Feature", "MeanDecreaseGini",
        path(clf_dir, "importance_plot.png")
      )
      
      
      corr_png <- path(clf_dir, "correlation_heatmap.png")
      if (!file_exists(corr_png)) {
        top_feats <- slice_max(imp_df, importance, n = 20) %>% pull(feature)
        num_feats <- intersect(top_feats, names(select_if(train_df, is.numeric)))
        
        if (length(num_feats) > 1) {
          corr_mat <- cor(select(train_df, all_of(num_feats)), use = "pairwise.complete.obs")
          
          corr_long <- corr_mat %>%
            as.data.frame() %>%
            rownames_to_column("row") %>%
            pivot_longer(-row, names_to = "col", values_to = "value")
          
          heatplot(corr_long,
                   paste0("Correlation Heatmap – ", ds, " / ", target),
                   corr_png)
        }
      }
      
     
      shap_path <- path(clf_dir, "shap_importance.png")
      if (!file_exists(shap_path)) {
        create_shap_plot(clf, test_df, target, "classification", 
                         paste0(ds, " / ", target), shap_path)
      }
    }
  }
}

message("Diagnostics pass finished")