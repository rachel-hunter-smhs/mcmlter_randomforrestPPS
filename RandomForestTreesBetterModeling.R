suppressPackageStartupMessages({
  library(tidyverse)
  library(fastDummies)
  library(randomForest)
  library(caret)
  library(pROC)
  library(janitor)
  library(fs)
  library(viridis)
  library(glue)
})
new_root        <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
processed_dir   <- path(new_root, "processed")
class_dir       <- path(new_root, "classification")
reg_dir         <- path(new_root, "regression")
tables_dir      <- path(new_root, "results_tables")
diagnostics_dir <- path(new_root, "diagnostics")
for (d in c(processed_dir, class_dir, reg_dir, tables_dir, diagnostics_dir)) {
  if (dir_exists(d)) try(fs::dir_delete(d), silent = TRUE)
  dir_create(d, recurse = TRUE)
}
files   <- c(
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-bee-20250304.csv",
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-et-20250305.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_rowbind_shared.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_full_join.csv"
)
targets <- c("total","total_live","total_dead")
set.seed(2025)
tmedian <- function(x) {
  x_clean <- x[!is.na(x) & is.finite(x)]
  if(length(x_clean) == 0) return(0)
  median(x_clean)
}
mymode <- function(x) { 
  ux <- na.omit(x)
  if(length(ux) == 0) return(NA_character_)
  mode_val <- names(which.max(table(ux)))
  if(is.null(mode_val)) return(NA_character_)
  mode_val
}
safe_var <- function(x) {
  x_clean <- x[!is.na(x) & is.finite(x)]
  if(length(x_clean) <= 1) return(0)
  if(length(unique(x_clean)) <= 1) return(0)
  var(x_clean)
}
detect_leakage <- function(df, target, threshold = 0.95) {
  nums <- df %>% select(where(is.numeric)) %>% names() %>% setdiff(target)
  zero_var <- nums[sapply(df[nums], safe_var) == 0]
  valid <- setdiff(nums, zero_var)
  high_cor <- character(0)
  if(length(valid) > 0 && target %in% names(df)) {
    target_vals <- df[[target]]
    if(sum(!is.na(target_vals) & is.finite(target_vals)) > 1) {
      cors <- sapply(valid, function(v) {
        vals <- df[[v]]
        valid_idx <- !is.na(vals) & is.finite(vals) & !is.na(target_vals) & is.finite(target_vals)
        if(sum(valid_idx) > 1) {
          abs(cor(vals[valid_idx], target_vals[valid_idx]))
        } else {
          0
        }
      })
      high_cor <- names(cors)[cors > threshold & !is.na(cors)]
    }
  }
  other_targets <- intersect(setdiff(targets, target), names(df))
  pattern_matches <- character(0)
  if(length(valid) > 0) {
    pattern_matches <- unique(c(
      valid[grepl(paste0("^", target, "$"), valid, ignore.case = TRUE)],
      valid[grepl(paste0("^", target, "_"), valid, ignore.case = TRUE)],
      valid[grepl(paste0("_", target, "$"), valid, ignore.case = TRUE)],
      valid[grepl(paste0("_", target, "_"), valid, ignore.case = TRUE)]
    ))
  }
  leaks <- unique(c(zero_var, high_cor, other_targets, pattern_matches))
  if(length(leaks) > 0.8 * length(nums)) {
    leaks <- unique(c(zero_var, names(cors)[cors > 0.99 & !is.na(cors)], other_targets))
  }
  leaks
}
evaluate_model <- function(model, train_data, test_data, target_col, type) {
  if(type == "classification") {
    train_pred <- predict(model, train_data)
    train_actual <- train_data[[target_col]]
    train_acc <- mean(train_pred == train_actual, na.rm = TRUE)
    test_pred <- predict(model, test_data)
    test_actual <- test_data[[target_col]]
    test_acc <- mean(test_pred == test_actual, na.rm = TRUE)
    train_auc <- test_auc <- NA
    tryCatch({
      train_prob <- predict(model, train_data, type = "prob")
      if(!is.null(train_prob) && ncol(train_prob) >= 2) {
        train_auc <- as.numeric(auc(roc(train_actual, train_prob[,2], quiet = TRUE)))
      }
    }, error = function(e) {})
    tryCatch({
      test_prob <- predict(model, test_data, type = "prob")
      if(!is.null(test_prob) && ncol(test_prob) >= 2) {
        test_auc <- as.numeric(auc(roc(test_actual, test_prob[,2], quiet = TRUE)))
      }
    }, error = function(e) {})
    list(
      train_accuracy = train_acc, test_accuracy = test_acc,
      train_auc = train_auc, test_auc = test_auc,
      type = type
    )
  } else {
    train_pred <- predict(model, train_data)
    train_actual <- train_data[[target_col]]
    train_valid <- !is.na(train_pred) & !is.na(train_actual) & is.finite(train_pred) & is.finite(train_actual)
    test_pred <- predict(model, test_data)
    test_actual <- test_data[[target_col]]
    test_valid <- !is.na(test_pred) & !is.na(test_actual) & is.finite(test_pred) & is.finite(test_actual)
    if(!any(train_valid) || !any(test_valid)) {
      return(list(train_rmse = NA, test_rmse = NA, train_r2 = NA, test_r2 = NA, 
                  train_mae = NA, test_mae = NA, type = type))
    }
    train_rmse <- sqrt(mean((train_pred[train_valid] - train_actual[train_valid])^2))
    test_rmse <- sqrt(mean((test_pred[test_valid] - test_actual[test_valid])^2))
    train_r2 <- cor(train_pred[train_valid], train_actual[train_valid])^2
    test_r2 <- cor(test_pred[test_valid], test_actual[test_valid])^2
    train_mae <- mean(abs(train_pred[train_valid] - train_actual[train_valid]))
    test_mae <- mean(abs(test_pred[test_valid] - test_actual[test_valid]))
    list(
      train_rmse = train_rmse, test_rmse = test_rmse,
      train_r2 = train_r2, test_r2 = test_r2,
      train_mae = train_mae, test_mae = test_mae,
      type = type
    )
  }
}
diagnose_model <- function(metrics, leaks, n_features) {
  issues <- character(0)
  severity <- "usable"
  if(length(leaks) > 0) {
    issues <- c(issues, paste("Data leakage:", length(leaks), "features removed"))
    if(length(leaks) > n_features * 0.5) {
      severity <- "unusable"
    } else {
      severity <- "caution"
    }
  }
  if(metrics$type == "classification") {
    train_acc <- metrics$train_accuracy
    test_acc <- metrics$test_accuracy
    if(!is.na(train_acc) && train_acc > 0.99) {
      issues <- c(issues, "Perfect training accuracy - likely overfitting")
      severity <- "unusable"
    }
    if(!is.na(train_acc) && !is.na(test_acc)) {
      acc_gap <- train_acc - test_acc
      if(acc_gap > 0.15) {
        issues <- c(issues, paste("Large accuracy gap:", round(acc_gap, 3)))
        if(severity != "unusable") severity <- "caution"
      }
    }
    if(!is.na(test_acc) && test_acc < 0.6) {
      issues <- c(issues, "Poor test accuracy - model may not be learning")
      if(severity != "unusable") severity <- "caution"
    }
  } else {
    train_r2 <- metrics$train_r2
    test_r2 <- metrics$test_r2
    train_rmse <- metrics$train_rmse
    test_rmse <- metrics$test_rmse
    if(!is.na(train_r2) && train_r2 > 0.99) {
      issues <- c(issues, "Perfect R² - likely overfitting")
      severity <- "unusable"
    }
    if(!is.na(train_r2) && !is.na(test_r2)) {
      r2_gap <- train_r2 - test_r2
      if(r2_gap > 0.2) {
        issues <- c(issues, paste("Large R² gap:", round(r2_gap, 3)))
        if(severity != "unusable") severity <- "caution"
      }
    }
    if(!is.na(train_rmse) && !is.na(test_rmse) && train_rmse > 0) {
      rmse_inflation <- (test_rmse - train_rmse) / train_rmse
      if(rmse_inflation > 0.2) {
        issues <- c(issues, paste("RMSE inflation:", round(rmse_inflation, 3)))
        if(severity != "unusable") severity <- "caution"
      }
    }
    if(!is.na(test_r2) && test_r2 < 0.1) {
      issues <- c(issues, "Very low R² - model may not be learning")
      if(severity != "unusable") severity <- "caution"
    }
  }
  if(length(issues) == 0) issues <- "None"
  list(issues = issues, severity = severity)
}
prepare_raw <- function(path) {
  if(!file.exists(path)) return(NULL)
  df <- tryCatch({
    readr::read_csv(path, col_types = cols(.default = col_character()), 
                    na = c("", "NA", "NULL", "null", "N/A"))
  }, error = function(e) {
    message(paste("Error reading file:", path, "-", e$message))
    return(NULL)
  })
  if(is.null(df) || !nrow(df)) return(NULL)
  df <- df %>% 
    clean_names() %>% 
    select(where(~!all(is.na(.x))))
  if(ncol(df) == 0) return(NULL)
  names(df) <- make.names(names(df), unique = TRUE)
  nums <- df %>% 
    select(where(~all(is.na(.x) | str_detect(as.character(.x), "^[-+]?\\d*\\.?\\d+([eE][-+]?\\d+)?$")))) %>% 
    names()
  if(length(nums) > 0) {
    df <- df %>% mutate(across(all_of(nums), ~as.numeric(as.character(.x))))
  }
  nums_final <- df %>% select(where(is.numeric)) %>% names()
  chars <- df %>% select(where(~is.character(.x) | is.factor(.x))) %>% names()
  if(length(nums_final) > 0) {
    df <- df %>% mutate(across(all_of(nums_final), ~ifelse(is.na(.x) | !is.finite(.x), tmedian(.x), .x)))
  }
  if(length(chars) > 0) {
    keep_chars <- chars[sapply(chars, function(c) {
      n_unique <- length(unique(df[[c]][!is.na(df[[c]])]))
      n_unique > 1 && n_unique <= min(50, nrow(df) * 0.8)
    })]
    if(length(keep_chars) > 0) {
      df <- df %>% mutate(across(all_of(keep_chars), ~ifelse(is.na(.x), mymode(.x), .x)))
      df <- df %>% 
        fastDummies::dummy_cols(select_columns = keep_chars, remove_selected_columns = TRUE, 
                                remove_first_dummy = TRUE) %>%
        clean_names()
    }
    unused_chars <- setdiff(chars, keep_chars)
    if(length(unused_chars) > 0) {
      df <- df %>% select(-all_of(unused_chars))
    }
  }
  names(df) <- make.names(names(df), unique = TRUE)
  if(any(names(df) == "" | is.na(names(df)))) return(NULL)
  df
}
cv_ctrl_clf <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary, 
  savePredictions = "final",
  allowParallel = FALSE
)
cv_ctrl_reg <- trainControl(
  method = "cv", 
  number = 5, 
  savePredictions = "final",
  allowParallel = FALSE
)
all_diagnostics <- list()
all_leaks_classification <- list()
all_leaks_regression <- list()
all_performance <- list()
for(file in files) {
  message(glue("Processing {basename(file)}..."))
  df <- prepare_raw(file)
  if(is.null(df) || nrow(df) < 20) {
    message("  Skipping: insufficient data")
    next
  }
  key <- tools::file_path_sans_ext(basename(file))
  write_csv(df, path(processed_dir, paste0(key, "_processed.csv")))
  set.seed(2025)
  train_idx <- sample(nrow(df), size = floor(0.8 * nrow(df)))
  train_df <- df[train_idx, ]
  test_df <- df[-train_idx, ]
  prop_from_train <- 0.10
  prop_from_test  <- 0.10
  set.seed(2025)
  val_idx_train <- sample(nrow(train_df), size = floor(prop_from_train * nrow(train_df)))
  valid_from_train <- train_df[val_idx_train, ]
  train_df         <- train_df[-val_idx_train, ]
  val_idx_test <- sample(nrow(test_df), size = floor(prop_from_test * nrow(test_df)))
  valid_from_test <- test_df[val_idx_test, ]
  test_df         <- test_df[-val_idx_test, ]
  valid_df <- bind_rows(valid_from_train, valid_from_test)
  train_csv <- path(processed_dir, paste0(key, "_train.csv"))
  valid_csv <- path(processed_dir, paste0(key, "_validation.csv"))
  test_csv  <- path(processed_dir, paste0(key, "_test.csv"))
  write_csv(train_df, train_csv)
  write_csv(valid_df, valid_csv)
  write_csv(test_df,  test_csv)
  for(tgt in intersect(targets, names(df))) {
    message(glue("  Processing target: {tgt}"))
    target_vals <- df[[tgt]]
    target_clean <- target_vals[!is.na(target_vals) & is.finite(target_vals)]
    if(length(unique(target_clean)) < 2) {
      message("    Skipping: target has insufficient variation")
      next
    }
    leaks <- detect_leakage(df, tgt)
    feats <- setdiff(names(df), c(tgt, leaks))
    if(length(feats) < 2) {
      message("    Skipping: insufficient features after leakage removal")
      next
    }
    if(length(target_clean) >= 10) {
      q75 <- quantile(target_clean, 0.75, na.rm = TRUE)
      train_clf <- train_df %>% 
        select(all_of(feats), all_of(tgt)) %>%
        filter(!is.na(!!sym(tgt)), is.finite(!!sym(tgt))) %>%
        mutate(label = factor(if_else(!!sym(tgt) >= q75, "High", "Low"), levels = c("Low", "High"))) %>%
        select(-all_of(tgt))
      test_clf <- test_df %>% 
        select(all_of(feats), all_of(tgt)) %>%
        filter(!is.na(!!sym(tgt)), is.finite(!!sym(tgt))) %>%
        mutate(label = factor(if_else(!!sym(tgt) >= q75, "High", "Low"), levels = c("Low", "High"))) %>%
        select(-all_of(tgt))
      if(nrow(train_clf) >= 10 && nrow(test_clf) >= 5 && 
         n_distinct(train_clf$label) == 2 && n_distinct(test_clf$label) == 2) {
        class_props <- prop.table(table(train_clf$label))
        if(min(class_props) >= 0.1) {
          set.seed(2025)
          tryCatch({
            m_cv <- train(
              x = train_clf %>% select(all_of(feats)),
              y = train_clf$label,
              method = "rf",
              metric = "ROC",
              trControl = cv_ctrl_clf,
              tuneGrid = data.frame(mtry = max(1, floor(sqrt(length(feats))))),
              nodesize = max(1, floor(nrow(train_clf) * 0.05)),
              ntree = 300,
              importance = TRUE
            )
            train_clf$label <- train_clf$label
            test_clf$label <- test_clf$label
            metrics <- evaluate_model(m_cv, train_clf, test_clf, "label", "classification")
            diag <- diagnose_model(metrics, leaks, length(feats))
            key_diag <- paste(basename(file), tgt, "classification", sep = "_")
            all_diagnostics[[key_diag]] <- list(
              type = "classification", 
              diagnosis = diag, 
              metrics = metrics,
              n_features = length(feats),
              n_train = nrow(train_clf),
              n_test = nrow(test_clf)
            )
            if(length(leaks) > 0) {
              all_leaks_classification[[key_diag]] <- tibble(
                feature = leaks, 
                file = basename(file), 
                target = tgt
              )
            }
            dir_create(path(class_dir, basename(file), tgt), recurse = TRUE)
            saveRDS(m_cv, path(class_dir, basename(file), tgt, paste0(basename(file), "_", tgt, "_clf.rds")))
            if("importance" %in% names(m_cv$finalModel)) {
              imp_matrix <- m_cv$finalModel$importance
              imp_df <- data.frame(
                feature = rownames(imp_matrix),
                importance = imp_matrix[, "MeanDecreaseGini"],
                stringsAsFactors = FALSE
              ) %>%
                arrange(desc(importance))
              write_csv(imp_df, path(class_dir, basename(file), tgt, paste0(basename(file), "_", tgt, "_clf_importance.csv")))
            }
            message(glue("    Classification model trained successfully"))
          }, error = function(e) {
            message(glue("    Classification model failed: {e$message}"))
          })
        }
      }
    }
    if(length(target_clean) >= 10) {
      train_reg <- train_df %>% 
        select(all_of(feats), target_val = all_of(tgt)) %>%
        filter(!is.na(target_val), is.finite(target_val))
      test_reg <- test_df %>% 
        select(all_of(feats), target_val = all_of(tgt)) %>%
        filter(!is.na(target_val), is.finite(target_val))
      if(nrow(train_reg) >= 10 && nrow(test_reg) >= 5) {
        set.seed(2025)
        tryCatch({
          m_cv_r <- train(
            x = train_reg %>% select(all_of(feats)),
            y = train_reg$target_val,
            method = "rf",
            metric = "RMSE",
            trControl = cv_ctrl_reg,
            tuneGrid = data.frame(mtry = max(1, floor(sqrt(length(feats))))),
            nodesize = max(1, floor(nrow(train_reg) * 0.05)),
            ntree = 300,
            importance = TRUE
          )
          metrics <- evaluate_model(m_cv_r, train_reg, test_reg, "target_val", "regression")
          diag <- diagnose_model(metrics, leaks, length(feats))
          key_diag <- paste(basename(file), tgt, "regression", sep = "_")
          all_diagnostics[[key_diag]] <- list(
            type = "regression", 
            diagnosis = diag, 
            metrics = metrics,
            n_features = length(feats),
            n_train = nrow(train_reg),
            n_test = nrow(test_reg)
          )
          if(length(leaks) > 0) {
            all_leaks_regression[[key_diag]] <- tibble(
              feature = leaks, 
              file = basename(file), 
              target = tgt
            )
          }
          dir_create(path(reg_dir, basename(file), tgt), recurse = TRUE)
          saveRDS(m_cv_r, path(reg_dir, basename(file), tgt, paste0(basename(file), "_", tgt, "_reg.rds")))
          if("importance" %in% names(m_cv_r$finalModel)) {
            imp_matrix <- m_cv_r$finalModel$importance
            imp_df <- data.frame(
              feature = rownames(imp_matrix),
              importance = imp_matrix[, "IncNodePurity"],
              stringsAsFactors = FALSE
            ) %>%
              arrange(desc(importance))
            write_csv(imp_df, path(reg_dir, basename(file), tgt, paste0(basename(file), "_", tgt, "_reg_importance.csv")))
          }
          message(glue("    Regression model trained successfully"))
        }, error = function(e) {
          message(glue("    Regression model failed: {e$message}"))
        })
      }
    }
  }
}
if(length(all_diagnostics) > 0) {
  diag_df <- map_dfr(names(all_diagnostics), function(key) {
    diag_info <- all_diagnostics[[key]]
    metrics <- diag_info$metrics
    base_info <- tibble(
      model = key,
      type = diag_info$type,
      severity = diag_info$diagnosis$severity,
      issues = paste(diag_info$diagnosis$issues, collapse = "; "),
      n_features = diag_info$n_features,
      n_train = diag_info$n_train,
      n_test = diag_info$n_test
    )
    if(diag_info$type == "classification") {
      base_info$train_accuracy <- metrics$train_accuracy %||% NA
      base_info$test_accuracy <- metrics$test_accuracy %||% NA
      base_info$train_auc <- metrics$train_auc %||% NA
      base_info$test_auc <- metrics$test_auc %||% NA
    } else {
      base_info$train_rmse <- metrics$train_rmse %||% NA
      base_info$test_rmse <- metrics$test_rmse %||% NA
      base_info$train_r2 <- metrics$train_r2 %||% NA
      base_info$test_r2 <- metrics$test_r2 %||% NA
    }
    base_info
  })
  write_csv(diag_df, path(diagnostics_dir, "model_diagnostics.csv"))
  severity_counts <- diag_df %>% count(severity, sort = TRUE)
  write_csv(severity_counts, path(diagnostics_dir, "severity_summary.csv"))
  type_counts <- diag_df %>% count(type, sort = TRUE)
  write_csv(type_counts, path(diagnostics_dir, "type_summary.csv"))
  message(glue("Total models analyzed: {nrow(diag_df)}"))
  walk(seq_len(nrow(severity_counts)), ~{
    message(glue("{severity_counts$severity[.x]}: {severity_counts$n[.x]} models"))
  })
} else {
  message("No models were successfully trained")
}
create_leakage_plot <- function(leak_data, title, filename) {
  if(length(leak_data) == 0) return(NULL)
  leak_summary <- bind_rows(leak_data) %>% 
    count(feature, sort = TRUE) %>%
    slice_head(n = 20)
  if(nrow(leak_summary) == 0) return(NULL)
  p <- leak_summary %>%
    ggplot(aes(x = reorder(feature, n), y = n)) +
    geom_col(fill = viridis(1, option = "D"), alpha = 0.8) +
    coord_flip() +
    labs(
      title = title,
      subtitle = paste("Top 20 most frequently removed features"),
      x = "Feature",
      y = "Number of models"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
  ggsave(path(new_root, filename), p, width = 10, height = 8, dpi = 300)
  return(p)
}
create_leakage_plot(all_leaks_classification, 
                    "Removed Leakage Features - Classification Models", 
                    "leakage_removed_classification.png")
create_leakage_plot(all_leaks_regression, 
                    "Removed Leakage Features - Regression Models", 
                    "leakage_removed_regression.png")
message("Training data exported and diagnostics file structure created!")
clf_models <- dir_ls(class_dir, recurse = TRUE, glob = "*_clf.rds")
for (clf_model in clf_models) {
  model_dir <- path_dir(clf_model)
  if (!file_exists(path(model_dir, "model.rds"))) {
    file_copy(clf_model, path(model_dir, "model.rds"))
  }
}
reg_models <- dir_ls(reg_dir, recurse = TRUE, glob = "*_reg.rds")
for (reg_model in reg_models) {
  model_dir <- path_dir(reg_model)
  if (!file_exists(path(model_dir, "model.rds"))) {
    file_copy(reg_model, path(model_dir, "model.rds"))
  }
}
clf_importance <- dir_ls(class_dir, recurse = TRUE, glob = "*_clf_importance.csv")
for (clf_imp in clf_importance) {
  model_dir <- path_dir(clf_imp)
  if (!file_exists(path(model_dir, "importance.csv"))) {
    file_copy(clf_imp, path(model_dir, "importance.csv"))
  }
}
reg_importance <- dir_ls(reg_dir, recurse = TRUE, glob = "*_reg_importance.csv")
for (reg_imp in reg_importance) {
  model_dir <- path_dir(reg_imp)
  if (!file_exists(path(model_dir, "importance.csv"))) {
    file_copy(reg_imp, path(model_dir, "importance.csv"))
  }
}
message("Ready to run diagnostics script.")
message("Analysis complete!")
if(!interactive()) q(save = "no", status = 0)