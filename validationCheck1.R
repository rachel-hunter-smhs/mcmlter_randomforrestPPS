suppressPackageStartupMessages({
  library(tidyverse)
  library(fs)
  library(glue)
  library(randomForest)
  library(caret)
  library(pROC)
})

root_dir     <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_new"
class_dir    <- path(root_dir, "classification")
reg_dir      <- path(root_dir, "regression")
results_dir  <- path(root_dir, "results_tables")
issues_dir   <- path(root_dir, "validationDiagnostics")
processed_dir <- path(root_dir, "processed")
diagnostics_dir <- path(root_dir, "diagnostics")

dir_create(results_dir, recurse = TRUE)
dir_create(issues_dir,  recurse = TRUE)


evaluate_model_performance <- function(model_path, train_data, test_data, target_col, type) {
  if (!file_exists(model_path)) return(NULL)
  
  model <- tryCatch({
    readRDS(model_path)
  }, error = function(e) {
    message(glue("Error loading model {model_path}: {e$message}"))
    return(NULL)
  })
  
  if (is.null(model)) return(NULL)
  
  tryCatch({
    if (type == "classification") {
      
      train_pred <- predict(model, train_data)
      train_actual <- train_data[[target_col]]
      train_acc <- mean(train_pred == train_actual, na.rm = TRUE)
      
      test_pred <- predict(model, test_data)
      test_actual <- test_data[[target_col]]
      test_acc <- mean(test_pred == test_actual, na.rm = TRUE)
      
      
      train_balanced_acc <- NA
      test_balanced_acc <- NA
      
      if (length(unique(train_actual)) == 2) {
        train_cm <- table(train_actual, train_pred)
        if (nrow(train_cm) == 2 && ncol(train_cm) == 2) {
          sens <- train_cm[2,2] / sum(train_cm[2,])
          spec <- train_cm[1,1] / sum(train_cm[1,])
          train_balanced_acc <- (sens + spec) / 2
        }
      }
      
      if (length(unique(test_actual)) == 2) {
        test_cm <- table(test_actual, test_pred)
        if (nrow(test_cm) == 2 && ncol(test_cm) == 2) {
          sens <- test_cm[2,2] / sum(test_cm[2,])
          spec <- test_cm[1,1] / sum(test_cm[1,])
          test_balanced_acc <- (sens + spec) / 2
        }
      }
      
      
      train_auc <- test_auc <- NA
      tryCatch({
        train_prob <- predict(model, train_data, type = "prob")
        if (!is.null(train_prob) && ncol(train_prob) >= 2) {
          train_auc <- as.numeric(auc(roc(train_actual, train_prob[,2], quiet = TRUE)))
        }
      }, error = function(e) {})
      
      tryCatch({
        test_prob <- predict(model, test_data, type = "prob")
        if (!is.null(test_prob) && ncol(test_prob) >= 2) {
          test_auc <- as.numeric(auc(roc(test_actual, test_prob[,2], quiet = TRUE)))
        }
      }, error = function(e) {})
      
     
      train_f1 <- test_f1 <- NA
      if (length(unique(train_actual)) == 2) {
        train_cm <- table(train_actual, train_pred)
        if (nrow(train_cm) == 2 && ncol(train_cm) == 2) {
          precision <- train_cm[2,2] / sum(train_cm[,2])
          recall <- train_cm[2,2] / sum(train_cm[2,])
          if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
            train_f1 <- 2 * precision * recall / (precision + recall)
          }
        }
      }
      
      if (length(unique(test_actual)) == 2) {
        test_cm <- table(test_actual, test_pred)
        if (nrow(test_cm) == 2 && ncol(test_cm) == 2) {
          precision <- test_cm[2,2] / sum(test_cm[,2])
          recall <- test_cm[2,2] / sum(test_cm[2,])
          if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
            test_f1 <- 2 * precision * recall / (precision + recall)
          }
        }
      }
      
      tibble(
        train_accuracy = train_acc,
        test_accuracy = test_acc,
        train_balanced_accuracy = train_balanced_acc,
        test_balanced_accuracy = test_balanced_acc,
        train_auc = train_auc,
        test_auc = test_auc,
        train_f1 = train_f1,
        test_f1 = test_f1,
        type = "classification"
      )
      
    } else {
     
      train_pred <- predict(model, train_data)
      train_actual <- train_data[[target_col]]
      train_valid <- !is.na(train_pred) & !is.na(train_actual) & is.finite(train_pred) & is.finite(train_actual)
      
      test_pred <- predict(model, test_data)
      test_actual <- test_data[[target_col]]
      test_valid <- !is.na(test_pred) & !is.na(test_actual) & is.finite(test_pred) & is.finite(test_actual)
      
      if (!any(train_valid) || !any(test_valid)) {
        return(tibble(
          train_rmse = NA, test_rmse = NA, train_r2 = NA, test_r2 = NA,
          train_mae = NA, test_mae = NA, type = "regression"
        ))
      }
      
      train_rmse <- sqrt(mean((train_pred[train_valid] - train_actual[train_valid])^2))
      test_rmse <- sqrt(mean((test_pred[test_valid] - test_actual[test_valid])^2))
      
      train_r2 <- cor(train_pred[train_valid], train_actual[train_valid])^2
      test_r2 <- cor(test_pred[test_valid], test_actual[test_valid])^2
      
      train_mae <- mean(abs(train_pred[train_valid] - train_actual[train_valid]))
      test_mae <- mean(abs(test_pred[test_valid] - test_actual[test_valid]))
      
      tibble(
        train_rmse = train_rmse,
        test_rmse = test_rmse,
        train_r2 = train_r2,
        test_r2 = test_r2,
        train_mae = train_mae,
        test_mae = test_mae,
        type = "regression"
      )
    }
  }, error = function(e) {
    message(glue("Error evaluating model performance: {e$message}"))
    return(NULL)
  })
}


classification_file <- path(results_dir, "classification_diagnostics_table.csv")
if (!file_exists(classification_file)) {
  message("Generating classification diagnostics from model outputs...")
  
  
  clf_dirs <- dir_ls(class_dir, type = "directory", recurse = TRUE)
  clf_results <- list()
  
  for (clf_dir in clf_dirs) {
    model_file <- path(clf_dir, "model.rds")
    if (!file_exists(model_file)) {
      
      model_files <- dir_ls(clf_dir, glob = "*_clf.rds")
      if (length(model_files) > 0) {
        model_file <- model_files[1]
      } else {
        next
      }
    }
    
    
    path_parts <- path_split(clf_dir)[[1]]
    dataset <- path_parts[length(path_parts) - 1]
    target <- path_parts[length(path_parts)]
    
    
    train_file <- path(processed_dir, paste0(dataset, "_train.csv"))
    test_file <- path(processed_dir, paste0(dataset, "_test.csv"))
    
    if (file_exists(train_file) && file_exists(test_file)) {
      train_data <- read_csv(train_file, show_col_types = FALSE)
      test_data <- read_csv(test_file, show_col_types = FALSE)
      
      
      if (target %in% names(train_data)) {
        target_vals <- train_data[[target]]
        target_clean <- target_vals[!is.na(target_vals) & is.finite(target_vals)]
        
        if (length(target_clean) > 0) {
          q75 <- quantile(target_clean, 0.75, na.rm = TRUE)
          
          train_data <- train_data %>%
            filter(!is.na(!!sym(target)), is.finite(!!sym(target))) %>%
            mutate(label = factor(if_else(!!sym(target) >= q75, "High", "Low"), levels = c("Low", "High")))
          
          test_data <- test_data %>%
            filter(!is.na(!!sym(target)), is.finite(!!sym(target))) %>%
            mutate(label = factor(if_else(!!sym(target) >= q75, "High", "Low"), levels = c("Low", "High")))
          
         
          metrics <- evaluate_model_performance(model_file, train_data, test_data, "label", "classification")
          
          if (!is.null(metrics)) {
            result <- metrics %>%
              mutate(
                dataset = dataset,
                target = target,
                accuracy = test_accuracy,
                balanced_accuracy = test_balanced_accuracy,
                auc = test_auc,
                f1 = test_f1
              ) %>%
              select(dataset, target, accuracy, balanced_accuracy, auc, f1, everything())
            
            clf_results[[paste(dataset, target, sep = "_")]] <- result
          }
        }
      }
    }
  }
  
  if (length(clf_results) > 0) {
    clf_diag <- bind_rows(clf_results)
    write_csv(clf_diag, classification_file)
    message(glue("Generated classification diagnostics for {nrow(clf_diag)} models"))
  } else {
    message("No classification models found to generate diagnostics")
  }
}


regression_file <- path(results_dir, "regression_diagnostics_table.csv")
if (!file_exists(regression_file)) {
  message("Generating regression diagnostics from model outputs...")
  
  
  reg_dirs <- dir_ls(reg_dir, type = "directory", recurse = TRUE)
  reg_results <- list()
  
  for (reg_dir_path in reg_dirs) {
    model_file <- path(reg_dir_path, "model.rds")
    if (!file_exists(model_file)) {
     
      model_files <- dir_ls(reg_dir_path, glob = "*_reg.rds")
      if (length(model_files) > 0) {
        model_file <- model_files[1]
      } else {
        next
      }
    }
    
    
    path_parts <- path_split(reg_dir_path)[[1]]
    dataset <- path_parts[length(path_parts) - 1]
    target <- path_parts[length(path_parts)]
    
   
    train_file <- path(processed_dir, paste0(dataset, "_train.csv"))
    test_file <- path(processed_dir, paste0(dataset, "_test.csv"))
    
    if (file_exists(train_file) && file_exists(test_file)) {
      train_data <- read_csv(train_file, show_col_types = FALSE)
      test_data <- read_csv(test_file, show_col_types = FALSE)
      
      
      if (target %in% names(train_data)) {
        train_data <- train_data %>%
          select(everything()) %>%
          rename(target_val = !!sym(target)) %>%
          filter(!is.na(target_val), is.finite(target_val))
        
        test_data <- test_data %>%
          select(everything()) %>%
          rename(target_val = !!sym(target)) %>%
          filter(!is.na(target_val), is.finite(target_val))
        
        # Evaluate model
        metrics <- evaluate_model_performance(model_file, train_data, test_data, "target_val", "regression")
        
        if (!is.null(metrics)) {
          result <- metrics %>%
            mutate(
              dataset = dataset,
              target = target,
              rmse = test_rmse,
              r2 = test_r2,
              mae = test_mae
            ) %>%
            select(dataset, target, rmse, r2, mae, everything())
          
          reg_results[[paste(dataset, target, sep = "_")]] <- result
        }
      }
    }
  }
  
  if (length(reg_results) > 0) {
    reg_diag <- bind_rows(reg_results)
    write_csv(reg_diag, regression_file)
    message(glue("Generated regression diagnostics for {nrow(reg_diag)} models"))
  } else {
    message("No regression models found to generate diagnostics")
  }
}


safe_read <- function(f) {
  if (file_exists(f)) {
    message("Reading: ", f)
    read_csv(f, show_col_types = FALSE) |> rename_with(tolower)
  } else {
    message("Missing file: ", f)
    tibble()
  }
}

classification_data <- safe_read(classification_file)
regression_data     <- safe_read(regression_file)

detect_duplicates <- function(df, ignore_cols = character()) {
  cols <- setdiff(names(df), ignore_cols)
  df |> group_by(across(all_of(cols))) |> filter(n() > 1) |> ungroup()
}


classification_issues <- tibble()
if (nrow(classification_data) > 0) {
  stopifnot(all(c("dataset", "target") %in% names(classification_data)))
  metric_cols <- intersect(names(classification_data), c("accuracy", "balanced_accuracy", "auc", "f1"))
  classification_data <- classification_data |> mutate(across(all_of(metric_cols), as.numeric))
  
  duplicates       <- detect_duplicates(classification_data) |> mutate(issue = "duplicate_row")
  high_values      <- classification_data |> filter(if_any(all_of(metric_cols), ~ .x >= 0.98)) |> mutate(issue = "suspiciously_high_metric")
  missing_metrics  <- classification_data |> filter(if_any(all_of(metric_cols), is.na)) |> mutate(issue = "missing_metric")

  
  perfect_scores <- classification_data |> filter(if_any(all_of(metric_cols), ~ .x == 1.0)) |> mutate(issue = "perfect_score")
  
  low_performance <- classification_data |> filter(if_any(all_of(metric_cols), ~ .x < 0.5)) |> mutate(issue = "low_performance")
  
  classification_issues <- bind_rows(duplicates, high_values, missing_metrics, perfect_scores, low_performance) |> distinct()
  
  if (nrow(classification_issues) > 0) {
    write_csv(classification_issues, path(issues_dir, "classification_validation_issues.csv"))
    message(glue("Classification issues found: {nrow(classification_issues)} rows written to validation issues."))
  } else {
    message("No classification issues found.")
  }
} else {
  message("No classification data found.")
}

# Validation for regression
regression_issues <- tibble()
if (nrow(regression_data) > 0) {
  stopifnot(all(c("dataset", "target") %in% names(regression_data)))
  metric_cols <- intersect(names(regression_data), c("rmse", "r2", "mae"))
  regression_data <- regression_data |> mutate(across(all_of(metric_cols), as.numeric))
  
  duplicates       <- detect_duplicates(regression_data) |> mutate(issue = "duplicate_row")
  high_r2          <- regression_data |> filter(r2 >= 0.99) |> mutate(issue = "suspiciously_high_r2")
  missing_values   <- regression_data |> filter(if_any(all_of(metric_cols), is.na)) |> mutate(issue = "missing_metric")
  
  
  perfect_r2 <- regression_data |> filter(r2 == 1.0) |> mutate(issue = "perfect_r2")
  
  
  low_r2 <- regression_data |> filter(r2 < 0.1) |> mutate(issue = "very_low_r2")
  
  
  negative_r2 <- regression_data |> filter(r2 < 0) |> mutate(issue = "negative_r2")
  
 
  zero_rmse <- regression_data |> filter(rmse == 0) |> mutate(issue = "zero_rmse")
  
  variability <- regression_data |> 
    group_by(dataset, target) |> filter(n() > 1) |>
    summarise(
      rmse_sd = sd(rmse, na.rm = TRUE),
      mae_sd  = sd(mae,  na.rm = TRUE),
      r2_sd   = sd(r2,   na.rm = TRUE),
      rmse_mean = mean(rmse, na.rm = TRUE),
      mae_mean  = mean(mae,  na.rm = TRUE),
      .groups = "drop"
    ) |>
    filter(
      rmse_sd > 0.1 * rmse_mean |
        mae_sd  > 0.1 * mae_mean |
        r2_sd   > 0.05
    ) |>
    inner_join(regression_data, by = c("dataset", "target")) |>
    mutate(issue = "large_split_gap")
  
  regression_issues <- bind_rows(duplicates, high_r2, missing_values, perfect_r2, low_r2, negative_r2, zero_rmse, variability) |> distinct()
  
  if (nrow(regression_issues) > 0) {
    write_csv(regression_issues, path(issues_dir, "regression_validation_issues.csv"))
    message(glue("Regression issues found: {nrow(regression_issues)} rows written to validation issues."))
  } else {
    message("No regression issues found.")
  }
} else {
  message("No regression data found.")
}


if (nrow(classification_data) > 0) {
  clf_summary <- classification_data |>
    summarise(
      n_models = n(),
      mean_accuracy = mean(accuracy, na.rm = TRUE),
      mean_auc = mean(auc, na.rm = TRUE),
      high_performance = sum(accuracy >= 0.8, na.rm = TRUE),
      low_performance = sum(accuracy < 0.6, na.rm = TRUE)
    )
  
  write_csv(clf_summary, path(results_dir, "classification_summary.csv"))
  message("Classification summary statistics written.")
}

if (nrow(regression_data) > 0) {
  reg_summary <- regression_data |>
    summarise(
      n_models = n(),
      mean_r2 = mean(r2, na.rm = TRUE),
      mean_rmse = mean(rmse, na.rm = TRUE),
      high_performance = sum(r2 >= 0.7, na.rm = TRUE),
      low_performance = sum(r2 < 0.3, na.rm = TRUE)
    )
  
  write_csv(reg_summary, path(results_dir, "regression_summary.csv"))
  message("Regression summary statistics written.")
}

message("Validation issue detection complete.")
message(glue("Classification issues: {nrow(classification_issues)}"))
message(glue("Regression issues: {nrow(regression_issues)}"))