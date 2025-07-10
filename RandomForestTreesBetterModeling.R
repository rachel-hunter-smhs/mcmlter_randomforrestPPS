suppressPackageStartupMessages({
  library(tidyverse)
  library(fastDummies)
  library(randomForest)
  library(caret)
  library(pROC)
  library(janitor)
  library(fs)
})

files <- c(
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-bee-20250304.csv",
  "C:/Users/rache/OneDrive/Documents/mcmurdo/mcmlter-soil-et-20250305.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_rowbind_shared.csv",
  "C:/Users/rache/OneDrive/Documents/resultsPostRevision/bee_et_full_join.csv"
)

output_root <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_full"
# safely delete old output
if (fs::dir_exists(output_root)) {
  try({ fs::dir_delete(output_root) }, silent = TRUE)
}
fs::dir_create(output_root)

processed_dir <- fs::path(output_root, "processed")
class_root    <- fs::path(output_root, "classification")
reg_root      <- fs::path(output_root, "regression")
fs::dir_create(c(processed_dir, class_root, reg_root))

median_impute <- function(x) {
  m <- median(x, na.rm = TRUE)
  replace(x, is.na(x) | !is.finite(x), m)
}

mode_impute <- function(x) {
  ux <- na.omit(x)
  if (!length(ux)) return(x)
  mv <- names(which.max(table(ux)))
  replace(x, is.na(x), mv)
}

set.seed(2025)

for (fpath in files) {
  name <- tools::file_path_sans_ext(basename(fpath))
  message("Processing dataset: ", name)
  
  df   <- suppressWarnings(
    read_csv(fpath, show_col_types = FALSE, guess_max = 100000)
  ) %>%
    mutate(across(where(is.numeric), median_impute)) %>%
    mutate(across(where(~ is.character(.x) || is.factor(.x)), mode_impute)) %>%
    dummy_cols(remove_selected_columns = TRUE) %>%
    clean_names()
  
  n        <- nrow(df)
  idx      <- sample(n)
  n_train  <- floor(0.6 * n)
  n_valid  <- floor(0.2 * n)
  train_df <- df[idx[1:n_train], ]
  valid_df <- df[idx[(n_train + 1):(n_train + n_valid)], ]
  test_df  <- df[idx[(n_train + n_valid + 1):n], ]
  
  write_csv(train_df, fs::path(processed_dir, paste0(name, "_train.csv")))
  write_csv(valid_df, fs::path(processed_dir, paste0(name, "_validation.csv")))
  write_csv(test_df,  fs::path(processed_dir, paste0(name, "_test.csv")))
  
  targets <- intersect(c("total", "total_live", "total_dead"), names(df))
  message("Found targets: ", paste(targets, collapse = ", "))
  
  for (t in targets) {
    message("  Processing target: ", t)
    p80 <- quantile(train_df[[t]], 0.8, na.rm = TRUE)
    
    # Classification
    train_clf <- train_df %>%
      mutate(soil_class = factor(if_else(.data[[t]] >= p80, "High", "Low"))) %>%
      select(-all_of(targets))
    
    if (n_distinct(train_clf$soil_class) < 2) {
      message("    Skipping classification for ", name, "/", t, ": only one class.")
    } else {
      clf <- tryCatch(
        randomForest(soil_class ~ ., data = train_clf, ntree = 500, importance = TRUE, na.action = na.omit),
        error = function(e) { message("    RF error for ", name, "/", t, ": ", e$message); NULL }
      )
      
      if (!is.null(clf)) {
        clf_dir <- fs::path(class_root, name, t)
        fs::dir_create(clf_dir)
        
        # Save model with consistent naming
        saveRDS(clf, fs::path(clf_dir, paste0(name, "_", t, "_clf.rds")))
        
        # Save importance with consistent naming
        imp_df <- importance(clf) %>% 
          as_tibble(rownames = "feature")
        write_csv(imp_df, fs::path(clf_dir, paste0(name, "_", t, "_clf_importance.csv")))
        
        # Also save as the original name for backwards compatibility
        write_csv(imp_df, fs::path(clf_dir, "importance.csv"))
        
        message("    Classification model saved for ", name, "/", t)
        
        # Validation and test predictions
        for (split in c("validation", "test")) {
          split_df <- if (split == "validation") valid_df else test_df
          split_clf <- split_df %>%
            mutate(soil_class = factor(if_else(.data[[t]] >= p80, "High", "Low"))) %>%
            select(-all_of(targets))
          
          preds <- predict(clf, split_clf)
          probs <- predict(clf, split_clf, type = "prob")[, "High"]
          cm    <- confusionMatrix(preds, split_clf$soil_class)
          
          write_csv(as_tibble(t(cm$overall)), fs::path(clf_dir, paste0("clf_overall_", split, ".csv")))
          write_csv(as_tibble(cm$byClass), fs::path(clf_dir, paste0("clf_byClass_", split, ".csv")))
          
          roc_obj <- roc(split_clf$soil_class, probs, levels = c("Low","High"))
          write_csv(tibble(AUC = as.numeric(auc(roc_obj))), fs::path(clf_dir, paste0("clf_auc_", split, ".csv")))
        }
      }
    }
    
    # Regression
    reg_data <- train_df %>% 
      select(-all_of(setdiff(targets, t)))  # Keep only the current target
    
    reg <- tryCatch(
      randomForest(as.formula(paste(t, "~ .")), data = reg_data, ntree = 500, importance = TRUE, na.action = na.omit),
      error = function(e) { message("    Regression error for ", name, "/", t, ": ", e$message); NULL }
    )
    
    if (!is.null(reg)) {
      reg_dir <- fs::path(reg_root, name, t)
      fs::dir_create(reg_dir)
      saveRDS(reg, fs::path(reg_dir, "model.rds"))
      write_csv(importance(reg) %>% as_tibble(rownames = "feature"), fs::path(reg_dir, "importance.csv"))
      
      message("    Regression model saved for ", name, "/", t)
      
      for (split in c("validation", "test")) {
        split_df <- if (split == "validation") valid_df else test_df
        obs   <- split_df[[t]]
        preds <- predict(reg, split_df %>% select(-all_of(setdiff(targets, t))))
        res   <- obs - preds
        metrics <- tibble(
          RMSE = sqrt(mean(res^2, na.rm = TRUE)),
          R2   = cor(obs, preds, use = "complete.obs")^2,
          MAE  = mean(abs(res), na.rm = TRUE)
        )
        write_csv(metrics, fs::path(reg_dir, paste0("reg_metrics_", split, ".csv")))
      }
    }
  }
}

message("Model generation complete. Files saved to: ", output_root)