suppressPackageStartupMessages({
  library(tidyverse)
  library(pdp)
  library(caret)
  library(pROC)
  library(viridis)
  library(fs)
})

root <- "C:/Users/rache/OneDrive/Documents/resultsPostRevision/random_forest_full"
processed_dir <- file.path(root, "processed")
class_root    <- file.path(root, "classification")

message("Checking directories...")
message("Root exists: ", dir.exists(root))
message("Processed dir exists: ", dir.exists(processed_dir))
message("Classification dir exists: ", dir.exists(class_root))

if (!dir.exists(class_root)) {
  stop("Classification directory does not exist: ", class_root)
}

dataset_dirs <- list.dirs(class_root, recursive = FALSE, full.names = TRUE)
message("Found ", length(dataset_dirs), " dataset directories:")
for(ds_dir in dataset_dirs) {
  message("  - ", basename(ds_dir))
}

set.seed(2025)

for(ds_dir in dataset_dirs) {
  ds <- basename(ds_dir)
  train_csv <- file.path(processed_dir, paste0(ds, "_train.csv"))
  valid_csv <- file.path(processed_dir, paste0(ds, "_validation.csv"))
  test_csv  <- file.path(processed_dir, paste0(ds, "_test.csv"))
  
  message("\nChecking splits for dataset: ", ds)
  message("  Train: ", file.exists(train_csv), " (", train_csv, ")")
  message("  Valid: ", file.exists(valid_csv), " (", valid_csv, ")")
  message("  Test: ", file.exists(test_csv), " (", test_csv, ")")
  
  if(!all(file.exists(c(train_csv, valid_csv, test_csv)))) {
    message("  WARNING: Missing splits for dataset: ", ds)
  }
}


for(ds_dir in dataset_dirs) {
  ds <- basename(ds_dir)
  target_dirs <- list.dirs(ds_dir, recursive = FALSE, full.names = TRUE)
  
  message("\nProcessing dataset: ", ds)
  message("Found ", length(target_dirs), " target directories:")
  for(tdir in target_dirs) {
    message("  - ", basename(tdir))
  }
  
  for(target_dir in target_dirs) {
    target <- basename(target_dir)
    message("\n  Processing ", ds, "/", target)
    
    train_csv <- file.path(processed_dir, paste0(ds, "_train.csv"))
    imp_csv   <- file.path(target_dir, paste0(ds, "_", target, "_clf_importance.csv"))
    model_rds <- file.path(target_dir, paste0(ds, "_", target, "_clf.rds"))
    
    if (!file.exists(imp_csv)) {
      imp_csv <- file.path(target_dir, "importance.csv")
    }
    if (!file.exists(model_rds)) {
      model_rds <- file.path(target_dir, "model.rds")
    }
    
    message("    Checking files:")
    message("      Train CSV: ", file.exists(train_csv), " (", train_csv, ")")
    message("      Importance CSV: ", file.exists(imp_csv), " (", imp_csv, ")")
    message("      Model RDS: ", file.exists(model_rds), " (", model_rds, ")")
    
    if(!file.exists(train_csv)) {
      message("    SKIP: Missing train split")
      next
    }
    if(!file.exists(imp_csv)) {
      message("    SKIP: Missing importance CSV")
      next
    }
    if(!file.exists(model_rds)) {
      message("    SKIP: Missing model RDS")
      next
    }
    
    # Load data and model
    tryCatch({
      train <- read_csv(train_csv, show_col_types = FALSE)
      imp   <- read_csv(imp_csv,   show_col_types = FALSE)
      model <- readRDS(model_rds)
      
      message("    Successfully loaded data and model")
      message("    Train data dimensions: ", nrow(train), " x ", ncol(train))
      message("    Number of features in importance: ", nrow(imp))
      
      
      if (!target %in% names(train)) {
        message("    SKIP: Target '", target, "' not found in training data")
        message("    Available columns: ", paste(names(train)[1:min(10, ncol(train))], collapse = ", "))
        next
      }
      
      thr <- quantile(train[[target]], 0.8, na.rm = TRUE)
      message("    Threshold (80th percentile): ", thr)
      
      
      X <- train %>% 
        mutate(soil_class = factor(if_else(.data[[target]] >= thr, "High", "Low"), levels = c("Low", "High"))) %>%
        select(-any_of(c("total", "total_live", "total_dead")))  # Remove all possible targets
      
      message("    Class distribution: ", table(X$soil_class))
      
      
      top_feats <- imp %>% 
        arrange(desc(MeanDecreaseGini)) %>% 
        slice_head(n = 3) %>% 
        pull(feature)
      
      message("    Top 3 features: ", paste(top_feats, collapse = ", "))
      
  
      missing_feats <- setdiff(top_feats, names(X))
      if (length(missing_feats) > 0) {
        message("    WARNING: Missing features in data: ", paste(missing_feats, collapse = ", "))
        top_feats <- intersect(top_feats, names(X))
        message("    Using available features: ", paste(top_feats, collapse = ", "))
      }
      
      if (length(top_feats) == 0) {
        message("    SKIP: No valid features found")
        next
      }
      
     
      for(feat in top_feats) {
        message("      Generating PDP/ICE for feature: ", feat)
        
        tryCatch({
          
          pred_wrapper <- function(object, newdata) {
            predict(object, newdata, type = "prob")[, "High"]
          }
          
         
          pdp_data <- partial(model, pred.var = feat, train = X %>% select(-soil_class), 
                              pred.fun = pred_wrapper, grid.resolution = 30)
          write_csv(pdp_data, file.path(target_dir, paste0("pdp_data_", feat, ".csv")))
          
          p_pdp <- ggplot(pdp_data, aes(x = .data[[feat]], y = yhat)) + 
            geom_line(color = viridis(1), linewidth = 1) +
            labs(title = paste("PDP of", feat, "for", ds, "/", target), 
                 x = feat, y = "P(High)") + 
            theme_minimal()
          ggsave(file.path(target_dir, paste0("pdp_", feat, ".png")), p_pdp, 
                 width = 8, height = 6, dpi = 300)
          
          ice_data <- partial(model, pred.var = feat, train = X %>% select(-soil_class), 
                              pred.fun = pred_wrapper, ice = TRUE, grid.resolution = 20)
          write_csv(ice_data, file.path(target_dir, paste0("ice_data_", feat, ".csv")))
          
          p_ice <- ggplot(ice_data, aes(x = .data[[feat]], y = yhat, group = yhat.id)) + 
            geom_line(alpha = 0.3, color = viridis(1)) +
            labs(title = paste("ICE of", feat, "for", ds, "/", target), 
                 x = feat, y = "P(High)") + 
            theme_minimal()
          ggsave(file.path(target_dir, paste0("ice_", feat, ".png")), p_ice, 
                 width = 8, height = 6, dpi = 300)
          
          message("        PDP/ICE plots saved for ", feat)
        }, error = function(e) {
          message("        ERROR in PDP/ICE for ", feat, ": ", e$message)
        })
      }
      
      for(split in c("validation", "test")) {
        split_csv <- file.path(processed_dir, paste0(ds, "_", split, ".csv"))
        if(!file.exists(split_csv)) {
          message("      SKIP calibration for ", split, ": file not found")
          next
        }
        
        tryCatch({
          df_sp <- read_csv(split_csv, show_col_types = FALSE) %>% 
            mutate(soil_class = factor(if_else(.data[[target]] >= thr, "High", "Low"), levels = c("Low", "High")))
          
          probs <- predict(model, df_sp, type = "prob")[, "High"]
          cal_df <- tibble(obs = as.integer(df_sp$soil_class == "High"), pred = probs) %>% 
            mutate(bin = ntile(pred, 10)) %>%
            group_by(bin) %>% 
            summarise(mean_pred = mean(pred), obs_rate = mean(obs), .groups = "drop")
          
          write_csv(cal_df, file.path(target_dir, paste0("calibration_", split, ".csv")))
          
          p_cal <- ggplot(cal_df, aes(mean_pred, obs_rate)) + 
            geom_line(color = viridis(1), linewidth = 1) + 
            geom_abline(linetype = "dashed", alpha = 0.5) +
            labs(title = paste("Calibration", split, "for", ds, "/", target), 
                 x = "Mean Predicted Probability", y = "Observed Rate") + 
            theme_minimal()
          ggsave(file.path(target_dir, paste0("calibration_", split, ".png")), p_cal, 
                 width = 8, height = 6, dpi = 300)
          
          message("      Calibration plot saved for ", split)
        }, error = function(e) {
          message("      ERROR in calibration for ", split, ": ", e$message)
        })
      }
      
      if(length(top_feats) >= 2) {
        tryCatch({
          message("      Generating interaction heatmap...")
          
    
          pred_wrapper <- function(object, newdata) {
            predict(object, newdata, type = "prob")[, "High"]
          }
          
          int_df <- partial(model, pred.var = top_feats[1:2], train = X %>% select(-soil_class), 
                            pred.fun = pred_wrapper, grid.resolution = 20)
          write_csv(int_df, file.path(target_dir, paste0("interaction_data_", target, ".csv")))
          
          p_int <- ggplot(int_df, aes(x = .data[[top_feats[1]]], y = .data[[top_feats[2]]], fill = yhat)) + 
            geom_tile() + scale_fill_viridis_c() +
            labs(title = paste("Interaction of", top_feats[1], "vs", top_feats[2], "for", ds, "/", target), 
                 x = top_feats[1], y = top_feats[2], fill = "P(High)") + 
            theme_minimal()
          ggsave(file.path(target_dir, paste0("interaction_heatmap_", target, ".png")), p_int, 
                 width = 8, height = 6, dpi = 300)
          
          message("      Interaction heatmap saved")
        }, error = function(e) {
          message("      ERROR in interaction heatmap: ", e$message)
        })
      }
      
      message("    Completed processing for ", ds, "/", target)
      
    }, error = function(e) {
      message("    ERROR loading data/model for ", ds, "/", target, ": ", e$message)
    })
  }
}

message("\nAll graphs generated under ", class_root)
message("Check the individual target directories for outputs.")