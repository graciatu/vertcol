library(readxl)

# Load data
verts <- read_excel("C:/Users/youss/Downloads/verts.xlsx")
#View(verts)
vert.data <- data.frame(verts[1:153, 4:27])

# Total vertebral sampling 
total.sampling = function(x.y, t) {
  # Ensure t is within the valid range
  if (t > 23 || t < 1) {
    stop("Sample size t must be between 1 and 23.")
  }
  
  # Sampling number of targets
  sampled_columns <- sample(1:23, t)
  vert.sampled <- x.y[, sampled_columns, drop = FALSE]
  
  # Dataset without targets
  vert.sampled_counter <- x.y[, 24, drop = FALSE]
  
  # Combine sampled vertebrae with target
  vert.combined <- cbind(vert.sampled, vert.sampled_counter)
  colnames(vert.combined)[ncol(vert.combined)] <- "Sum_Verts"
  
  # Train linear model
  lm_model <- lm(Sum_Verts ~ ., data = vert.combined)
  predictions <- predict(lm_model, vert.combined)
  residuals <- vert.combined$Sum_Verts - predictions
  vert_sampling_SSE <- sum(residuals^2)  # Sum of Squared Errors
  vert_sampling_R2 <- summary(lm_model)$adj.r.squared
  
  # Create presence indicators for vertebrae
  vertebrae_names <- c("C2", "C3", "C4", "C5", "C6", "C7",
                       "T1", "T2", "T3", "T4", "T5", "T6", "T7", 
                       "T8", "T9", "T10", "T11", "T12", 
                       "L1", "L2", "L3", "L4", "L5")
  
  variable.set <- sapply(vertebrae_names, function(v) {
    if (v %in% colnames(vert.combined)) { 
      1 
    } else { 
      0 
    }
  })
  
  # return results including the model
  #print("done")
  return(list(model = lm_model, SSE = vert_sampling_SSE, R2 = vert_sampling_R2, variable.set = variable.set))
}

results = total.sampling(vert.data, t = 10)
print(results$SSE)
print(results$R2)

new_data <- vert.data[3, 1:23]  # Sample new data (ensure it matches the training data format)
new_data[,14:23]= NA
# Mean imputation for missing values
new_data_imputed <- new_data
for (i in 1:ncol(new_data_imputed)) {
  if (any(is.na(new_data_imputed[, i]))) {
    #print("A")
    mean_value <- mean(vert.data[, i], na.rm = TRUE)  # Calculate mean, ignoring NAs
    new_data_imputed[is.na(new_data_imputed[, i]), i] <- mean_value  # Replace NA with mean
  }
}

# Make predictions with the imputed data
predictions <- predict(results$model, new_data_imputed)
print(predictions)
print((predictions[1] - verts[1,27]) * 100 / verts[1,27]) #calculating error of one example
