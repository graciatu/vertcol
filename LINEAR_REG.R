##Data file without categorical variables
library(readxl)
verts <- read_excel("verts.xlsx")
View(verts)
vert.data <- data.frame(verts[1:153, 4:27])
#total vertebral elements sampling.
total.sampling <- function(x.y, t){#t: number of vertebrae sampled
  #sampling "t" number of target vertebrae
  vert.sampled <- x.y[,sample(1:23, t), drop=F]
  #dataset without target vertebrae
  vert.sampled_counter <- x.y[,24, drop=F]
  #dataset combining sampled "t" number of target vertebrae and sum of total vertebrae
  vert.combined <- as.data.frame(cbind(vert.sampled, vert.sampled_counter))
  colnames(vert.combined)[ncol(vert.combined)] = "Sum_Verts"
  #record which vertebrae are sampled and/or used for regression
  variable1 = if (("C2" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable2 = if (("C3" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable3 = if (("C4" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable4 = if (("C5" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable5 = if (("C6" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable6 = if (("C7" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable7 = if (("T1" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable8 = if (("T2" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable9 = if (("T3" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable10 = if (("T4" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable11 = if (("T5" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable12 = if (("T6" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable13 = if (("T7" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable14 = if (("T8" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable15 = if (("T9" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable16 = if (("T10" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable17 = if (("T11" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable18 = if (("T12" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable19 = if (("L1" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable20 = if (("L2" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable21 = if (("L3" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable22 = if (("L4" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable23 = if (("L5" %in% colnames(vert.combined)) == TRUE){1} else {0}
  variable.set = c(variable1,	variable2,	variable3,	variable4,	variable5,	variable6,	variable7,	variable8,	variable9,	
                   variable10,	variable11, variable12,	variable13,	variable14,	variable15,	variable16,	variable17,	
                   variable18,	variable19,	variable20,	variable21,	variable22,	variable23)
  #SEE of sampled data
  vert_sampling_SSE <- summary(lm(Sum_Verts ~ ., data = vert.combined))$sigma
  #R2 of sampled data
  vert_sampling_R2 <- summary(lm(Sum_Verts ~ ., data = vert.combined))$adj.r.squared
  return(list(vert_sampling_SSE, vert_sampling_R2, v.set = variable.set))
  #return(vert_sampling_SSE)
  #return(vert_sampling_R2)
}
