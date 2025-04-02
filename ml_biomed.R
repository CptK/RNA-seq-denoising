if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# The following initializes usage of Bioc devel
BiocManager::install(version='devel')

BiocManager::install("splatter")

library(splatter)
library(scater)
library(ggplot2)
set.seed(123)

sim1 = splatSimulate(group.prob = c(0.5,0.5),
                     nGenes = 200, dropout.type ="experiment",
                     dropout.shape = -1, dropout.mid = 5, method = "groups")
# Two cell groups with dropout effect 

counts_1 = assay(sim1, "counts") # Count matrix 

sim2 = splatSimulate(group.prob = c(0.5,0.5), nGenes = 200, dropout.type ="none",
                     method = "groups")
# Two cell groups without dropout effect 

counts_2 = assay(sim2, "counts") # Count matrix 

sim3 = splatSimulate(group.prob = c(1/6,1/6,1/6,1/6,1/6,1/6), nGenes = 200,
                     dropout.type ="experiment", dropout.shape = -1,
                     dropout.mid = 1, method = "groups")
# Six cell groups with dropout effect 

counts_3 = assay(sim3, "counts") # Count matrix 

sim4 = splatSimulate(group.prob = c(1/6,1/6,1/6,1/6,1/6,1/6), nGenes = 200,
                     dropout.type ="none", method = "groups")
# Six cell groups without dropout effect 

counts_4 = assay(sim4, "counts") # Count matrix 

#write.csv(counts_1, file = "sim_scdata_2_groups.csv", row.names = TRUE)

# Log-normalization 
sim1 = logNormCounts(sim1)
sim2 = logNormCounts(sim2)
sim3 = logNormCounts(sim3)
sim4 = logNormCounts(sim4)


# Calculating the pca  
pca_result_1 <- prcomp(t(counts_1))
reducedDim(sim1, "PCA") <- pca_result_1$x

pca_result_2 <- prcomp(t(counts_2))
reducedDim(sim2, "PCA") <- pca_result_2$x

pca_result_3 <- prcomp(t(counts_3))
reducedDim(sim3, "PCA") <- pca_result_3$x

pca_result_4 <- prcomp(t(counts_4))
reducedDim(sim4, "PCA") <- pca_result_4$x


# Plotting 
plotPCA(sim1, ncomponents = 2, colour_by = "Group")

plotPCA(sim2, ncomponents = 2, colour_by = "Group")

#plotPCA(sim3, ncomponents = 2, colour_by = "Group")

#plotPCA(sim4, ncomponents = 2, colour_by = "Group")


# I tried to make a work-around for the tSNE but not sure how well it works. 
library(Rtsne)

# Calculations for tSNE
tsne_result_3 <- Rtsne(t(counts_3), dims = 2)  

reducedDim(sim3, "TSNE") <- tsne_result_3$Y  

tsne_result_4 <- Rtsne(t(counts_4), dims = 2)  

reducedDim(sim4, "TSNE") <- tsne_result_4$Y  

# Plotting for tSNE
plotTSNE(sim3, ncomponents = 2, colour_by = "Group")

plotTSNE(sim4, ncomponents = 2, colour_by = "Group")
