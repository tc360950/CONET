library(RColorBrewer)
library(ggplot2)
library(data.tree)
library(data.table)
library(htmlwidgets)
library(webshot)
library(DiagrammeR)
library(stringr)
library(DescTools)
library(PerformanceAnalytics)
library(heatmap3)
library(cowplot)

# Working directory must be set to dir containing this script.
path <- getwd()
source(paste0(path, "/functionsCONET.R"))


## read in per-bin data input
## here we read in a subsample of 50 cells from TN2 breast cancer sample sequenced using ACT
corrected_counts_file <- "/SA501X3F_filtered_corrected_counts_chr_17_18_20_23.csv"
all_reads <- read.table(paste0(path, corrected_counts_file), sep =",", header = T, stringsAsFactors = F)

## give bin width - will be considered when checking which genes overlap with CN events
bin_width_input <- 150000

## set basal ploidy
neutral_CN <- 2

## set maximum inferred integer CN
## if higher than 2*neutral_CN the colors in heatmaps may not reflect higher CNs
maximum_inferred_CN <- 2*neutral_CN

# set model output file names
# in the example we have output from our CONET inferred for TN2 sample
# please remember that the tree looks differently from publication 
# because we only use subsample for illustration purposes
TREE_FILE_NAME <- "inferred_tree (3)"
ATTACHMENT_FILE_NAME <- "inferred_attachment (3)"


#######################    cancer genes data    #######################
genes_location <- readRDS(paste0(path, "/cosmic_cancer_genes.RDS"))
available_cancer_types <- readRDS(paste0(path, "/available_cancer_types.RDS"))

# print available cancer types
sort(available_cancer_types)

#### USER can set cancer type of cancer from the ones printed above
# list of character cancer types
# or "no" - all cosmic cancer genes will be printed
cancer_type_input <- c("breast", "breast carcinoma", "lobular breast", "luminal A breast", "secretory breast")


all_reads <- read.table(paste0(path, corrected_counts_file), sep =",", header = T, stringsAsFactors = F)

attachment <- create_attachment_matrix(attachment_file = paste0(path, "/", ATTACHMENT_FILE_NAME))

edges <- fill_edges_matrix(create_edges_matrix(tree_file=paste0(path, "/", TREE_FILE_NAME)), attachment)

# build tree object 
tree <- tree_from_edges(edges_matrix=edges)

# tree plotting - a little time consuming
# saving the tree plot in a chosen version as a pdf
# in node label we have:
# chromosome strat breakpoint locus and end breakpoint locus
# breast cancer genes if they overlap with above genomic locations
# number of cells attached to a given node
plot_tree(tree_object = tree, output_file=paste0(path, "/tree_", "_with_genes.pdf"))


##### save tree to Newick format
sep <- "__"
edgesNewick <- edges
for (i in 1:nrow(edgesNewick)) {
  edgesNewick$child_hash[i] <- gsub(",", sep, edgesNewick$child_hash[i])
  edgesNewick$hash[i] <- gsub(",", sep, edgesNewick$hash[i])
}

# create data.tree tree object
rootNewick <- Node$new(edgesNewick$hash[1])
build_tree(rootNewick, edges = edgesNewick)

# switch to Newick
newick <- ToNewick(rootNewick, heightAttribute = NULL)
write(newick, paste0(TREE_FILE_NAME, "_Newick_"))


####################################################################################
####################################################################################

#### infer CN matrix and additional statistics
#### the most time consuming function
results <- CN_inference(tree_object=tree, cc_input=all_reads, edges_matrix=edges, attachment=attachment, only_CNs_for_heatmaps=T, main_CN = neutral_CN, max_CN = maximum_inferred_CN)


# save inferred CN matrix
CNs <- results[[1]]
CNs[1:10,1:10]
write.csv(CNs, paste0(path,"/inferred_CN_matrix_", ".csv"))

# save calculated quality measures (described in Additional File 1: Section S6.1)
# is calculated only if results were calculated with only_CNs_for_heatmaps=F
stats <- results[[3]]
write.csv(stats, paste0(path,"/CONET_quality_measures_", ".csv"))

############# heatmaps for inferred_counts
############# 
############# important set resolution


#############
############# please note that heatmap color scales work automatically only if
############# neutral_CN in {2, 3, 4} and maximum_inferred_CN <- 2*neutral_CN


image_qual <- 600

CNs_heatmap_input <- prepare_CNs_heatmap_input(CNs_data = CNs, main_CN = neutral_CN) 
CNs_heatmap <- CNs_heatmap_input[[1]]

### outside clustering for faster heatmap
row_hc = hclust(dist(CNs_heatmap))

tiff(paste0(path, "/heatmap_CNs_", ".tiff"), units="in", width=13, height=8, res=image_qual)
heatmap3(CNs_heatmap, col=CNs_heatmap_input[[5]], breaks=CNs_heatmap_input[[4]], Colv = NA, Rowv = as.dendrogram(row_hc), scale = "none", labCol = CNs_heatmap_input[[2]], cexCol = 1.5, lasCol=1, add.expr = abline(v=CNs_heatmap_input[[3]], col="black", lwd=0.2, lty=3),
  legendfun=function() showLegend(legend=c(0:(2*neutral_CN)),col=CNs_heatmap_input[[5]]))
dev.off()

############# heatmaps for corrected_counts

CCs_heatmap_input <- prepare_CCs_heatmap_input(CCs_data = all_reads, main_CN = neutral_CN)
CCs_heatmap <- CCs_heatmap_input[[1]]

  #!!!! laeve the same clustering as for CNs

tiff(paste0(path, "/heatmap_CCs_", ".tiff"), units="in", width=13, height=8, res=image_qual)

heatmap3(CCs_heatmap, col=CCs_heatmap_input[[3]], breaks=CCs_heatmap_input[[2]], Colv = NA, Rowv = as.dendrogram(row_hc), scale = "none", labCol = CNs_heatmap_input[[2]],  cexCol = 1.5, lasCol=1, add.expr = abline(v=CNs_heatmap_input[[3]], col="black", lwd=0.2, lty=3))
dev.off()
