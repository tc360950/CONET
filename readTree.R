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


cn <- read.table("./cn_cbs", sep =",", header = F, stringsAsFactors = F)
cn_inf <- read.table("./inferred_counts", sep=",", header=T, stringsAsFactors = F)
cc <- read.table("./cc", sep=",", header=T, stringsAsFactors = F)

cn <- cbind(cc[,1:5], cn)
cc <- cbind(cc[,1:5], cn_inf)
# preparing CN input
prepare_CNs_heatmap_input <- function(CNs_data, main_CN) {
  CNs_heatmap <- as.matrix(transpose(CNs_data[,6:ncol(CNs_data)]))
  
  rownames(CNs_heatmap) <- colnames(CNs_data)[6:ncol(CNs_data)]
  colnames(CNs_heatmap) <- CNs_data$chr
  
  x_labels <- c("1")
  x_ticks <- c(1)
  
  
  for (i in 1:(ncol(CNs_heatmap)-1)) {
    
    if (colnames(CNs_heatmap)[i] != colnames(CNs_heatmap)[i+1])
    {
      x_labels <- append(x_labels, colnames(CNs_heatmap)[i+1])
      x_ticks <- append(x_ticks, i+1)
    }
    else {
      x_labels <- append(x_labels, NA)
      
      
    }
  }
  
  x_labels[which(x_labels==23)] <-"X"
  x_labels[which(x_labels==24)] <-"Y"
  
  ### color palette
  ### 
  if (main_CN==2) {
    col_breaks <- seq(-0.5, 4.5, by = 1)
    my_palette <- c("#273871", "#8FB7D9", "white", "#FF8870","red4")
  }
  else if (main_CN==3) {
    col_breaks <- seq(-0.5, 6.5, by = 1)
    my_palette <- c("#273871", "steelblue", "lightskyblue2", "white", "#FF8870", "#FF0000", "red4")
  }
  
  else if (main_CN==4) {
    col_breaks <- seq(-0.5, 8.5, by = 1)
    my_palette <- c("#273871","royalblue4", "steelblue", "lightskyblue2", "white", "#FF8870", "#FF0000", "firebrick", "red4")
  }
  
  
  return(list(CNs_heatmap, x_labels, x_ticks, col_breaks, my_palette))
}



prepare_CCs_heatmap_input <- function(CCs_data, main_CN) {
  CCs_heatmap <- as.matrix(transpose(CCs_data[,6:ncol(CCs_data)]))
  max_CN <- 2*main_CN
  
  CCs_heatmap[CCs_heatmap>(max_CN-0.1)] <- (max_CN-0.1)
  
  rownames(CCs_heatmap) <- colnames(CCs_data)[6:ncol(CCs_data)]
  colnames(CCs_heatmap) <- CCs_data$chr
  
  col_breaks_2 <- seq(0, max_CN, by = 0.2)
  n_col <- length(col_breaks_2)-1
  
  my_palette_2 <- c(hcl.colors(n_col/2, palette = "Blues"), 
                    hcl.colors(n_col/2, palette = "Reds", rev=-1))
  
  return(list(CCs_heatmap, col_breaks_2, my_palette_2))
}

## set basal ploidy
neutral_CN <- 2

## set maximum inferred integer CN
## if higher than 2*neutral_CN the colors in heatmaps may not reflect higher CNs
maximum_inferred_CN <- 2*neutral_CN


image_qual <- 600

CNs_heatmap_input <- prepare_CNs_heatmap_input(CNs_data = cn, main_CN = neutral_CN) 
CNs_heatmap <- CNs_heatmap_input[[1]]

### outside clustering for faster heatmap
row_hc = hclust(dist(CNs_heatmap))

tiff("./heatmap_CNs.tiff", units="in", width=13, height=8, res=image_qual)
heatmap3(CNs_heatmap, col=CNs_heatmap_input[[5]], breaks=CNs_heatmap_input[[4]], Colv = NA, Rowv = as.dendrogram(row_hc), scale = "none", labCol = CNs_heatmap_input[[2]], cexCol = 1.5, lasCol=1, add.expr = abline(v=CNs_heatmap_input[[3]], col="black", lwd=0.2, lty=3),
  legendfun=function() showLegend(legend=c(0:(2*neutral_CN)),col=CNs_heatmap_input[[5]]))
dev.off()

############# heatmaps for corrected_counts

CCs_heatmap_input <- prepare_CCs_heatmap_input(CCs_data = cc, main_CN = neutral_CN)
CCs_heatmap <- CCs_heatmap_input[[1]]

  #!!!! laeve the same clustering as for CNs

tiff( "./heatmap_CCs_.tiff",  units="in", width=13, height=8, res=image_qual)

heatmap3(CCs_heatmap, col=CCs_heatmap_input[[3]], breaks=CCs_heatmap_input[[2]], Colv = NA, Rowv = as.dendrogram(row_hc), scale = "none", labCol = CNs_heatmap_input[[2]],  cexCol = 1.5, lasCol=1, add.expr = abline(v=CNs_heatmap_input[[3]], col="black", lwd=0.2, lty=3))
dev.off()
