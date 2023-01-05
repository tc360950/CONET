library(aCGH)
library(DNAcopy)
library(optparse)



option_list = list(
  make_option(c("-X", "--dataset"), type="character", default=NULL,
              help="Dataset file path. ; separated with header.", metavar="path"),
  make_option(c("-o", "--output"), type="character", default="./output",
              help="Output file path.", metavar="path"),
    make_option(c("-c", "--cn_output"), type="character", default="./output_cn",
              help="Output file for cn profiles.", metavar="path"),
  make_option(c("-N", "--neutralcn"), type="integer", default=2,
              help="Neutral copy number."),
  make_option(c("-M", "--mincells"), type="integer", default=10,
              help="CN difference must be present in at least this number of cells to be considered a breakpoint.")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$dataset)){
  print_help(opt_parser)
  stop("Pass path to dataset file", call.=FALSE)
}


MIN_CELLS <- opt$mincells
NEUTRAL_CN <- opt$neutralcn

X <- read.table(opt$dataset, sep=",", header=T)
X_segmented <- X

cells <- colnames(X)[-c(1,2,3,4,5)]


for (cell in cells) {
  CNA.object <- CNA(cbind(log2(X[cell])), X$chr, X$start, data.type="logratio",sampleid=cell)

  smoothed.CNA.object <- smooth.CNA(CNA.object)
  segment.smoothed.CNA.object <- segment(smoothed.CNA.object, verbose=1)
  out <- segment.smoothed.CNA.object$output

  for (i in 1:dim(out)[1]) {
    X_segmented[,cell][X_segmented$chr == out$chrom[i] & X_segmented$start >= out$loc.start[i] & X_segmented$start <= out$loc.end[i]] <- out$seg.mean[i]
  }

  merged <- mergeLevels(log2(X[, cell]),X_segmented[,cell],pv.thres=0.001,ansari.sign=0.05,thresMin=0.05,thresMax=0.5,verbose=1,scale=TRUE)
  X_segmented[,cell] <- 2^merged$vecMerged
}

X_segmented <- round(X_segmented[, -c(1,2,3,4,5)])
X$candidate_brkp <- 0
for (i in 1:dim(X)[1]) {
  if (i == 1 || X$chr[i] != X$chr[i-1]) {
    previous_bin <- rep(NEUTRAL_CN, dim(X_segmented)[2])
  } else {
    previous_bin <- X_segmented[i-1, ]
  }
  if (sum(X_segmented[i, ] != previous_bin) > MIN_CELLS) {
    X$candidate_brkp[i] <- 1
  }
}

print(dim(X_segmented))
print(dim(X))
write.table(X, file=opt$output, sep=",", row.names=F, quote=FALSE, col.names=T)
write.table(X_segmented, file=opt$cn_output, sep=",", row.names=F, quote=FALSE, col.names=F)