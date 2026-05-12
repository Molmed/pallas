data <- function(gex_file, dnam_file, labels_file) {
    # --- Read in data ---
    X_gex  <- read.csv(gex_file,    row.names = 1, check.names = FALSE)
    X_dnam <- read.csv(dnam_file,   row.names = 1, check.names = FALSE)
    labels <- read.csv(labels_file, row.names = 1)

    # --- Extract Y as a factor ---
    # Replace "subtype" with your actual column name
    Y <- factor(labels[, "class"])

    # --- Align sample order across all three ---
    # (critical — mixOmics assumes rows are in the same order)
    common_samples <- Reduce(intersect, list(rownames(X_gex),
                                            rownames(X_dnam),
                                            rownames(labels)))

    X_gex  <- as.matrix(X_gex[common_samples, ])
    X_dnam <- as.matrix(X_dnam[common_samples, ])
    Y      <- Y[match(common_samples, rownames(labels))]

    # --- Sanity checks ---
    stopifnot(nrow(X_gex)  == length(Y))
    stopifnot(nrow(X_dnam) == length(Y))
    stopifnot(all(rownames(X_gex) == rownames(X_dnam)))

    cat("Samples:", length(Y), "\n")
    cat("GEX features:", ncol(X_gex), "\n")
    cat("DNAm features:", ncol(X_dnam), "\n")
    cat("Subtypes:", levels(Y), "\n")
    return(list(X_gex = X_gex, X_dnam = X_dnam, Y = Y))
}
