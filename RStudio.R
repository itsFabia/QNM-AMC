library(corrplot)

# Nur starke Korrelationen behalten
strong_corr <- cor_matrix
strong_corr[abs(strong_corr) < 0.7] <- NA

corrplot(strong_corr, method = "color", type = "upper", tl.cex = 0.7, na.label = " ")

