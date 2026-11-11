# Independent reference calculations for the external validation corpus.
# The legacy ESS port follows Stan v2.24.0's
# src/stan/analyze/mcmc/compute_effective_sample_size.hpp, using R's acf()
# for independently computed biased autocovariances. See test/external/licenses.
reference_lib <- Sys.getenv("MCMC_VALIDATION_R_LIBRARY")
if (!nzchar(reference_lib)) stop("Run this reference through scripts/validation/generate.py")
.libPaths(c(reference_lib, .libPaths()))
suppressPackageStartupMessages(library(posterior, lib.loc = reference_lib))
stopifnot(as.character(packageVersion("posterior", lib.loc = reference_lib)) == "1.7.1")
stopifnot(normalizePath(getNamespaceInfo(asNamespace("posterior"), "path")) ==
          normalizePath(file.path(reference_lib, "posterior")))
stopifnot(as.character(getRversion()) == "4.5.2")
stopifnot(as.character(packageVersion("matrixStats")) == "1.5.0")

write_precise_csv <- function(x, path) {
  lines <- apply(x, 1, function(row) {
    paste(vapply(row, function(value) {
      if (is.numeric(value)) sprintf("%.17g", value) else as.character(value)
    }, character(1)), collapse = ",")
  })
  writeLines(c(paste(colnames(x), collapse = ","), lines), path, useBytes = TRUE)
}

split_legacy <- function(x) {
  n <- nrow(x)
  half <- n %/% 2L
  cbind(x[seq_len(half), , drop = FALSE],
        x[(n - half + 1L):n, , drop = FALSE])
}

legacy_ess <- function(x) {
  n <- nrow(x)
  m <- ncol(x)
  stopifnot(n >= 4L, all(is.finite(x)))
  acov <- vapply(seq_len(m), function(chain) {
    as.numeric(stats::acf(x[, chain], lag.max = n - 1L,
                         type = "covariance", plot = FALSE)$acf)
  }, numeric(n))
  mean_var <- mean(acov[1L, ]) * n / (n - 1L)
  var_plus <- mean_var * (n - 1L) / n
  if (m > 1L) var_plus <- var_plus + var(colMeans(x))
  rho <- rep(0, n)
  even <- 1
  odd <- 1 - (mean_var - mean(acov[2L, ])) / var_plus
  rho[1L:2L] <- c(even, odd)
  s <- 1L
  while (s < n - 4L && even + odd > 0) {
    even <- 1 - (mean_var - mean(acov[s + 2L, ])) / var_plus
    odd <- 1 - (mean_var - mean(acov[s + 3L, ])) / var_plus
    if (even + odd >= 0) {
      rho[s + 2L] <- even
      rho[s + 3L] <- odd
    }
    s <- s + 2L
  }
  max_s <- s
  if (even > 0) rho[max_s + 2L] <- even
  s <- 1L
  while (max_s >= 3L && s <= max_s - 3L) {
    if (rho[s + 2L] + rho[s + 3L] > rho[s] + rho[s + 1L]) {
      rho[s + 2L] <- (rho[s] + rho[s + 1L]) / 2
      rho[s + 3L] <- rho[s + 2L]
    }
    s <- s + 2L
  }
  tau <- -1 + 2 * sum(rho[seq_len(max_s)]) + rho[max_s + 2L]
  if (!is.finite(tau)) return(NA_real_)
  min(m * n / tau, m * n * log10(m * n))
}

args <- commandArgs(trailingOnly = TRUE)
if (args[1L] == "export") {
  # export EXAMPLE OUTPUT ALIAS SOURCE_VARIABLE [ALIAS SOURCE_VARIABLE ...]
  draws <- as.array(example_draws(args[2L]))
  mapping <- matrix(args[-(1L:3L)], ncol = 2L, byrow = TRUE)
  n <- dim(draws)[1L]
  m <- dim(draws)[2L]
  out <- cbind(chain = rep(seq_len(m), each = n),
               draw = rep(seq_len(n), times = m))
  for (i in seq_len(nrow(mapping))) {
    values <- draws[, , mapping[i, 2L]]
    out <- cbind(out, as.vector(values))
    colnames(out)[ncol(out)] <- mapping[i, 1L]
  }
  write_precise_csv(out, args[3L])
} else if (args[1L] == "reference") {
  # reference OUTPUT FIXTURE.csv [FIXTURE.csv ...]
  rows <- list()
  for (path in args[-(1L:2L)]) {
    data <- read.csv(path, check.names = FALSE, comment.char = "#")
    fixture <- sub("[.]csv$", "", basename(path))
    for (parameter in names(data)[-(1L:2L)]) {
      x <- do.call(cbind, split(data[[parameter]], data$chain))
      stopifnot(all(is.finite(x)))
      ess <- legacy_ess(x)
      split_ess <- legacy_ess(split_legacy(x))
      current <- suppressWarnings(ess_basic(x, split = FALSE))
      current_split <- suppressWarnings(ess_basic(x, split = TRUE))
      row <- data.frame(
        fixture = fixture, parameter = parameter,
        draws = nrow(x), chains = ncol(x),
        rhat = rhat_basic(x, split = FALSE),
        split_rhat = rhat_basic(x, split = TRUE),
        legacy_ess = ess, legacy_split_ess = split_ess,
        legacy_mcse = sd(as.vector(x)) / sqrt(ess),
        posterior_ess = current, posterior_split_ess = current_split,
        posterior_mcse_unsplit = sd(as.vector(x)) / sqrt(current),
        posterior_mcse_mean = suppressWarnings(mcse_mean(x))
      )
      rows[[length(rows) + 1L]] <- row
    }
  }
  # Format numbers before binding character identifiers; preserve all 17 digits.
  out <- do.call(rbind, rows)
  for (name in names(out)) {
    if (is.numeric(out[[name]])) out[[name]] <- sprintf("%.17g", out[[name]])
  }
  write_precise_csv(out, args[2L])

  # Synthetic probes document boundary differences; they are not counted as
  # external model datasets. Keep their inputs aligned with tests/external.rs.
  probes <- list(
    negative_tau = matrix(c(-1, -1, -1, 1, -1, 0), ncol = 1L),
    positive_tail_cutoff = cbind(c(1, 3, 2, 5, 4, 7, 6, 8),
                                 c(21, 23, 22, 25, 24, 27, 26, 28))
  )
  semantics <- do.call(rbind, lapply(names(probes), function(name) {
    x <- probes[[name]]
    legacy <- legacy_ess(x)
    data.frame(probe = name,
      stan_2_24_ess = sprintf("%.17g", legacy),
      crate_outcome = if (is.na(legacy) || legacy <= 0) "error" else "value",
      posterior_ess = sprintf("%.17g", suppressWarnings(ess_basic(x, split = FALSE))))
  }))
  write_precise_csv(semantics, file.path(dirname(args[2L]), "semantics.csv"))
} else {
  stop("Expected export or reference mode")
}
