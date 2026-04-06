###identify missing patients
library(dplyr)

data500 <- read.csv("~/Desktop/ivf_ml_research/data/raw/patientdata_fresh1.csv", header = TRUE, sep = ",")
data100 <- read.csv("~/Desktop/ivf_ml_research/data/processed/freshdata_cleaned1.csv", header = TRUE, sep = ",")
data <- read.csv("~/Desktop/ivf_ml_research/data/raw/Final_Fresh_Merged.csv", header = TRUE, sep = ",")

data500 <- data500 %>%
  rename(Unique_number = Unique.Number)
data <- data %>%
  rename(Unique_number = UniqueNbr)


repeated_ids <- data500 %>%
  count(Unique_number) %>%
  filter(n > 1)

#unique IDs
ids_500 <- data500 %>% pull(Unique_number) %>% unique()
ids_100 <- data100 %>% pull(Unique_number) %>% unique()
ids <- data %>% pull(Unique_number) %>% unique()

#patients missing before
missing_before <- setdiff(ids_500, ids_100)

##check which overlap from new dataset with those i already had
overlap_with_100 <- intersect(ids, ids_100)

#new patients
new_from_missing <- intersect(ids, missing_before)

#check
outside_500 <- setdiff(ids, ids_500) #0

cat("Total in data500:", length(ids_500), "\n") #510
cat("Had data before (data100):", length(ids_100), "\n") #89
cat("Missing before:", length(missing_before), "\n\n") #421

cat("Unique patients in new data:", length(ids), "\n") #226
cat("Overlap with original 100:", length(overlap_with_100), "\n") #89 overlap with original, all patients I had are in the new data
cat("New patients (previously missing):", length(new_from_missing), "\n") #137 new patient data
cat("Outside cohort (should be 0):", length(outside_500), "\n") #0 


final_ids_with_data <- union(ids_100, ids) #226

#stats
coverage_before <- length(ids_100) / length(ids_500)
coverage_after  <- length(final_ids_with_data) / length(ids_500)
cat("Coverage before:", round(coverage_before * 100, 2), "%\n")
cat("Coverage after:", round(coverage_after * 100, 2), "%\n")

##get missing
ids_still_missing <- setdiff(ids_500, final_ids_with_data)

write.csv(
  data.frame(Unique_number = ids_still_missing),
  "patientsIDs_missing_data.csv",
  row.names = FALSE
)