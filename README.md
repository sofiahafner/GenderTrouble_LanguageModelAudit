# Gender Trouble in Word Embeddings
### Code Repository for Poster Presented at the NeurIPS 2024 Queer in AI Workshop

## Repo Structure:

### Probing
- `probing/probing_sex_gender`: Code for probing language models to investigate how they conceptualize sex characteristics in relation to gendered words
- `probing/probing_pathologisation_gender`: Code for probing language models to examine if and how they pathologize gender identities that fall outside a binary 
- `probing/probing_criminalisation_gender`: Code for probing language models to investigate if and how they criminalize gender identities that fall outside a binary 

### Result Analysis
- `result_analysis/analysing_sex_gender`: Code for analyzing results from the sex and gender probing
- `result_analysis/analysing_pathologisation`: Code for analyzing results from the pathologization probing
- `result_analysis/analysing_criminalisation`: Code for analyzing results from the criminalization probing

### Data
- `data/input/sex_characteristics.txt`: List of terms related to physical sex characteristics used for probing language models
- `data/input/gender_words.txt`: List of gendered terms for probing associations in language models
- `data/input/illness.txt`: Terms related to illness and pathology, used for the pathologization analysis
- `data/input/conviction.txt`: Terms related to crime and criminality, used in the criminalization analysis

- `data/output/sex_gender_prompting_output.csv`: FIle containing the output of probing_sex_gender

### Visualizations
- `visualisations/`: Folder containing visuals produced for the research paper and the poster presentation