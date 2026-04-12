---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
# {{ card_data }}
---

# Model Card for BINND v1.0.0

BINND (Binding and Interaction Neural Network for DNA), is a Convolutional Neural Network for DNA-DNA binding prediction targeting interactions between dissimilar sequences.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
BINND is a deep learning-based classifier designed to predict DNA-DNA hybridization with high accuracy and speed, even in "messy" molecular environments far from standard orthogonal sequence spaces. It addresses the computational intensity and accuracy drops experienced by traditional thermodynamic models when dealing with highly dissimilar sequence pairings.

- **Developed by:** Gunavaran Brihadiswaran, Karishma Matange, Kyle J. Tomek, Kevin Volkel, Doug Townsend, Albert J. Keung, James M. Tuck
<!-- - **Funded by [optional]:** {{ funded_by | default("[More Information Needed]", true)}} -->
<!-- - **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}} -->
- **Model type:** Convolutional Neural Network
<!-- - **Language(s) (NLP):** {{ language | default("[More Information Needed]", true)}} -->
- **License:** MIT
<!-- - **Finetuned from model [optional]:** {{ base_model | default("[More Information Needed]", true)}} -->

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/dna-storage/BINND
<!-- - **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}} -->
- **Demo:** https://github.com/dna-storage/BINND/blob/main/examples/BINND_demo.ipynb

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->


### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The BINND model is primarily intended for the rapid and accurate prediction of DNA-DNA hybridization across a wide sequence space, specifically focusing on ~20mer DNA sequences. It is designed to function even in "messy" molecular environments far from standard orthogonal sequence spaces, where conventional thermodynamic models often fail. Foreseeable users include researchers in genomics, synthetic biology, and molecular bioengineering.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

BINND is optimized for high-volume inference, making it suitable for integration into larger ecosystems for
- Engineering Hyperconnected Networks: Designing complex, non-orthogonal DNA interaction networks where single sequences act as highly connected hubs.
- Bioengineering & Nanotechnology: Design of hybridization sites for PCR, DNA origami, and diagnostic probes.
- Information Systems: Implementation of sophisticated search, random access, and computational motifs in DNA-based data storage and computation.
- Diagnostics: Identifying pathogen biomarkers and developing point-of-care diagnostic tools.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

- Non-DNA Interactions: The model is specifically trained on DNA-DNA hybridization and is not currently validated for RNA or hybrid nucleic acid systems (e.g., DNA-RNA).
- In Vivo Environments: All models were trained and evaluated using in vitro interaction datasets. While they capture meaningful principles, they may not account for the additional biological constraints found in living cells.
- Alternative Sequence Lengths: The current version is optimized for ~20mer sequences. Use with significantly different sequence lengths without extending the encoding scheme (e.g., via zero-padding) is out of scope.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

BINND (v1.0.0) is designed specifically for 20-mer sequence pairs. This limitation is inherent to the current one-hot encoding scheme and the fixed-size input layer of the CNN architecture. Inputs exceeding or falling short of this length are not supported in the current release.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->
To support variable-length sequences, the encoding module should be configured to the maximum expected length, with shorter sequences handled via zero-padding and masking to maintain a consistent input dimensionality for the CNN.


## How to Get Started with the Model

The Jupyter Notebook available at https://github.com/dna-storage/BINND/blob/main/examples/BINND_demo.ipynb is a good starting point.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training data consists of 144.5 million unique sequence pairs obtained from a high-throughput wet-lab platform. The dataset is balanced with an equal number of bound and unbound sequences. These sequences represent a sparse sampling (~0.01%) of the theoretical trillion-sequence space of 20-mers.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The dataset was randomly split into training, validation, and test sets in an 8:1:1 ratio. An interleaved one-hot encoding was used to convert input sequence pairs into input matrices.

#### Training Hyperparameters

- Optimizer: Adam.
- Loss Function: Binary cross-entropy loss.
- Learning Rate: 0.0004.
- Batch Size: 512.
- Epochs: 15 (with an early stopping patience of 2)

<!-- #### Speeds, Sizes, Times [optional] -->

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

<!-- {{ speeds_sizes_times | default("[More Information Needed]", true)}} -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data
The test set comprises approximately 14.5 million sequence pairs (10% of the total generated dataset).

<!-- This should link to a Dataset Card if possible. -->

<!-- {{ testing_data | default("[More Information Needed]", true)}} -->

<!-- #### Factors -->

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

<!-- {{ testing_factors | default("[More Information Needed]", true)}} -->

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- Accuracy: Percentage of correct classifications at a fixed threshold (0.5).
- AUC-ROC: Area under the Receiver Operating Characteristic curve to measure discriminatory power independent of a threshold.
- Precision, Recall and Confusion Matrix
- Execution time and memory usage: Measured in comparison to state-of-the-art thermodynamic models.


### Results

- Overall Accuracy: 83%.
- AUC: 0.88.
- 100% accuracy on perfectly complementary sequences not seen during training.

#### Summary

BINND significantly outperforms traditional models like NUPACK and Primer3 by at least 10% in classification accuracy and provides superior discrimination between bound and unbound populations.

<!-- ## Model Examination [optional] -->

<!-- Relevant interpretability work for the model goes here -->

<!-- {{ model_examination | default("[More Information Needed]", true)}} -->

<!-- ## Environmental Impact -->

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

<!-- Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). -->

<!-- - **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}} -->
<!-- - **Hours used:** {{ hours_used | default("[More Information Needed]", true)}} -->
<!-- - **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}} -->
<!-- - **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}} -->
<!-- - **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}} -->

<!-- ## Technical Specifications [optional] -->

<!-- ### Model Architecture and Objective -->

<!-- {{ model_specs | default("[More Information Needed]", true)}} -->

<!-- ### Compute Infrastructure -->

<!-- {{ compute_infrastructure | default("[More Information Needed]", true)}} -->

<!-- #### Hardware -->

<!-- {{ hardware_requirements | default("[More Information Needed]", true)}} -->

<!-- #### Software -->

<!-- {{ software | default("[More Information Needed]", true)}} -->

<!-- ## Citation [optional] -->

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

<!-- **BibTeX:** -->

<!-- {{ citation_bibtex | default("[More Information Needed]", true)}} -->

<!-- **APA:** -->

<!-- {{ citation_apa | default("[More Information Needed]", true)}} -->

<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

<!-- {{ glossary | default("[More Information Needed]", true)}} -->

<!-- ## More Information [optional] -->

<!-- {{ more_information | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Authors [optional] -->

<!-- {{ model_card_authors | default("[More Information Needed]", true)}} -->

<!-- ## Model Card Contact -->

<!-- {{ model_card_contact | default("[More Information Needed]", true)}} -->
