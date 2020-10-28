# MRAM 
MRAM is a mixed RNN and Attention model for locating faulty methods when given a bug report.

## Code revision Graph

The code revision grap is built from past code commits and bug reports,  
which not only reveal the latentrelationships among methods to expand short methods but also provide all revisions of code and past  fixes to calculate more accurate method structured features and bug-fixing features.
With code revision graphs, MRAM can expand short methods with their related methods.
More implementation details are available [here](https://github.com/ysliang0108/MRAM/tree/main/graph).

## MRAM
Fig.1 shows the overall architecture of MRAM, consistingof three main components:
(1) SMNN (semantic matching network),  which uses bidi-rectional RNNs and soft attention to capture both semanticand structural information of source method so that it can bematched with bug report accurately in a unified vector space.
(2) MENN (method expansion network), which enriches the representation of a method with short length by retrieving theinformation from its relevant methods.
(3) FLNN (fault localization network), which predicts faultprobability of a method by combining both its implicit reference and explicit relevance to the bug report.

The high-level goal of MRAM is: given a bug report R and an arbitrary method M with its corresponding features,
MRAM predicts a high relevance score if M is a faulty method for R, and a low relevance score otherwise.
More implementation details are available [here](https://github.com/ysliang0108/MRAM/tree/main/model).

## dataset
We have evaluated on five open source projects (AspectJ, Birt, JDT, SWT, and Tomcat).
The dataset relevant to the experiment are available [here](https://jbox.sjtu.edu.cn/l/aoMeGs).
