# WikiQA-based Open Domain Question Answering
Advanced NLP major project: Open domain question answering on WikiQA corpus

[Presentation link](https://docs.google.com/presentation/d/1ir22FmPh3d-rEY24vr2MzvYMIW3QHN8iESVevjf4mF8/edit?usp=sharing)

In this project, we study the problem of Answer Selection, a subproblem in Open Domain Question Answering using the WikiQA dataset. We implement three methods on the problem. Our literature survey, analysis and findings can be found under the reports folder as well as the presentation link shared above.

To try out the code here:

## Attentive Pooling Networks
[This](https://github.com/sagarsj42/wikiqa_open_domain_qa/blob/main/notebooks/Attentive_Pooling_Networks.ipynb) notebook contains the implementation of Attentive Pooling Networks using Bi-LSTMs and CNNs. This architecture was proposed in [this](https://arxiv.org/abs/1602.03609) paper for computation of question and answer representations with mutual influence in each.

## BERT Finetuning
A simple BERT finetuning beats the baseline established by Attentive Pooling Networks. [This](https://github.com/sagarsj42/wikiqa_open_domain_qa/blob/main/notebooks/bert_finetuning.ipynb) notebook can be used to try the BERT-based modeling of the problem.

## TANDA
The [**T**ransfer **AND** **A**dapt (TANDA)](https://ojs.aaai.org//index.php/AAAI/article/view/6282) strategy, which is currently a SoTA on the problem, proposed a 2-stage finetuning on top of a pretrained architecture like BERT:
1. Transferring the capability of the model with the use of a large, generic task-specific dataset to learn the task
2. Domain-adaptive finetuning on the downstream task-specific dataset for few shot learning
[This](https://github.com/sagarsj42/wikiqa_open_domain_qa/blob/main/notebooks/tanda.ipynb) notebook contains the code for the same. A runnable Python script can be found [here](https://github.com/sagarsj42/wikiqa_open_domain_qa/blob/main/codes/tanda.py).
