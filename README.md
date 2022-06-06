## Mitigating Algorithmic Bias with Limited Annotations

### Research Motivation

Existing work on fairness modeling commonly assumes that sensitive attributes are fully available for all instances, which may not hold in many real-world applications due to the high cost of acquiring sensitive information. 
When sensitive attributes are not disclosed or available, it is in need to manually annotate some sensitive attributes as part of the training data for bias mitigation.
However, selecting appropriate instances for annotation is a nontrivial task, since skewed distributions across sensitive groups lead to a sub-optimal solution which still preserves discrimination. 
In this work, we propose **APOD**, an end-to-end framework to actively select a small portion of representative instances for annotation and maximally mitigate algorithmic bias with limited annotated sensitive information. 

### Research Challenge

An example of binary classification task (e.g. positive class denoted as gray + and •, negative class as red + and •) with two sensitive groups shown in the following figure.
In the left-side figure, the positive instances (gray +) is significantly less than negative instances (red +) in group 0, which leads to a classification boundary deviated from perfect fair boundary.
An intuitive way to annotate sensitive attributes is through random selection. 
The randomly selected instances follow the same skewed distribution across sensitive groups, which still preserve the bias information in the classification model, as shown in the middle figure.

<div align=center>
<img width="250" height="170" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/unfair_classification-cropped.png">
<img width="250" height="170" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/RS_debias-cropped.png">
<img width="250" height="170" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/Global_optimal_debias.png">
</div>

### APOD Framework

As shown in the following figure, APOD integrates penalization of discrimination (POD) and active instance selection (AIS) in a unified and iterative framework.
Specifically, in each iteration, POD focus on the debiasing of classifier f on the partially annotated dataset (x, y, a) ∈ S and (x, y) ∈ U; 
while AIS selects the optimal instance (x*, y*)
from the unannotated dataset U that can promote the bias mitigation.
The sensitive attribute of selected instance will be annotated by human experts: (x*, y*) → (x*, y*, a*).
After that, the instance will be moved from the unannotated dataset U ← U\\{(x*, y*)} to the annotated dataset S ← S ∪ {(x*, y*, a*)} for debiasing the model in the next iteration.


<div align=center>
<img width="400" height="270" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/active_fairness-cropped.png">
</div>


### Dependency:
````angular2html
torch >= 1.9.0
scikit-learn >= 0.24.2
````

### Train APOD and baseline methods on the MEPS dataset:
````angular2html
bash script/apd/medical.sh
bash script/fal/medical_fal.sh
bash script/DRO/medical_DRO.sh
bash script/lff/medical_lff.sh
````

### Estimate the Equality of Opportunity of APOD and baseline methods on the testing dataset:
````angular2html
cd test_script
python apd_test_eop.py
python fal_test.py
python lff_test.py
cd ../
````

### Accuracy-Fairness plot
````angular2html
cd plot
python acc_eop_plot_sota.py
cd ../
````

### Reproduce our experiment results:

#### Accuracy-Fairness curve &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Fairness versus Annotation ratio
<div align=center>
<img width="400" height="300" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/ACC_vs_EOP_medical_SOTA.png">
<img width="400" height="300" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/EO_vs_label_medical.png">
</div>

#### Annotated instanced visualization
<div align=center>
<img width="400" height="300" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/apd_labelset_show_Y.png">
<img width="400" height="300" src="https://github.com/guanchuwang/APOD-fairness/blob/main/figure/apd_labelset_show_Z.png">
</div>

