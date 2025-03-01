### **Confusion Matrix for Multi-Class Classification**

In a **multi-class classification problem**, we have more than two possible classes. A confusion matrix for multi-class classification extends beyond the binary case by tracking the predictions for multiple classes.

---

### **Example: Classifying Animals (Dog, Cat, Rabbit)**
Suppose we have a model that classifies images into three categories:
- **Class 0** → Dog 🐶  
- **Class 1** → Cat 🐱  
- **Class 2** → Rabbit 🐰  

After testing our model on 20 images, we obtain the following confusion matrix:

| **Actual \ Predicted** | **Dog (0)** | **Cat (1)** | **Rabbit (2)** |
|------------------------|------------|------------|-------------|
| **Dog (0)**    | **8** (TP for Dog)  | 2 (Misclassified as Cat)  | 1 (Misclassified as Rabbit) |
| **Cat (1)**    | 3 (Misclassified as Dog) | **7** (TP for Cat)  | 2 (Misclassified as Rabbit) |
| **Rabbit (2)** | 0 (Misclassified as Dog) | 2 (Misclassified as Cat) | **5** (TP for Rabbit) |

---

### **Interpreting the Multi-Class Confusion Matrix**
Each row represents **actual classes**, and each column represents **predicted classes**.

- **Diagonal elements (bold numbers)** → **Correct predictions (True Positives for each class).**  
- **Off-diagonal elements** → **Misclassifications (False Positives and False Negatives for different classes).**  

---

### **Calculating Metrics for Each Class**
For **multi-class problems**, TP, FP, FN, and TN must be computed **per class**.

#### **1. TP, FP, FN for Each Class**
For **Dog (Class 0)**:
- **TP (True Positive)** = 8 (Correctly predicted as Dog)
- **FN (False Negative)** = 3 + 0 = 3 (Actual Dog misclassified as Cat or Rabbit)
- **FP (False Positive)** = 3 + 0 = 3 (Predicted Dog, but was actually Cat or Rabbit)

For **Cat (Class 1)**:
- **TP** = 7  
- **FN** = 2 + 2 = 4  
- **FP** = 2 + 2 = 4  

For **Rabbit (Class 2)**:
- **TP** = 5  
- **FN** = 1 + 2 = 3  
- **FP** = 2 + 2 = 4  

---

### **Key Metrics for Multi-Class Classification**
For **each class**, we calculate **Precision, Recall, and F1-score**:

#### **1. Precision (Per Class)**
Measures how many of the predicted instances for a class were correct.
\[
\text{Precision} = \frac{TP}{TP + FP}
\]
- Precision for **Dog** = **\( 8 / (8 + 3) = 0.73 \)**
- Precision for **Cat** = **\( 7 / (7 + 4) = 0.64 \)**
- Precision for **Rabbit** = **\( 5 / (5 + 4) = 0.56 \)**

#### **2. Recall (Per Class)**
Measures how many actual instances of a class were correctly predicted.
\[
\text{Recall} = \frac{TP}{TP + FN}
\]
- Recall for **Dog** = **\( 8 / (8 + 3) = 0.73 \)**
- Recall for **Cat** = **\( 7 / (7 + 4) = 0.64 \)**
- Recall for **Rabbit** = **\( 5 / (5 + 3) = 0.62 \)**

#### **3. F1-Score (Per Class)**
Harmonic mean of precision and recall.
\[
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]
- F1-score for **Dog** = **\( 2 \times \frac{0.73 \times 0.73}{0.73 + 0.73} = 0.73 \)**
- F1-score for **Cat** = **\( 2 \times \frac{0.64 \times 0.64}{0.64 + 0.64} = 0.64 \)**
- F1-score for **Rabbit** = **\( 2 \times \frac{0.56 \times 0.62}{0.56 + 0.62} = 0.59 \)**
  
---

### **How to Compute This in Python?**
You can use **Scikit-learn** to calculate the confusion matrix and classification report:

```python
from sklearn.metrics import confusion_matrix, classification_report

# Example actual labels
y_true = [0, 0, 1, 2, 2, 1, 0, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 0, 0, 1]

# Example predicted labels
y_pred = [0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 2, 1, 1, 0, 2, 2, 1, 0, 0, 1]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute Precision, Recall, and F1-score
report = classification_report(y_true, y_pred, target_names=["Dog", "Cat", "Rabbit"])
print("\nClassification Report:")
print(report)
```

**Output Example:**
```
Confusion Matrix:
[[8 2 1]
 [3 7 2]
 [0 2 5]]

Classification Report:
              precision    recall  f1-score   support

        Dog       0.73      0.73      0.73        11
        Cat       0.64      0.64      0.64        11
     Rabbit       0.56      0.62      0.59         8

   Accuracy:  0.7
```

---

### **Summary**
- **Multi-class confusion matrices extend binary confusion matrices to multiple categories.**
- **Each class has its own TP, FP, FN, and TN values.**
- **Precision, Recall, and F1-score are calculated per class.**
- **Scikit-learn makes it easy to compute these metrics.**

Would you like to explore **macro-averaged vs. weighted-averaged scores** for multi-class problems? 🚀