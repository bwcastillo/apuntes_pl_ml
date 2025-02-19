## **2.1.3 Probability as extension of logic**

### **2.1.3.1 Probability of a event** <br>

- Event: denoted by the binary variable $A$, as some state of the world that either holds or does not -  hold. <br>
- $Pr(A)$: denotes the probability with which you believe event $A$ is true (or the long run fraction of times that $A$ will occur).<br>

### **2.1.3.2 Probability of a conjunction of two events** <br>
- Joint probability<br>
  $Pr(A\wedge B)=Pr(A,B)$<br>
  $Pr(A,B)=Pr(A)Pr(B)$
  <br>

### **2.1.3.3 Probability of a union of two events**<br>
The probability of event $A$ or $B$ happening is given by: <br>
  $$Pr(A\vee B)=P(rA)+Pr(B)-Pr(A \wedge B)$$<br>

if the events are mutually exclusive (so they cannot happen at the same time), we get: <br>
  $$Pr(A\vee B) =Pr(A)+Pr(B)$$

### **Scripts for the Joint and Union probabilities**


```python
prob_a=float(input("Give me the prob A:"))
prob_b=float(input("Give me the prob B:"))
#Joint probability 
joint_prob= prob_a*prob_b
#Union probability
union_prob=prob_a+prob_b-(prob_a*prob_b)
union_prob_ind=prob_a+prob_b
print("The Joint probability is: ",joint_prob," and \nThe Union probability is: ",union_prob, 
      "if the the events are dependent \nbut the union probability is", union_prob_ind, "if the events are exlusive.")
```

    Give me the prob A: 0.5
    Give me the prob B: 0.3
    

    The Joint probability is:  0.15  and 
    The Union probability is:  0.65 if the the events are dependent 
    but the union probability is 0.8 if the events are exlusive.
    


```python
if union_prob > joint_prob:
    print("The union probability is bigger than the joint probability")
else:
    print("The joint probability is bigger than the union probability")
```

    The union probability is bigger than the joint probability
    

### **2.1.3.4 Conditional probability of one event given another**

We define the conditional probability of event $B$ happening given that $A$ has ocurred as follow: <br>
$$P(A|B)= \frac{Pr(A,B)}{Pr(A)}$$

### **2.1.3.5 Independence of events**
We say that event $A$ is conditionally independent of event $B$ if: <br>
$$Pr(A,B)=Pr(A)Pr(B)$$

### **2.1.3.6 Condtionally independence of events**
We say that <ins>events are</ins> conditionally independent given $C$ if:
$$Pr(A,B|C)=Pr(A|C)P(B|C)$$ 

This is written as $A \perp B|C$. Events are often dependent on each other, but may be rendered independent if we condition on the relevant intermediate variables.
