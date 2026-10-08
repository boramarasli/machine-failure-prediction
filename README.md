# Machine-Failure Prediction

Oct 2026 · Python · scikit-learn · AI-assisted

Which machine runs end in a failure? Predicted from five sensor readings, with the alarm level set by what each mistake costs.

Data: the AI4I 2020 Predictive Maintenance Dataset from UCI (`ai4i2020.csv`): 10,000 milling machine runs, 339 of them failures (3.4%).

## Results

| Model | Average precision |
|---|---|
| Random forest, raw sensors | 0.78 |
| Random forest, with physics features | 0.90 |
| Logistic regression, with physics features | 0.50 |

5-fold cross-validation on the training data (80%). The test data (20%) is used once, at the end: average precision 0.89.

| Alarm level | Failures caught (of 68) | False alarms | Cost |
|---|---|---|---|
| 0.50, the default | 56 | 2 | 60,400 € |
| 0.03, chosen by cost | 64 | 138 | 47,600 € |

21% cheaper, and 8 more failures caught. The costs are assumptions: 5,000 € for a breakdown nobody saw coming, 200 € for checking a machine that was fine.

## Why not accuracy

96.6% of runs don't fail. A "model" that always says "no failure" is right 96.6% of the time and catches nothing. Average precision looks only at how well the real failures are ranked at the top.

## The steps

1. Look at the data: 339 failures in 10,000 runs.
2. Clean and split. Drop the IDs and the five failure-type columns (TWF, HDF, PWF, OSF, RNF): they're only known after a failure, so using them would be cheating. Keep 20% aside for the end.
3. Random forest on the raw sensors.
4. Add physics: power (torque × angular speed), the gap between process and air temperature, and strain (tool wear × torque). Three of the dataset's failure types are defined using these.
5. A simpler model, logistic regression, on the same inputs.
6. Pick the cheapest alarm level on the training data only. Then the test data, once.

## Run it

```
pip3 install pandas numpy scikit-learn
python3 pm.py
```

---

Bora Marasli · Information Technology in Mechanical Engineering, TU Berlin · [LinkedIn](https://www.linkedin.com/in/boramarasli) · [More projects](https://github.com/boramarasli)
