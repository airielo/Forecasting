```python
sales = pd.read_csv("sales.csv")

t = TS(sales)
t.preprocess()
t.predict()
t.assess()
t.plot()

t.MAPE
```
