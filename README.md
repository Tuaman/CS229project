# CS229project

Problem statement:

Given the technical data of a stock (i.e. stock price and volume), what action will maximize the profit. The set of actions are buy, sell, and hold (state the amount of money in the case of buy and sell). The profit can be calculated using the realized PnL (profit and loss) where the agent received reward when it closes a position, meaning it sells off the asset that previously bought.

Baseline:
Two simple model is used for the baseline:

1. Buy once at the start, and sell at the end

2. Moving Average Convergence Divergence (MACD)
- MACD is used by professional traders as a reference point of buy and sell. It is calculated by subtracting the 12-day exponential moving average (EMA) out of the 26-day EMA. The signal line can be calculated usind the 9-day EMA of the MACD line. When the MACD crosses above the signal line, it is the signal to buy, conversely when it crosses the signal line below, it is the signal to sell. Using this simple rule, we calculate our baseline result. 

We start from having $10000 as a starting point and buy as much as we can at the buy signals and sell as much as we can at the sell signals.

Using this baseline model, we calculate profit for three stocks that