PROJECT DESCRIPTION:

Summary:
The program will take user input of stocks and start and end dates. It will then conduct visualisation, risk analysis, and a trading strategy based on the tickers inputted.
The program also conducts Machine learning on the tickers for Apple, Google, Microsoft and IBM data from the dates 
Visualisation:
Plotting the histogram of daily percentage change for each input stock
Plotting rolling time window for 50 days and 200 days with day to day price movement
Plotting the volatility of the stocks
plot the adjusted close
plot the volume
Adjusted close moving average graph
The relationship between each stock and the ETF 

Risk Analysis:
Expected return and risk for each stock is calculated.
The VAR model is used to find the maximum losses by holding a given stock.
Choosing a Portfolio:
	a) Standard deviation of the portfolio with all stocks equally 
	b) Finding the portfolio that that:
		- Minimises the risk 
		- Maximises the sharpe ratio

Trading Strategy:
The strategy: The moving average cross trading strategy is used. According to this strategy, a long window and a short window moving average return is calculated. When the two are equal, the stock must be bought or sold.
Testing the strategy: The gains or losses made when this strategy is implemented is calculated using backtesting.Evaluating the strategy:
The CAGR of this strategy is shown.

Machine Learning: Done on stocks for Amazon, Google, Microsoft, IBM.

Prediction:  Predict the performance of the ETF stock for technology industry using APPLE, IBM and MICROSOFT stock data.
Using Learning Regression, Log-it Regression, Random Forest and Neutral Network:
Dependent Variable: A binary variable indicating whether the close price is greater than the open price for the ETF stock
Independent Variables: Close price and Open price for each individual stock
Results: The logistic regression generates the best result. We correct prediction with probability above 80%

Linear Regression:
Linear Regression 1 shows that the returns of individual stocks have a statistically significant impact on the return of the ETF at 5% significance level.
Linear Regression 2 shows that the close price of individual stocks have a statistically significant impact on the close price of ETF at 5% significance level.

Time Series Analysis
The Moving Average Model results show that for each stock, moving average model would be a good fit since data are normalized and no heavy tail problem.
The results here lead to the following moving average model.

GROUP NAME AND SECTION:
Project Name: TBD
Section: 2

INSTALLATION INSTRUCTIONS:

RUN INSTRUCTIONS:
User Input: 
User must input number of stocks.
User must input _______ ticker names.
User must input start and end dates.
