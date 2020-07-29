# Option Pricer

This is an open source module I rewrote and translated from an open source Jupyter notebook to calculate the fair price of stock options so you can know how much you are overpaying or getting a deal.

I took the formulas, calculus and math and transpiled it into this library. You can find the math, the explanations, etc. from [this excellent site](http://www.understandtrading.com/gbs.html) that explains it in detail, and provides the bulk of the code.

### Input Parameters

##### Inputs used by all models
##### Parameter 	Description
##### option_type 	Put/Call Indicator Single character, "c" indicates a call; "p" a put
- fs -	Price of Underlying FS is generically used, but for specific models, the following abbreviations may be used: F = Forward Price, S = Spot Price)
- x -	Strike Price
- t -	Time to Maturity This is in years (1.0 = 1 year, 0.5 = six months, etc)
- r -	Risk Free Interest Rate Interest rates (0.10 = 10% interest rate
- v -	Implied Volatility Annualized implied volatility (1=100% annual volatility, 0.34 = 34% annual volatility
##### Inputs used by some models
##### Parameter 	Description
- b 	Cost of Carry This is only found in internal implementations, but is identical to the cost of carry (b) term commonly found in academic option pricing literature
- q - Continuous Dividend Used in Merton and American models; Internally, this is converted into cost of carry, b, with formula b = r-q
- rf -	Foreign Interest Rate Only used GK model; this functions similarly to q
- t_a -	Asian Start Used for Asian options; This is the time that starts the averaging period (TA=0 means that averaging starts immediately). As TA approaches T, the Asian value should become very close to the Black76 Value
- cp -	Option Price Used in the implied vol calculations; This is the price of the call or put observed in the market