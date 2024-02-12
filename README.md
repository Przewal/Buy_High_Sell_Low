# Buy High, Sell Low!

**Authors** - `Prajwal Patnaik, Tejesh Pattnaik, Diana Zhang`

**Portfolio Analysis - Page 1 Code** - `P1.py`

**Individual Stock Analysis - Page 2 Code** - `P2.py`

**Chatbot - Page 3 Code** - `P3.py`

### Libraries and Dependencies Used

- Pandas [https://pandas.pydata.org/docs/getting_started/install.html]
- Numpy [https://numpy.org/install/]
- Matplotlib [https://matplotlib.org/stable/users/installing/index.html]
- Yfinance [https://pypi.org/project/yfinance/]
- Hvplot [https://hvplot.holoviz.org/getting_started/installation.html]
- Datetime (the standard Python library is adequate)
- Holoviews [https://holoviews.org/install.html]
- Scikit-learn [https://scikit-learn.org/stable/install.html]
- Imbalanced-learn [https://pypi.org/project/imbalanced-learn/]
- Finta [https://pypi.org/project/finta/]
- Pandas-ta [https://github.com/twopirllc/pandas-ta]
- Streamlit: [https://pypi.org/project/streamlit/]
- Bokeh: [https://pypi.org/project/bokeh/]
- Plotly: [https://pypi.org/project/plotly/]
- Yahooquery: [https://pypi.org/project/yahooquery/]
- OpenAI: [https://pypi.org/project/openai/]

***Note*** - Bokeh plot needs to be downgraded to run with the streamlit interface.

***Note-2*** - Use your own secret key from openAI in P3, line 7. 


### The Idea
Combining Portfolio Analysis, Trading Algorithms with Machine Learning, and Roboadvisor functionalities.

### The Goal
Empowers users with actionable insights for informed investing decisions. Facilitates as a one-step solution for portfolio management, algorithmic trading, and personalized investment recommendations.

## Features

### Interactive Portfolio Builder and Analysis
- Select dates and stocks for historical price analysis.
- Review the finance data and quantitative analysis of the portfolio (benchmarked against the S&P500).

### Individual Stock Analysis with Trading Algorithms and Machine Learning
- Select dates, trading algorithms, trading windows intervals, stop-loss percentages, initial capital & share size for back-testing, machine learning models.
- Review metrics of algorithm and ML classification reports.

### Chatbot using OpenAI
- Use the previous two pages to frame questions to use advice or gain an understanding of the analyses from the OpenAI chatbot.