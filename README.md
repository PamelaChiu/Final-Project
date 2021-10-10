# Final-Project: Housing Price Prediction and Analysis in major Canadian Cities

This is the capstone project! Time to make all the skills you worked so hard for shine.

# Requirements

- A new project on data you gathered yourself. Existing datasets are OK but have a higher bar in terms of difficulty

- The project is deployed in some form -- it's not a static notebook, but has a server component.

- The project can be analysis/visualization or a model. If it's a model you should build an inference server for it taking JSON. If it's an analysis you can build a presentation page using something like https://dash.plotly.com/

- Submit the project on your own repo

[Google Sheet is here](https://docs.google.com/spreadsheets/d/1EaQZuBYEhv606F24Zz324jyfrqWjCKIz4hfLzVRh6rE/edit?usp=sharing)

# Summary of the project

The final project is to predict the future prices of homes in Montreal, Toronto, Vancouver and Quebec City.

During the search for a home in 2020 (like everyone else), I noticed that prices were growing at an unsustainable pace and were becoming unaffordable for the average Canadian, whose wages did not see the same increase. I wanted to get an overall view of the real estate market in large Canadian cities, and get an idea of the trajectory for the next 12 months.  

In this project, I will be using 2 different time-series models: the AutoRegresssive Integrated Moving Average(ARIMA) Method and the Holt-Winters Method.

# Conclusion

ARIMA is a time series model that uses past values to predict future trends.
Holt-Winters (another time series model) predicts trends based on putting more weight in recent data and less weight on the early data. It also looks at 3 different aspects: trends, value and seasonality. This is best known as triple exponential smoothing.

Based on my testing data of 12 months, the dataset for Montreal worked better on Holt-Winters Method because the root mean square error (RMSE) was lower compared to ARIMA.
While the datasets for Toronto, Quebec City and Vancouver had a lower RMSE for ARIMA than Holt Winters.

Below is the result of the forecast for Montreal using Holt-Winters Model.
![Graph of Housing Price](./Montreal.GIF)


[Click Here](./assets) to view the forecasts on Quebec City, Toronto, & Vancouver.

