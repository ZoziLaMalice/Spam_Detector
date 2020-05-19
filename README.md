## Spam Detector

# Project Description

The objective of this project was to develop a Sklearn NLP model and deploy it with the Flask framework.

I used the **Multinomial Naives Bayes** model with **GridSearch** tuning, as well as the **Sklearn pipelines** for data processing.

The **F1 score** of this model is very good on the test set: *0.95*

Since the dataset was quite small, I chose to train the model online, that is to say each time the user wants to make a prediction, the model must train. It would have been better to train the model offline, then dump it in pickle format and then use it in online prediction.

Special thanks to Susan Li -> [GitHub](https://github.com/susanli2016/SMS-Message-Spam-Detector.) & the [article](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776).
