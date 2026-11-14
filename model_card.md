# Model Card

## Model Details

Scikit-learn's RandomForestClassifier was used to predict whether a person earns more than $50,000 a year or $50,000 or less. The default settings were used, except for setting random_state to 42 so the results can be recreated. I used OneHotEncoder to convert the catgorical features into numerical values and LabelBinarizer to convert the salary labels into 0 and 1.

## Intended Use

This model was created to demonstrate a machine learning pipeline and make predictions through a FastAPI application. It is for learning purposes only and should not be used to make decisions about people.

## Training Data

The census.csv dataset used contains 32,561 rows and 15 columns. The model predicts salary and the other 14 columns are inputs. I used 80% of the data for training, which is 26,048 rows. The split uses random_state=42 and keeps salary proportions in the training and test sets. Of note, about 76% of the full dataset earns $50,000 or less. Any missing information marked with a question mark is kept as its own category.  

## Evaluation Data

The remaining 20% that was not used as training data, 6,513 rows, is used to test the model.I also checked performance for each value in every categorical feature. The results and number of rows in each group are saved in slice_output.txt.

## Metrics

The model was evaluated using precision, recall, and F1, with income above $50,000 as the positive class. The results were:

- Precision: 0.7353
- Recall: 0.6378
- F1: 0.6831

Of  the people the model predicted earning more than $50,000 about 73.5% actually did. In total, this model identified about 63.8% of everyone who actually earn more than $50,000. Performance differed between groups. For example, the group with an unknown workclass had 363 test rows and an F1 score of 0.4923, which was lower than the overall results.

## Ethical Considerations

This dataset includes information related to race and sex. Because of this, the model could potentially learn patterns that are related to existing inequalities in society. Due to this, the predictions created by this model should not be used in decisions affecting individuals. 

## Caveats and Recommendations

There is an imbalance in this dataset as most people in this data earn $50,000 or less. This imbalancce means the model may not be capturing the entire picture. Additionally, some groups had very few rows and because they are underrepresented in the data, their scores may be less reliable. The provided metrics function uses zero_division=1. This can produce a score of 1.0000 when a metric cannot be calculated. This should be noted as a perfecft score for a small group could be misleading.

If this model were to be used for applications beyond this project, I would recommend testing it on newer data. Also, I would recommend taking a closer look at the differences between groups and comparing other models and ways of handling missing information. 
