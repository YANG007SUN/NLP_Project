# NLP Project (Behold) - Team Orange

## **Business Problem:** 
A common problem for Behold is the extremely manual classification of new products into their specific brands. Behold has supplier relationships from many global clothing vendors and marketplaces, but has found that the bottleneck for scaling out the number of products they can sell is identifying the brands associated with each new product.

## **Project Goal:**
1. Build an NLP classification model to predict which brand a new product should be assigned.
2. Create a brand recommender algorithm that would recommend an outfit given a customer’s search query.

## **Part 1 - Classification**
### Model Overview: <br>
- We used Neural Network LSTM to build an classification algorithm which classifies 31 different brands with **89.5%** testing.accuracy. (testing data contains 20% of entire dataset) <br><br>
![classification_results](images/classification_result.png) <br>

### Data Pre-Processing
- **Feature Engineer**
    - We created 5 features, and they are `location`, `color`,`material`,`category`,`random_description`
    - `location`: where did this product make.
    - `color`: color of this product.
    - `material`: what did this problem made of.
    - `category`: which category does this product belong. (top, bottom.etc)
    - `random_description`: details or description of this product. 
    - After creating features, we concatenated those 5 features into one column, separated by a blank space between each feature. `concat_features`.

- **Word Embedding**
    - We used word embedding technique to vectorize `concat_features` column. 
    - We used `GloVe Vectors` which was pre-trained by google. 
    - We only looked at top 200 words with the most frequency, meaning, ignoring words which show less frequently to aviod overfitting, and we also defined our max sentence length as 128. 

### Fit LSTM Model
- We had 61355 rows X 128 columns word vector ready for modeling, and we had 31 categories for y labels.
- Then, we separated training (80%) and testing (20%) dataset.
- We used LSTM model with two Dense layers, 10 epochs, 0.02 learning rate.
- Finally, we achieved **89.5%** accuracy on testing dataset.

## **Part 2 - Recommender Algorithm**
