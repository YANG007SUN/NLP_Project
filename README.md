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
    - We only looked at top 200 words with the most frequency, meaning, ignoring words which show less frequently to avoid overfitting, and we also defined our max sentence length as 128. 

### Fit LSTM Model
- We had 61355 rows X 128 columns word vector ready for modeling, and we had 31 categories for y labels.
- Then, we separated training (80%) and testing (20%) dataset.
- We used LSTM model with two Dense layers, 10 epochs, 0.02 learning rate.
- Finally, we achieved **89.5%** accuracy on testing dataset.

## **Part 2 - Recommender Algorithm**

### Overview
The most important part of our very beginning attempts was to create an algorithm rule to predict the category of those products without category, and then use around 60,000 product data for the recommendation system. The accuracy rate of our prediction was about 88%. We made up some input queries and checked out the output (recommended outfit combination) may not work as well as we predicted since some products were assigned to the wrong category.

Therefore, we decided to only use product data with category labels and outfit id in our final version, which was the data in the `outfit_combination` dataset. 

The algorithm would process the input queries, find out the product with the highest similarity and its outfit id, output the recommended outfit combination using the same outfit ID.

### Data Pre-Processing for the Original Datasets
- We merged the `outfit_combination` dataset with the product dataset using `left join`. 
- We combined the features `product_full_name`, `details`, and `description` together as our new text feature. 
- We used regex to remove punctuations on the combined feature and used the `nltk` package to remove stopwords. 
- We performed lemmatization on the combined text feature.

### Data Pre-Processing for the Input Query
- We used regex to remove punctuation and digits of the input query and used the `nltk` package to remove stopwords. 
- We performed lemmatization on the input query.

### Label Product Category of the Input Query
We applied domain knowledge and created a rule-based algorithm to determine the category of the product that the user was trying to search for. 

To be specific, we collected over 20 relevant keywords for each category using our business domain knowledge. The rule-based algorithm created a dictionary for each record and counted the frequency of the keywords in each category, and set the most frequent category as the category of the product. If there was no category matched, the algorithm would set the category as `Unknown`. If two or more categories had the same frequency, the algorithm would randomly pick one as the category of the product. 

### Our Brand Recommender Algorithm
We used the `TF-IDF` method to vectorize the product table, and the `Max_feature` is set to 1000 to reduce complexity. First, we summed the `TF-IDF` scores for each document. Second, we tokenized each product and calculated the using total embedding for each token. Third, we divided the running total by the sum of the `TF-IDF` score for the document to generate the weighted `TF-IDF` embedding for each sentence. 

After that, we calculated the cosine similarity between the query and each product. As mention in the last section, we determined the product category of the input query beforehand. Our recommender algorithm would then return the product with the highest cosine similarity in that category based on the input query. After that, the algorithm would extract the outfit ID of the most similar product and find out all the products sharing the same outfit ID. Finally, the algorithm would return all the products in that outfit set.

### Sample Input Query and Output

A sample input query was `yellow onepiece for beach with pink flower` and the output (recommended outfit combination) was:
- onepiece: **Ida Dress** (01DPD4R5X5TQCWTVTC2AEAFC10)
- shoe: **Virginia Boot** (01DPKNCMSFAWF2HVQSRHHXDV0K)
- accessory: **Cassi Belt Bag** (01DPEHS0XH9PDD1GH5ZE4P43A2)
 
### Our Previous Attempts
1. We have tried to use `spaCy`'s internal similarity function to calculate the similarity between the user query and each of our new text feature, which was the combination of `product_full_name`, `details`, and `description`. Then, we found the product with the largest similarity. If the product already has an expert-defined outfit, then the function would return all products with the same outfit ID; otherwise, it would return the single product with the highest similarity. We chose not to use this function because that `spaCy` had a computationally expensive algorithm to find the similarity, and each query might take up to 20 minutes to finish the execution. Therefore, we decided to try a more efficient algorithm.

2. We have tried to label the category for each item based on given text features such as `name`, `brand_category`, `description`, and `details`. There are four reasons why we chose to use word frequency and regex matching instead of building a classification model when making labeling the category:
    - The `product_category`column contained given information of each product’s category while `product_category` of most records was unknown.
    - For the products with known categories, there were only about 800 unique products. The number of records with labels was too small compared to the whole dataset. 
    - Among these products with category labels, we found out some cases that the name and category of one product were obviously contradicted. E.g., the name of one product with ID as `01DT0C8NM9KG2EF0A286VZRETE` is `Tank top in Re-Imagined Silk` while its category is `accessory1`.
    - There are only 5 product categories in the given labels while we thought there are many products which do not belong to any of these categories. E.g. the name of one product with ID as `01EEBHWPA3BEBQGBMXGN8KZTTG` is `PETAL Candle`. The description of this product also shows that this product is a kind of candle, not an outfit.

    However, when we tried to label the products without category based on word frequency, it turned out that the match results did not have a satisfying accuracy which may further affect the selected product by the search function. We believed that an accurate result was more important for our function so that we decided to use only the products with product categories and products with `outfit_id` in the outfit dataset to match the query. Although fewer data entries are used, we were able to obtain a more accurate result.

3. We have tried `CountVectorizer` to vectorize the products but we finally chose `TF-IDF` method since the `TF-IDF` method balanced out the term frequency (how often the word appears in the document) with its inverse document frequency (how often the term appears across all documents in the data set).
