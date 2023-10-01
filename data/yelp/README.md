# How is the closed category for Yelp generated


The closed_category.json is created based on frequency counting of tags in Yelp and a hierarchical structure is of categories built based on the frequency counting. Basically the idea is: the most frequently occurred tags will be highest level of category; categories below them are selected based on co-occurrences counting.

To create the file, run the following python file:
```
python yelp_SID_generation.py
```
