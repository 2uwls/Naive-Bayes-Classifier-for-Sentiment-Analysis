# Project: Naive Bayes Classifier for Sentiment 
## Requirements

- Python 3.9.7
- matplotlib
- csv

## file structure
```
.  
├── README.md   
├── data          
│   ├── train.csv    
│   ├── test.csv    
│   └── stopwords.txt           
└── main.py  
```

## Execution
```bash
pip install matplotlib
```
```bash
python main.py
```

## Data
- `train.csv` and `test.csv` files are used to train and test the classifier. The first column of the CSV file should contain the `label` and the second column should contain the `review text`.
- `stopwords.txt` file contains the list of stopwords that are used to preprocess the text data.
