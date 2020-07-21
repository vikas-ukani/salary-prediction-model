# Salary Prediction Model
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#  Firstly Import the mendatory packages like pandas numpy seaborn sklearn\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Feature models\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>8476</td>\n",
       "      <td>You Can Smell Hillary’s Fear</td>\n",
       "      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10294</td>\n",
       "      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>\n",
       "      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3608</td>\n",
       "      <td>Kerry to go to Paris in gesture of sympathy</td>\n",
       "      <td>U.S. Secretary of State John F. Kerry said Mon...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10142</td>\n",
       "      <td>Bernie supporters on Twitter erupt in anger ag...</td>\n",
       "      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>\n",
       "      <td>FAKE</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>875</td>\n",
       "      <td>The Battle of New York: Why This Primary Matters</td>\n",
       "      <td>It's primary day in New York and front-runners...</td>\n",
       "      <td>REAL</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Unnamed: 0                                              title  \\\n",
       "0        8476                       You Can Smell Hillary’s Fear   \n",
       "1       10294  Watch The Exact Moment Paul Ryan Committed Pol...   \n",
       "2        3608        Kerry to go to Paris in gesture of sympathy   \n",
       "3       10142  Bernie supporters on Twitter erupt in anger ag...   \n",
       "4         875   The Battle of New York: Why This Primary Matters   \n",
       "\n",
       "                                                text label  \n",
       "0  Daniel Greenfield, a Shillman Journalism Fello...  FAKE  \n",
       "1  Google Pinterest Digg Linkedin Reddit Stumbleu...  FAKE  \n",
       "2  U.S. Secretary of State John F. Kerry said Mon...  REAL  \n",
       "3  — Kaydee King (@KaydeeKing) November 9, 2016 T...  FAKE  \n",
       "4  It's primary day in New York and front-runners...  REAL  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#  Generate new data frame from news.csv\n",
    "\n",
    "df = pd.read_csv('news.csv')\n",
    "\n",
    "# Show first five News (rows)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6335, 4)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Know dataFrame shape\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    FAKE\n",
       "1    FAKE\n",
       "2    REAL\n",
       "3    FAKE\n",
       "4    REAL\n",
       "Name: label, dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fetch all labels ( FAKE | REAL )\n",
    "labels = df['label']\n",
    "\n",
    "# Get first five labels\n",
    "labels.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Split the Trainng and Testing data set\n",
    "x_train, x_test, y_train, y_test = train_test_split( df.text , labels, test_size=0.30, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    " # Apply TfidfVectorizer \n",
    "    \n",
    "tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)\n",
    "\n",
    "# fitting and transformming  test set\n",
    "\n",
    "tfidf_train = tfidf_vectorizer.fit_transform(x_train) \n",
    "tfidf_test = tfidf_vectorizer.transform(x_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,\n",
       "                            early_stopping=False, fit_intercept=True,\n",
       "                            loss='hinge', max_iter=50, n_iter_no_change=5,\n",
       "                            n_jobs=None, random_state=None, shuffle=True,\n",
       "                            tol=0.001, validation_fraction=0.1, verbose=0,\n",
       "                            warm_start=False)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# setup  a PassiveAggressiveClassifier\n",
    "\n",
    "pac = PassiveAggressiveClassifier( max_iter = 50)\n",
    "pac.fit(tfidf_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# check the model accuracy value "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 93.53\n"
     ]
    }
   ],
   "source": [
    "# Check Predict on the test set and calculate accuracy value\n",
    "\n",
    "y_pred = pac.predict(tfidf_test)\n",
    "score = accuracy_score(y_test,y_pred)\n",
    "print( 'Accuracy:', round(score*100,2) )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[913,  55],\n",
       "       [ 68, 865]], dtype=int64)"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Set Build the confusion matrics\n",
    "\n",
    "confusion_matrix(y_test, y_pred, labels=labels.unique())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* we got 913 true positives, 865 true negatives, 68 false positives, and 55 false negatives news. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
