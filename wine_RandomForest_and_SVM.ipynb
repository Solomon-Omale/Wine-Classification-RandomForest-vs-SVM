{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random Forest or Support Vector Machine, which of these MLA performs better in the classification of wine?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.datasets import load_wine\n",
    "wine=load_wine()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['DESCR', 'data', 'feature_names', 'frame', 'target', 'target_names']"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dir(wine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.DataFrame(wine.data, columns=data.feature_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
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
       "      <th>alcohol</th>\n",
       "      <th>malic_acid</th>\n",
       "      <th>ash</th>\n",
       "      <th>alcalinity_of_ash</th>\n",
       "      <th>magnesium</th>\n",
       "      <th>total_phenols</th>\n",
       "      <th>flavanoids</th>\n",
       "      <th>nonflavanoid_phenols</th>\n",
       "      <th>proanthocyanins</th>\n",
       "      <th>color_intensity</th>\n",
       "      <th>hue</th>\n",
       "      <th>od280/od315_of_diluted_wines</th>\n",
       "      <th>proline</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14.23</td>\n",
       "      <td>1.71</td>\n",
       "      <td>2.43</td>\n",
       "      <td>15.6</td>\n",
       "      <td>127.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.06</td>\n",
       "      <td>0.28</td>\n",
       "      <td>2.29</td>\n",
       "      <td>5.64</td>\n",
       "      <td>1.04</td>\n",
       "      <td>3.92</td>\n",
       "      <td>1065.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.20</td>\n",
       "      <td>1.78</td>\n",
       "      <td>2.14</td>\n",
       "      <td>11.2</td>\n",
       "      <td>100.0</td>\n",
       "      <td>2.65</td>\n",
       "      <td>2.76</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.28</td>\n",
       "      <td>4.38</td>\n",
       "      <td>1.05</td>\n",
       "      <td>3.40</td>\n",
       "      <td>1050.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13.16</td>\n",
       "      <td>2.36</td>\n",
       "      <td>2.67</td>\n",
       "      <td>18.6</td>\n",
       "      <td>101.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.24</td>\n",
       "      <td>0.30</td>\n",
       "      <td>2.81</td>\n",
       "      <td>5.68</td>\n",
       "      <td>1.03</td>\n",
       "      <td>3.17</td>\n",
       "      <td>1185.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14.37</td>\n",
       "      <td>1.95</td>\n",
       "      <td>2.50</td>\n",
       "      <td>16.8</td>\n",
       "      <td>113.0</td>\n",
       "      <td>3.85</td>\n",
       "      <td>3.49</td>\n",
       "      <td>0.24</td>\n",
       "      <td>2.18</td>\n",
       "      <td>7.80</td>\n",
       "      <td>0.86</td>\n",
       "      <td>3.45</td>\n",
       "      <td>1480.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13.24</td>\n",
       "      <td>2.59</td>\n",
       "      <td>2.87</td>\n",
       "      <td>21.0</td>\n",
       "      <td>118.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0.39</td>\n",
       "      <td>1.82</td>\n",
       "      <td>4.32</td>\n",
       "      <td>1.04</td>\n",
       "      <td>2.93</td>\n",
       "      <td>735.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>173</th>\n",
       "      <td>13.71</td>\n",
       "      <td>5.65</td>\n",
       "      <td>2.45</td>\n",
       "      <td>20.5</td>\n",
       "      <td>95.0</td>\n",
       "      <td>1.68</td>\n",
       "      <td>0.61</td>\n",
       "      <td>0.52</td>\n",
       "      <td>1.06</td>\n",
       "      <td>7.70</td>\n",
       "      <td>0.64</td>\n",
       "      <td>1.74</td>\n",
       "      <td>740.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>174</th>\n",
       "      <td>13.40</td>\n",
       "      <td>3.91</td>\n",
       "      <td>2.48</td>\n",
       "      <td>23.0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>1.80</td>\n",
       "      <td>0.75</td>\n",
       "      <td>0.43</td>\n",
       "      <td>1.41</td>\n",
       "      <td>7.30</td>\n",
       "      <td>0.70</td>\n",
       "      <td>1.56</td>\n",
       "      <td>750.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>175</th>\n",
       "      <td>13.27</td>\n",
       "      <td>4.28</td>\n",
       "      <td>2.26</td>\n",
       "      <td>20.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>1.59</td>\n",
       "      <td>0.69</td>\n",
       "      <td>0.43</td>\n",
       "      <td>1.35</td>\n",
       "      <td>10.20</td>\n",
       "      <td>0.59</td>\n",
       "      <td>1.56</td>\n",
       "      <td>835.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>176</th>\n",
       "      <td>13.17</td>\n",
       "      <td>2.59</td>\n",
       "      <td>2.37</td>\n",
       "      <td>20.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>1.65</td>\n",
       "      <td>0.68</td>\n",
       "      <td>0.53</td>\n",
       "      <td>1.46</td>\n",
       "      <td>9.30</td>\n",
       "      <td>0.60</td>\n",
       "      <td>1.62</td>\n",
       "      <td>840.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>177</th>\n",
       "      <td>14.13</td>\n",
       "      <td>4.10</td>\n",
       "      <td>2.74</td>\n",
       "      <td>24.5</td>\n",
       "      <td>96.0</td>\n",
       "      <td>2.05</td>\n",
       "      <td>0.76</td>\n",
       "      <td>0.56</td>\n",
       "      <td>1.35</td>\n",
       "      <td>9.20</td>\n",
       "      <td>0.61</td>\n",
       "      <td>1.60</td>\n",
       "      <td>560.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>178 rows × 13 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \\\n",
       "0      14.23        1.71  2.43               15.6      127.0           2.80   \n",
       "1      13.20        1.78  2.14               11.2      100.0           2.65   \n",
       "2      13.16        2.36  2.67               18.6      101.0           2.80   \n",
       "3      14.37        1.95  2.50               16.8      113.0           3.85   \n",
       "4      13.24        2.59  2.87               21.0      118.0           2.80   \n",
       "..       ...         ...   ...                ...        ...            ...   \n",
       "173    13.71        5.65  2.45               20.5       95.0           1.68   \n",
       "174    13.40        3.91  2.48               23.0      102.0           1.80   \n",
       "175    13.27        4.28  2.26               20.0      120.0           1.59   \n",
       "176    13.17        2.59  2.37               20.0      120.0           1.65   \n",
       "177    14.13        4.10  2.74               24.5       96.0           2.05   \n",
       "\n",
       "     flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \\\n",
       "0          3.06                  0.28             2.29             5.64  1.04   \n",
       "1          2.76                  0.26             1.28             4.38  1.05   \n",
       "2          3.24                  0.30             2.81             5.68  1.03   \n",
       "3          3.49                  0.24             2.18             7.80  0.86   \n",
       "4          2.69                  0.39             1.82             4.32  1.04   \n",
       "..          ...                   ...              ...              ...   ...   \n",
       "173        0.61                  0.52             1.06             7.70  0.64   \n",
       "174        0.75                  0.43             1.41             7.30  0.70   \n",
       "175        0.69                  0.43             1.35            10.20  0.59   \n",
       "176        0.68                  0.53             1.46             9.30  0.60   \n",
       "177        0.76                  0.56             1.35             9.20  0.61   \n",
       "\n",
       "     od280/od315_of_diluted_wines  proline  \n",
       "0                            3.92   1065.0  \n",
       "1                            3.40   1050.0  \n",
       "2                            3.17   1185.0  \n",
       "3                            3.45   1480.0  \n",
       "4                            2.93    735.0  \n",
       "..                            ...      ...  \n",
       "173                          1.74    740.0  \n",
       "174                          1.56    750.0  \n",
       "175                          1.56    835.0  \n",
       "176                          1.62    840.0  \n",
       "177                          1.60    560.0  \n",
       "\n",
       "[178 rows x 13 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['class_0', 'class_1', 'class_2'], dtype='<U7')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['alcohol',\n",
       " 'malic_acid',\n",
       " 'ash',\n",
       " 'alcalinity_of_ash',\n",
       " 'magnesium',\n",
       " 'total_phenols',\n",
       " 'flavanoids',\n",
       " 'nonflavanoid_phenols',\n",
       " 'proanthocyanins',\n",
       " 'color_intensity',\n",
       " 'hue',\n",
       " 'od280/od315_of_diluted_wines',\n",
       " 'proline']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wine.feature_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
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
       "      <th>alcohol</th>\n",
       "      <th>malic_acid</th>\n",
       "      <th>ash</th>\n",
       "      <th>alcalinity_of_ash</th>\n",
       "      <th>magnesium</th>\n",
       "      <th>total_phenols</th>\n",
       "      <th>flavanoids</th>\n",
       "      <th>nonflavanoid_phenols</th>\n",
       "      <th>proanthocyanins</th>\n",
       "      <th>color_intensity</th>\n",
       "      <th>hue</th>\n",
       "      <th>od280/od315_of_diluted_wines</th>\n",
       "      <th>proline</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>14.23</td>\n",
       "      <td>1.71</td>\n",
       "      <td>2.43</td>\n",
       "      <td>15.6</td>\n",
       "      <td>127.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.06</td>\n",
       "      <td>0.28</td>\n",
       "      <td>2.29</td>\n",
       "      <td>5.64</td>\n",
       "      <td>1.04</td>\n",
       "      <td>3.92</td>\n",
       "      <td>1065.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>13.20</td>\n",
       "      <td>1.78</td>\n",
       "      <td>2.14</td>\n",
       "      <td>11.2</td>\n",
       "      <td>100.0</td>\n",
       "      <td>2.65</td>\n",
       "      <td>2.76</td>\n",
       "      <td>0.26</td>\n",
       "      <td>1.28</td>\n",
       "      <td>4.38</td>\n",
       "      <td>1.05</td>\n",
       "      <td>3.40</td>\n",
       "      <td>1050.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>13.16</td>\n",
       "      <td>2.36</td>\n",
       "      <td>2.67</td>\n",
       "      <td>18.6</td>\n",
       "      <td>101.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>3.24</td>\n",
       "      <td>0.30</td>\n",
       "      <td>2.81</td>\n",
       "      <td>5.68</td>\n",
       "      <td>1.03</td>\n",
       "      <td>3.17</td>\n",
       "      <td>1185.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14.37</td>\n",
       "      <td>1.95</td>\n",
       "      <td>2.50</td>\n",
       "      <td>16.8</td>\n",
       "      <td>113.0</td>\n",
       "      <td>3.85</td>\n",
       "      <td>3.49</td>\n",
       "      <td>0.24</td>\n",
       "      <td>2.18</td>\n",
       "      <td>7.80</td>\n",
       "      <td>0.86</td>\n",
       "      <td>3.45</td>\n",
       "      <td>1480.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>13.24</td>\n",
       "      <td>2.59</td>\n",
       "      <td>2.87</td>\n",
       "      <td>21.0</td>\n",
       "      <td>118.0</td>\n",
       "      <td>2.80</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0.39</td>\n",
       "      <td>1.82</td>\n",
       "      <td>4.32</td>\n",
       "      <td>1.04</td>\n",
       "      <td>2.93</td>\n",
       "      <td>735.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  \\\n",
       "0    14.23        1.71  2.43               15.6      127.0           2.80   \n",
       "1    13.20        1.78  2.14               11.2      100.0           2.65   \n",
       "2    13.16        2.36  2.67               18.6      101.0           2.80   \n",
       "3    14.37        1.95  2.50               16.8      113.0           3.85   \n",
       "4    13.24        2.59  2.87               21.0      118.0           2.80   \n",
       "\n",
       "   flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  \\\n",
       "0        3.06                  0.28             2.29             5.64  1.04   \n",
       "1        2.76                  0.26             1.28             4.38  1.05   \n",
       "2        3.24                  0.30             2.81             5.68  1.03   \n",
       "3        3.49                  0.24             2.18             7.80  0.86   \n",
       "4        2.69                  0.39             1.82             4.32  1.04   \n",
       "\n",
       "   od280/od315_of_diluted_wines  proline  target  \n",
       "0                          3.92   1065.0       0  \n",
       "1                          3.40   1050.0       0  \n",
       "2                          3.17   1185.0       0  \n",
       "3                          3.45   1480.0       0  \n",
       "4                          2.93    735.0       0  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['target']=wine.target\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
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
       "      <th>alcohol</th>\n",
       "      <th>malic_acid</th>\n",
       "      <th>ash</th>\n",
       "      <th>alcalinity_of_ash</th>\n",
       "      <th>magnesium</th>\n",
       "      <th>total_phenols</th>\n",
       "      <th>flavanoids</th>\n",
       "      <th>nonflavanoid_phenols</th>\n",
       "      <th>proanthocyanins</th>\n",
       "      <th>color_intensity</th>\n",
       "      <th>hue</th>\n",
       "      <th>od280/od315_of_diluted_wines</th>\n",
       "      <th>proline</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>alcohol</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.094397</td>\n",
       "      <td>0.211545</td>\n",
       "      <td>-0.310235</td>\n",
       "      <td>0.270798</td>\n",
       "      <td>0.289101</td>\n",
       "      <td>0.236815</td>\n",
       "      <td>-0.155929</td>\n",
       "      <td>0.136698</td>\n",
       "      <td>0.546364</td>\n",
       "      <td>-0.071747</td>\n",
       "      <td>0.072343</td>\n",
       "      <td>0.643720</td>\n",
       "      <td>-0.328222</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>malic_acid</th>\n",
       "      <td>0.094397</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.164045</td>\n",
       "      <td>0.288500</td>\n",
       "      <td>-0.054575</td>\n",
       "      <td>-0.335167</td>\n",
       "      <td>-0.411007</td>\n",
       "      <td>0.292977</td>\n",
       "      <td>-0.220746</td>\n",
       "      <td>0.248985</td>\n",
       "      <td>-0.561296</td>\n",
       "      <td>-0.368710</td>\n",
       "      <td>-0.192011</td>\n",
       "      <td>0.437776</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ash</th>\n",
       "      <td>0.211545</td>\n",
       "      <td>0.164045</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.443367</td>\n",
       "      <td>0.286587</td>\n",
       "      <td>0.128980</td>\n",
       "      <td>0.115077</td>\n",
       "      <td>0.186230</td>\n",
       "      <td>0.009652</td>\n",
       "      <td>0.258887</td>\n",
       "      <td>-0.074667</td>\n",
       "      <td>0.003911</td>\n",
       "      <td>0.223626</td>\n",
       "      <td>-0.049643</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>alcalinity_of_ash</th>\n",
       "      <td>-0.310235</td>\n",
       "      <td>0.288500</td>\n",
       "      <td>0.443367</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.083333</td>\n",
       "      <td>-0.321113</td>\n",
       "      <td>-0.351370</td>\n",
       "      <td>0.361922</td>\n",
       "      <td>-0.197327</td>\n",
       "      <td>0.018732</td>\n",
       "      <td>-0.273955</td>\n",
       "      <td>-0.276769</td>\n",
       "      <td>-0.440597</td>\n",
       "      <td>0.517859</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>magnesium</th>\n",
       "      <td>0.270798</td>\n",
       "      <td>-0.054575</td>\n",
       "      <td>0.286587</td>\n",
       "      <td>-0.083333</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.214401</td>\n",
       "      <td>0.195784</td>\n",
       "      <td>-0.256294</td>\n",
       "      <td>0.236441</td>\n",
       "      <td>0.199950</td>\n",
       "      <td>0.055398</td>\n",
       "      <td>0.066004</td>\n",
       "      <td>0.393351</td>\n",
       "      <td>-0.209179</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_phenols</th>\n",
       "      <td>0.289101</td>\n",
       "      <td>-0.335167</td>\n",
       "      <td>0.128980</td>\n",
       "      <td>-0.321113</td>\n",
       "      <td>0.214401</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.864564</td>\n",
       "      <td>-0.449935</td>\n",
       "      <td>0.612413</td>\n",
       "      <td>-0.055136</td>\n",
       "      <td>0.433681</td>\n",
       "      <td>0.699949</td>\n",
       "      <td>0.498115</td>\n",
       "      <td>-0.719163</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>flavanoids</th>\n",
       "      <td>0.236815</td>\n",
       "      <td>-0.411007</td>\n",
       "      <td>0.115077</td>\n",
       "      <td>-0.351370</td>\n",
       "      <td>0.195784</td>\n",
       "      <td>0.864564</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.537900</td>\n",
       "      <td>0.652692</td>\n",
       "      <td>-0.172379</td>\n",
       "      <td>0.543479</td>\n",
       "      <td>0.787194</td>\n",
       "      <td>0.494193</td>\n",
       "      <td>-0.847498</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>nonflavanoid_phenols</th>\n",
       "      <td>-0.155929</td>\n",
       "      <td>0.292977</td>\n",
       "      <td>0.186230</td>\n",
       "      <td>0.361922</td>\n",
       "      <td>-0.256294</td>\n",
       "      <td>-0.449935</td>\n",
       "      <td>-0.537900</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.365845</td>\n",
       "      <td>0.139057</td>\n",
       "      <td>-0.262640</td>\n",
       "      <td>-0.503270</td>\n",
       "      <td>-0.311385</td>\n",
       "      <td>0.489109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>proanthocyanins</th>\n",
       "      <td>0.136698</td>\n",
       "      <td>-0.220746</td>\n",
       "      <td>0.009652</td>\n",
       "      <td>-0.197327</td>\n",
       "      <td>0.236441</td>\n",
       "      <td>0.612413</td>\n",
       "      <td>0.652692</td>\n",
       "      <td>-0.365845</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.025250</td>\n",
       "      <td>0.295544</td>\n",
       "      <td>0.519067</td>\n",
       "      <td>0.330417</td>\n",
       "      <td>-0.499130</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>color_intensity</th>\n",
       "      <td>0.546364</td>\n",
       "      <td>0.248985</td>\n",
       "      <td>0.258887</td>\n",
       "      <td>0.018732</td>\n",
       "      <td>0.199950</td>\n",
       "      <td>-0.055136</td>\n",
       "      <td>-0.172379</td>\n",
       "      <td>0.139057</td>\n",
       "      <td>-0.025250</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.521813</td>\n",
       "      <td>-0.428815</td>\n",
       "      <td>0.316100</td>\n",
       "      <td>0.265668</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>hue</th>\n",
       "      <td>-0.071747</td>\n",
       "      <td>-0.561296</td>\n",
       "      <td>-0.074667</td>\n",
       "      <td>-0.273955</td>\n",
       "      <td>0.055398</td>\n",
       "      <td>0.433681</td>\n",
       "      <td>0.543479</td>\n",
       "      <td>-0.262640</td>\n",
       "      <td>0.295544</td>\n",
       "      <td>-0.521813</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.565468</td>\n",
       "      <td>0.236183</td>\n",
       "      <td>-0.617369</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>od280/od315_of_diluted_wines</th>\n",
       "      <td>0.072343</td>\n",
       "      <td>-0.368710</td>\n",
       "      <td>0.003911</td>\n",
       "      <td>-0.276769</td>\n",
       "      <td>0.066004</td>\n",
       "      <td>0.699949</td>\n",
       "      <td>0.787194</td>\n",
       "      <td>-0.503270</td>\n",
       "      <td>0.519067</td>\n",
       "      <td>-0.428815</td>\n",
       "      <td>0.565468</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.312761</td>\n",
       "      <td>-0.788230</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>proline</th>\n",
       "      <td>0.643720</td>\n",
       "      <td>-0.192011</td>\n",
       "      <td>0.223626</td>\n",
       "      <td>-0.440597</td>\n",
       "      <td>0.393351</td>\n",
       "      <td>0.498115</td>\n",
       "      <td>0.494193</td>\n",
       "      <td>-0.311385</td>\n",
       "      <td>0.330417</td>\n",
       "      <td>0.316100</td>\n",
       "      <td>0.236183</td>\n",
       "      <td>0.312761</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.633717</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>target</th>\n",
       "      <td>-0.328222</td>\n",
       "      <td>0.437776</td>\n",
       "      <td>-0.049643</td>\n",
       "      <td>0.517859</td>\n",
       "      <td>-0.209179</td>\n",
       "      <td>-0.719163</td>\n",
       "      <td>-0.847498</td>\n",
       "      <td>0.489109</td>\n",
       "      <td>-0.499130</td>\n",
       "      <td>0.265668</td>\n",
       "      <td>-0.617369</td>\n",
       "      <td>-0.788230</td>\n",
       "      <td>-0.633717</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                               alcohol  malic_acid       ash  \\\n",
       "alcohol                       1.000000    0.094397  0.211545   \n",
       "malic_acid                    0.094397    1.000000  0.164045   \n",
       "ash                           0.211545    0.164045  1.000000   \n",
       "alcalinity_of_ash            -0.310235    0.288500  0.443367   \n",
       "magnesium                     0.270798   -0.054575  0.286587   \n",
       "total_phenols                 0.289101   -0.335167  0.128980   \n",
       "flavanoids                    0.236815   -0.411007  0.115077   \n",
       "nonflavanoid_phenols         -0.155929    0.292977  0.186230   \n",
       "proanthocyanins               0.136698   -0.220746  0.009652   \n",
       "color_intensity               0.546364    0.248985  0.258887   \n",
       "hue                          -0.071747   -0.561296 -0.074667   \n",
       "od280/od315_of_diluted_wines  0.072343   -0.368710  0.003911   \n",
       "proline                       0.643720   -0.192011  0.223626   \n",
       "target                       -0.328222    0.437776 -0.049643   \n",
       "\n",
       "                              alcalinity_of_ash  magnesium  total_phenols  \\\n",
       "alcohol                               -0.310235   0.270798       0.289101   \n",
       "malic_acid                             0.288500  -0.054575      -0.335167   \n",
       "ash                                    0.443367   0.286587       0.128980   \n",
       "alcalinity_of_ash                      1.000000  -0.083333      -0.321113   \n",
       "magnesium                             -0.083333   1.000000       0.214401   \n",
       "total_phenols                         -0.321113   0.214401       1.000000   \n",
       "flavanoids                            -0.351370   0.195784       0.864564   \n",
       "nonflavanoid_phenols                   0.361922  -0.256294      -0.449935   \n",
       "proanthocyanins                       -0.197327   0.236441       0.612413   \n",
       "color_intensity                        0.018732   0.199950      -0.055136   \n",
       "hue                                   -0.273955   0.055398       0.433681   \n",
       "od280/od315_of_diluted_wines          -0.276769   0.066004       0.699949   \n",
       "proline                               -0.440597   0.393351       0.498115   \n",
       "target                                 0.517859  -0.209179      -0.719163   \n",
       "\n",
       "                              flavanoids  nonflavanoid_phenols  \\\n",
       "alcohol                         0.236815             -0.155929   \n",
       "malic_acid                     -0.411007              0.292977   \n",
       "ash                             0.115077              0.186230   \n",
       "alcalinity_of_ash              -0.351370              0.361922   \n",
       "magnesium                       0.195784             -0.256294   \n",
       "total_phenols                   0.864564             -0.449935   \n",
       "flavanoids                      1.000000             -0.537900   \n",
       "nonflavanoid_phenols           -0.537900              1.000000   \n",
       "proanthocyanins                 0.652692             -0.365845   \n",
       "color_intensity                -0.172379              0.139057   \n",
       "hue                             0.543479             -0.262640   \n",
       "od280/od315_of_diluted_wines    0.787194             -0.503270   \n",
       "proline                         0.494193             -0.311385   \n",
       "target                         -0.847498              0.489109   \n",
       "\n",
       "                              proanthocyanins  color_intensity       hue  \\\n",
       "alcohol                              0.136698         0.546364 -0.071747   \n",
       "malic_acid                          -0.220746         0.248985 -0.561296   \n",
       "ash                                  0.009652         0.258887 -0.074667   \n",
       "alcalinity_of_ash                   -0.197327         0.018732 -0.273955   \n",
       "magnesium                            0.236441         0.199950  0.055398   \n",
       "total_phenols                        0.612413        -0.055136  0.433681   \n",
       "flavanoids                           0.652692        -0.172379  0.543479   \n",
       "nonflavanoid_phenols                -0.365845         0.139057 -0.262640   \n",
       "proanthocyanins                      1.000000        -0.025250  0.295544   \n",
       "color_intensity                     -0.025250         1.000000 -0.521813   \n",
       "hue                                  0.295544        -0.521813  1.000000   \n",
       "od280/od315_of_diluted_wines         0.519067        -0.428815  0.565468   \n",
       "proline                              0.330417         0.316100  0.236183   \n",
       "target                              -0.499130         0.265668 -0.617369   \n",
       "\n",
       "                              od280/od315_of_diluted_wines   proline    target  \n",
       "alcohol                                           0.072343  0.643720 -0.328222  \n",
       "malic_acid                                       -0.368710 -0.192011  0.437776  \n",
       "ash                                               0.003911  0.223626 -0.049643  \n",
       "alcalinity_of_ash                                -0.276769 -0.440597  0.517859  \n",
       "magnesium                                         0.066004  0.393351 -0.209179  \n",
       "total_phenols                                     0.699949  0.498115 -0.719163  \n",
       "flavanoids                                        0.787194  0.494193 -0.847498  \n",
       "nonflavanoid_phenols                             -0.503270 -0.311385  0.489109  \n",
       "proanthocyanins                                   0.519067  0.330417 -0.499130  \n",
       "color_intensity                                  -0.428815  0.316100  0.265668  \n",
       "hue                                               0.565468  0.236183 -0.617369  \n",
       "od280/od315_of_diluted_wines                      1.000000  0.312761 -0.788230  \n",
       "proline                                           0.312761  1.000000 -0.633717  \n",
       "target                                           -0.788230 -0.633717  1.000000  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df.drop(['target'], axis='columns')\n",
    "y=df.target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "36"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=RandomForestClassifier()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier()"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9722222222222222"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "confusion matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_predicted=model.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test,y_predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x91353a0>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x576 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "plt.figure(figsize=(8,8))\n",
    "sns.heatmap(cm,annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC\n",
    "model2 = SVC(C=10, gamma=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SVC(C=10, gamma=1)"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4166666666666667"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model2.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Conclusion: RandomForest performed with better accuracy than SVC."
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
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
