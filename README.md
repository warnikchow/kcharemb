# KCharEmb
Tutorial for character-level embeddings in Korean sentence classification

## Requirements
fasttext==0.8.3 (else gensim==3.6.0), hgtk==0.1.3, Keras==2.1.2,<br/> 
numpy==1.14.3, scikit-learn==0.19.1, tensorflow-gpu==1.4.1<br/>
**Currently available for python 3.5 and upper version is in implementation**

## Datasets
### NMSC
[Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)
Train:Test ratio is 3:1.<br/> 
Train set is again split into train:validation set in ratio 9:1.
### 3i4K
[Intonation-aided Intention Identification for Korean](https://github.com/warnikchow/3i4k)
Train:Test ratio is 9:1.<br/> 
Train set is again split into train:validation set in ratio 9:1.

## Character-level representations
<image src="https://github.com/warnikchow/kcharemb/blob/master/images/fig1.png" width="600"><br/>

### Word vector for *Cho2018a-Dense*
[Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE .BIN FILE in the NEW FOLDER named 'vectors'
* This can be replaced with whatever model the user employs, but it requires an additional training.

## Result
<image src="https://github.com/warnikchow/kcharemb/blob/master/images/fig2.PNG" width="600"><br/>

### The analysis can be found in [the paper](https://arxiv.org/abs/1905.13656)!

### DISCLAIMER
We added NSMC files to our repo since it is easier for cloning and replication, and most of all the data is open to the public domain. The files will be removed if any problem comes up.

### ACKNOWLEDGEMENT
The authors appreciate Yong Gyu Park for informing us the points that require improvement.

## Citation
### For the utilization of the dataset 3i4K, cite the following:
```
@article{cho2018speech,
	title={Speech Intention Understanding in a Head-final Language: 
	A Disambiguation Utilizing Intonation-dependency},
	author={Cho, Won Ik and Lee, Hyeon Seung and Yoon, Ji Won and Kim, Seok Min and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1811.04231},
	year={2018}
}
```
### For the utilization of the word vector dictionary, cite the following:
```
@article{cho2018real,
	title={Real-time Automatic Word Segmentation for User-generated Text},
	author={Cho, Won Ik and Cheon, Sung Jun and Kang, Woo Hyun and Kim, Ji Won and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1810.13113},
	year={2018}
}
```
### For the utilization of the result and the code, cite the following:
```
@article{cho2019investigating,
	title={Investigating an Effective Character-level Embedding in Korean Sentence Classification},
	author={Cho, Won Ik and Kim, Seok Min and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1905.13656},
	year={2019}
}
```
