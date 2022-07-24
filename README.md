# Syntactically-Informed Unsupervised Paraphrasing with Non-Parallel Data
The source code of the SUP model. Come soon...

## Datasets
Quora: download from [https://drive.google.com/file/d/1RdIQEoWJbm4HtNYaxFHjleBgX5FIZZtp/view?usp=sharing](https://drive.google.com/file/d/1RdIQEoWJbm4HtNYaxFHjleBgX5FIZZtp/view?usp=sharing).

ParaNMT: You can download from this paper [ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations](https://aclanthology.org/P18-1042/).

## Requirements

```shell
python >= 3.6
torch == 1.6.0
nltk == 3.4.5
zss == 1.2.0
```
## Data Processing
We use the [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml#Download) to obtain the parse tree and template. 

The command is as follows:
```shell
input file：
java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -file file.txt -outputFormat text -outputDirectory /outputdir/
input filelist:
java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -filelist filenames.txt -outputFormat text -outputDirectory /outputdir/
```
If the data is large, you can use the ***split*** command to divide the file into multiple small files for parsing. Then you can use the ***pos_to_file.py*** and ***template.py*** in the ***autocg*** directory to extract parse tree and template.
