The four features mentioned in the paper, 'Classifying Crises-Information Relevancy with Semantics', are Statistical Features (SF), BabelNet Features (SF+SemBN), DBpedia Features (SF+SemDB), and BabelNet+DBpedia Features (SF+SemBNDB).

The four folders, mentioned hereby, contain the training and test data for each case tested in the paper:
1. statistical_features: all the train and test files for SF
2. stat_n_hypernym_en: all the train and test files for SF+SemBN
3. stat_n_dbpedia_en: all the train and test files for SF+SemDB
4. stat_n_hypernym_dbpedia_en: all the train and test files for SF+SemBNDB

Folder 'babelfy_labelling' contains Babelfy labels for each tweet across each event.
Folder 'expanded_db_semantics_en' contains DBpedia semantics for each Synset extracted via Babelfy labeling.

Folder 'hypernym_en' contains BabelNet semantics (hypernyms) for each Synset and also annotationlemma_hypernym_semantics.csv file that contains entire SemBN for each tweet.
