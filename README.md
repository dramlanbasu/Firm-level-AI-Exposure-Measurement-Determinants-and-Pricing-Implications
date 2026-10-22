# Firm-level-AI-Exposure-Measurement-Determinants-and-Pricing-Implications
# Authors: Sakshi Basu, Chandra Thapa, Biwesh Neupane and Amlan Basu

The paper is available in the following link,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7575518

Abstract: We construct forward-looking, time-varying measures of firm-level AI exposure using a hybrid natural language processing approach applied to earnings call transcripts of 13,757 firms across 44 countries from 2010 to 2025. Beyond an overall measure, we build ten topic-specific measures spanning opportunity, productivity, innovation, business growth, business risk, regulatory risk, security risk, environmental risk, and positive and negative assessments. AI exposure rises sharply after ChatGPT's launch in November 2022 but is driven predominantly by firm-specific rather than industry-level factors. Managers frame AI overwhelmingly in an opportunity, productivity, and innovation context rather than a risk context, and assess it overwhelmingly favorably. Larger, cash-rich, higher-valued, younger, and less-levered firms show greater exposure. Changes in a firm's AI exposure are associated with subsequent stock returns, with a one-standard deviation increase associated with approximately 1.37% higher annual returns, an effect concentrated after ChatGPT's launch. A U.S. long-short portfolio of high-exposure firms earns no abnormal return before the launch, once its large-cap growth tilts are accounted for, but earns a large abnormal return afterward.

Keywords: Artificial Intelligence, AI Exposure, Earnings Conference Calls, YAKE, Word2Vec, ChatGPT

If you are using any of the codes please cite the paper
**Basu, Sakshi and Thapa, Chandra and Neupane, Biwesh and Basu, Amlan, Firm-level AI Exposure: Measurement, Determinants, and Pricing Implications (October 07, 2026). Available at SSRN: https://ssrn.com/abstract=7575518 or http://dx.doi.org/10.2139/ssrn.7575518**

There are 2 codes,
1. Yake_Similar_Keywords.py to extract nearest keywords.
2. Keywords_label_1.py to count the number of each keywords, extract only the sentences that have the specified keywords, label the sentences in specific categories and count the number of sentences under each label.

Before running the code make sure following modules are installed **(Use python 3)**,
1. pip install yake
2. pip install pandas
3. pip install nltk
4. pip install gensim
5. pip install tqdm
6. pip install openpyxl
7. python -m nltk.downloader punkt punkt_tab
8. **To use Word2Vec the weights must be downloaded from https://github.com/mmihaltz/word2vec-googlenews-vectors**
