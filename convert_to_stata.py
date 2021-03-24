import pandas as pd

load_path = "data/amazon/amazon_reviews2_regexd.json.gz"

data_loader = pd.read_json(load_path, lines=True, chunksize=int(12224024/2))

for i, df in enumerate(data_loader):

	column_order = ['review_num', 'gvkey', 'datadate', 'fyearq', 'fqtr', 'cik', 'year', 'month', 'cw_id',
	       'serial_no', 'brand', 'stars', 'asin', 'text', 'summary', 'vote',
	       'rank', 'price', 'category_id', 'num_words', 'neg_hits', 'pos_hits',
	       'qua_hits']

	df = df[column_order]

	save_path = "data/amazon/amazon_reviews2_regexd_" + str(i) + ".dta"
	df.to_stata(save_path, version=118)