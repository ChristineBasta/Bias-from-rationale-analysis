in_path = 'E:\\NLP\\dataset\\yelp_culture\\data_013.txt'
out_path = 'E:\\NLP\\dataset\\yelp_culture\\yelp_culture.txt'
cate_exist = 0
out_file = open(out_path, 'a+')

with open(in_path,'r',encoding='UTF8') as file:
    for line in file:
        category = line.split('\t')[4].lower()
        review = line.split('\t')[-1]
        if ('chinese' in category) or ('asian fusion' in category):
            category_num = 0
            cate_exist = 1
        if 'mexican' in category:
            category_num = 1
            cate_exist =1
        if ('american' in category) or ('canadian' in category):
            category_num = 2
            cate_exist =1
        if 'thai' in category:
            category_num = 3
            cate_exist =1
        if 'korean' in category:
            category_num = 4
            cate_exist =1
        if 'japanese' in category:
            category_num = 5
            cate_exist =1
        if 'italian' in category:
            category_num = 6
            cate_exist =1
        if 'french' in category:
            category_num = 7
            cate_exist =1
        if cate_exist >1 :
            out_file.write(category_num)
            out_file.write('\t')
            out_file.write(review + '\n')
            out_file.flush()
