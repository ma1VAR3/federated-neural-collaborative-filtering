import os
import pandas as pd
import numpy as np

class Loader():

    def __init__(self):
        pass

    def load_dataset(self):
        
        df = pd.read_csv('./ratings.csv')
        print(df.columns)
        df = df.drop(['timestamp'], axis=1)
        df.columns = ['user', 'item', 'rating']
        df = df.dropna()
        df = df.loc[df.rating != 0]


        df_count = df.groupby(['user']).count()
        df['count'] = df.groupby('user')['user'].transform('count')
        df = df[df['count'] > 1]

        
        df['user_id'] = df['user'].astype("category").cat.codes
        df['item_id'] = df['item'].astype("category").cat.codes

        # lookup 테이블 생성
        item_lookup = df[['item_id', 'item']].drop_duplicates()
        item_lookup['item_id'] = item_lookup.item_id.astype(str)

        # train, test 데이터 생성
        df = df[['user_id', 'item_id', 'rating']]
        

        # 전체 user, item 리스트 생성
        users = list(np.sort(df.user_id.unique()))
        items = list(np.sort(df.item_id.unique()))

        # train user, item 리스트 생성
        
        
        # 각 user 마다 negative item 생성
        

        return df, items, users
        # return uids, iids, df_train, df_test, df_neg, users, items, item_lookup

    def get_samples(self, df, items):

        
        rows = df['user_id'].astype(int)
        cols = df['item_id'].astype(int)
        values = list(df.rating)

        uids = np.array(rows.tolist())
        iids = np.array(cols.tolist())

        df_neg = self.get_negatives(uids, iids, items, df)

        return uids, iids

    def get_negatives(self, uids, iids, items, df_test):
        """
        negative item 리스트 생성함수
        """
        negativeList = []
        test_u = df_test['user_id'].values.tolist()
        test_i = df_test['item_id'].values.tolist()

        test_ratings = list(zip(test_u, test_i))  # test (user, item)세트
        zipped = set(zip(uids, iids))             # train (user, item)세트

        for (u, i) in test_ratings:

            negatives = []
            negatives.append((u, i))
            for t in range(100):
                j = np.random.randint(len(items))     # neg_item j 1개 샘플링
                while (u, j) in zipped:               # j가 train에 있으면 다시뽑고, 없으면 선택
                    j = np.random.randint(len(items))
                negatives.append(j)
            negativeList.append(negatives) # [(0,pos), neg, neg, ...]

        df_neg = pd.DataFrame(negativeList)

        return df_neg

    def mask_first(self, x):

        result = np.ones_like(x)
        result[0] = 0  # [0,1,1,....]

        return result

    def train_test_split(self, df):
        """
        train, test 나누는 함수
        """
        df_test = df.copy(deep=True)
        df_train = df.copy(deep=True)

        # df_test
        # user_id와 holdout_item_id(user가 플레이한 아이템 중 1개)뽑기
        df_test = df_test.groupby(['user_id']).first()
        df_test['user_id'] = df_test.index
        df_test = df_test[['user_id', 'item_id', 'plays']]
        df_test = df_test.reset_index(drop=True)

        # df_train
        # user_id 리스트에 make_first()적용
        mask = df.groupby(['user_id'])['user_id'].transform(self.mask_first).astype(bool)
        df_train = df.loc[mask]

        return df_train, df_test

    def get_train_instances(self, uids, iids, num_neg, num_items):
        """
        모델에 사용할 train 데이터 생성 함수
        """
        user_input, item_input, labels = [],[],[]
        zipped = set(zip(uids, iids)) # train (user, item) 세트

        for (u, i) in zip(uids, iids):

            # pos item 추가
            user_input.append(u)  # [u]
            item_input.append(i)  # [pos_i]
            labels.append(1)      # [1]

            # neg item 추가
            for t in range(num_neg):

                j = np.random.randint(num_items)      # neg_item j num_neg 개 샘플링
                while (u, j) in zipped:               # u가 j를 이미 선택했다면
                    j = np.random.randint(num_items)  # 다시 샘플링

                user_input.append(u)  # [u1, u1,  u1,  ...]
                item_input.append(j)  # [pos_i, neg_j1, neg_j2, ...]
                labels.append(0)      # [1, 0,  0,  ...]

        return user_input, item_input, labels